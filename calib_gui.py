"""Calibration GUI for the two-camera piece-picking setup (NiceGUI).

Steps (tabs): 1 Setup -> 2 Lens (once per camera: checkerboard views, kept across sessions) -> 3 Table pose (the one
step per session: the board flat on the table, one photo gives both cameras' poses and the stereo geometry)
-> 4 3D view + live piece detection, with the colour thresholds to tune (saved for the runner).

Without a lens calibration the table photo also estimates each camera's focal length (no distortion): fine to try
things out, but calibrate the lenses once for the real thing.

Sources are two cameras, or in test mode image files that stand in for them: one left/right set per step
(calibration pairs, table pose, detection scene), defaulting to the Blender renders in synthetic/.
Captured pairs, settings and calib.json live in calib_data/, so a session can be resumed. In test mode the pairs and
calib.json live in synthetic/gui/ instead, so test images can never end up in the lab's calibration.

One camera (--one-camera, or the switch in Setup) tries the whole flow with a single camera, for testing only: it shows
in both views, calibrates as one camera, and positions come from the table plane because one view has no depth. Its
data stays in synthetic/gui/one_camera/, where the runner never looks.

Run:  venv/bin/python calib_gui.py             (opens its own app window, dark mode)
      venv/bin/python calib_gui.py --browser   (only serve on http://localhost:8080)
      venv/bin/python calib_gui.py --one-camera
"""
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from nicegui import app, run, ui

import pieces
import stereo_calib as sc
from cameras import (BASLER, DATA, LENS_FILE, SETTINGS_PATH, STAGES, TEST_DATA, TEST_FILES, Source, list_cameras,
                     load_settings)

PREVIEW_W = 640
PORT = 8080
BROWSERS = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser", "brave", "microsoft-edge-stable")
HEX = {"red": "#e53935", "blue": "#1e40af", "green": "#2e9d4f"}


@dataclass
class Pair:
    """One capture for the lens calibration: the board's corners in each camera, None where it wasn't seen."""
    name: str
    left: np.ndarray | None
    right: np.ndarray | None
    err_l: float | None = None
    err_r: float | None = None


class State:
    def __init__(self):
        self.settings = load_settings()
        self.sources = {}  # stage -> (left, right); with cameras every stage shares the same pair
        self.lock = threading.Lock()
        self.pairs: list[Pair] = []
        self.image_size = None
        self.calib = None
        self.loaded = None  # the data folder pairs and calib came from
        self.rect_maps = None
        self.views = None  # (TopView left, TopView right) for detection
        self.last_auto = 0.0
        self.lens = {}     # camera key -> saved lens calibration (lens.json)
        self.params = pieces.load_params()

    def lens_key(self, side):
        """Lens calibrations belong to a camera, not to a side: swapping the cameras keeps each one's."""
        return f"test:{side}" if self.settings["test_mode"] else str(self.settings[side])

    def lens_for(self, side, size):
        e = self.lens.get(self.lens_key(side))
        if e is None or tuple(e["image_size"]) != tuple(size):
            return None
        return sc.CameraCalib(np.array(e["K"]), np.array(e["dist"]), tuple(e["image_size"]), e["rms"])

    def save_lens(self, side, cam, views):
        self.lens[self.lens_key(side)] = {"K": np.asarray(cam.K).tolist(), "dist": np.ravel(cam.dist).tolist(),
                                          "image_size": list(cam.image_size), "rms": cam.rms, "views": views,
                                          "saved": datetime.now().isoformat(timespec="minutes")}
        (self.data / LENS_FILE).write_text(json.dumps(self.lens, indent=2))

    @property
    def params_path(self):
        return self.data / "detection.json"

    @property
    def data(self):
        if self.settings["test_mode"]:
            return TEST_DATA
        return TEST_DATA / "one_camera" if self.settings["one_camera"] else DATA

    @property
    def pairs_dir(self):
        return self.data / "pairs"

    def load_data(self):
        """Pairs and calibration of the current mode's folder (again when test mode was switched)."""
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.pairs.clear()
        self.loaded = self.data
        self.rect_maps = self.views = None
        self.calib = sc.StereoCalib.load(self.data / "calib.json") if (self.data / "calib.json").exists() else None
        self.lens = json.loads((self.data / LENS_FILE).read_text()) if (self.data / LENS_FILE).exists() else {}
        self.params = pieces.load_params(self.params_path)
        load_saved_pairs()
        if self.calib is not None and self.calib.T_left_table is not None:
            self.save_calib()  # builds the detection views

    @property
    def pattern(self):
        return int(self.settings["cols"]), int(self.settings["rows"])

    @property
    def square(self):
        return float(self.settings["square_mm"])

    def save_settings(self):
        SETTINGS_PATH.write_text(json.dumps(self.settings, indent=2))

    def pair(self, stage):
        return self.sources.get(stage, (None, None))

    def connect(self):
        """Open the cameras, or in test mode the image files of every stage. Returns a list of problems."""
        with self.lock:
            for src in {s for pair in self.sources.values() for s in pair}:
                src.release()
            self.sources = {}
            if self.loaded != self.data:
                self.load_data()
            w, h = int(self.settings["width"]), int(self.settings["height"])
            errors = []
            if self.settings["test_mode"]:
                for st in STAGES:
                    try:
                        self.sources[st] = tuple(Source(self.settings[f"test_{st}_{side}"], w, h) for side in ("left", "right"))
                    except RuntimeError as e:
                        errors.append(f"{STAGES[st]}: {e}")
            elif self.settings["one_camera"]:
                cam = Source(self.settings["left"], w, h)
                self.sources = {st: (cam, cam) for st in STAGES}
            else:
                pair = (Source(self.settings["left"], w, h), Source(self.settings["right"], w, h))
                self.sources = {st: pair for st in STAGES}
        self.save_settings()
        return errors

    def read(self, stage):
        with self.lock:
            left, right = self.pair(stage)
            if left is None:
                return None, None
            if left is right:  # one camera: a second grab would only throw a frame away
                left.grab()
                frame = left.retrieve()
                return frame, None if frame is None else frame.copy()
            left.grab()
            right.grab()
            return left.retrieve(), right.retrieve()

    def step(self, stage, d):
        with self.lock:
            for src in self.pair(stage):
                if src is not None and src.is_folder:
                    src.idx = (src.idx + d) % len(src.files)

    def save_calib(self):
        self.calib.save(self.data / "calib.json")
        self.rect_maps = None
        self.views = None
        if self.calib.T_left_table is not None:
            b = float(self.settings["board_mm"])
            self.views = (pieces.view_from_calib(self.calib, "left", b), pieces.view_from_calib(self.calib, "right", b))


S = State()


# ---------- image helpers ----------

def to_url(img, width=PREVIEW_W):
    if img is None:
        img = np.full((480, 640, 3), 40, np.uint8)
        cv2.putText(img, "no image", (230, 245), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    scale = width / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def quick_corners(img, pattern):
    """Fast low-res board check for the live preview; corners returned in full-res pixels."""
    scale = min(1.0, PREVIEW_W / img.shape[1])
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, c = cv2.findChessboardCorners(gray, pattern, flags=flags)
    return c.reshape(-1, 2) / scale if ok else None


def draw_board(img, corners, pattern, coverage=()):
    vis = img.copy()
    if coverage:
        layer = vis.copy()
        for c in coverage:
            cv2.fillConvexPoly(layer, cv2.convexHull(c.astype(np.int32)), (0, 160, 0))
        vis = cv2.addWeighted(layer, 0.25, vis, 0.75, 0)
    if corners is not None:
        cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2).astype(np.float32), True)
    return vis


def rectified_pair(left, right):
    if S.rect_maps is None:
        S.rect_maps = S.calib.rectify_maps(alpha=0.5)
    mlx, mly, mrx, mry, _ = S.rect_maps
    both = np.hstack([cv2.remap(left, mlx, mly, cv2.INTER_LINEAR), cv2.remap(right, mrx, mry, cv2.INTER_LINEAR)])
    for y in range(0, both.shape[0], max(both.shape[0] // 24, 10)):
        cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 255), 1)
    return both


# ---------- workers (run in a thread) ----------

def capture_pair(left, right):
    """The board needs to be in one camera only: each lens is calibrated on its own."""
    cl, cr = sc.find_corners(left, S.pattern), sc.find_corners(right, S.pattern)
    if cl is None and cr is None:
        return None
    name = f"{int(time.time() * 1000) % 10**9:09d}"
    cv2.imwrite(str(S.pairs_dir / f"{name}_left.png"), left)
    cv2.imwrite(str(S.pairs_dir / f"{name}_right.png"), right)
    S.image_size = (left.shape[1], left.shape[0])
    return Pair(name, cl, cr)


def import_all():
    """Import every pair from the two folder sources."""
    added = 0
    left_src, right_src = S.pair("calib")
    n = min(len(left_src.files), len(right_src.files))
    for i in range(n):
        left, right = cv2.imread(left_src.files[i]), cv2.imread(right_src.files[i])
        p = capture_pair(left, right)
        if p is not None:
            S.pairs.append(p)
            added += 1
    return added, n


def load_saved_pairs():
    for f in sorted(S.pairs_dir.glob("*_left.png")):
        name = f.name.removesuffix("_left.png")
        left, right = cv2.imread(str(f)), cv2.imread(str(S.pairs_dir / f"{name}_right.png"))
        if right is None:
            continue
        cl, cr = sc.find_corners(left, S.pattern), sc.find_corners(right, S.pattern)
        if cl is not None or cr is not None:
            S.pairs.append(Pair(name, cl, cr))
            S.image_size = (left.shape[1], left.shape[0])


def one_camera():
    return S.settings["one_camera"] and not S.settings["test_mode"]


def calibrate_lens():
    """Each camera's lens from the views it saw the board in; saved for this camera, for every later session.
    Returns {side: (rms, views)}. The table pose, if captured, is recomputed with the new lenses."""
    out = {}
    for side in ("left",) if one_camera() else ("left", "right"):
        views = [p for p in S.pairs if getattr(p, side) is not None]
        if len(views) < 5:
            continue
        cam, errs = sc.calibrate_camera([getattr(p, side) for p in views], S.image_size, S.pattern, S.square)
        for p, e in zip(views, errs):
            setattr(p, "err_l" if side == "left" else "err_r", e)
        S.save_lens(side, cam, len(views))
        out[side] = (cam.rms, len(views))
    tl, tr = S.data / "table_left.png", S.data / "table_right.png"
    if out and tl.exists() and tr.exists():
        capture_table(cv2.imread(str(tl)), cv2.imread(str(tr)), save_images=False)
    return out


def capture_table(left, right, save_images=True):
    """The session calibration from one photo of the board flat on the table (stereo_calib.stereo_from_table).
    Returns None if a camera doesn't see the board, else (check, lens) with lens: side -> "saved" / "estimated"."""
    same = left is right or one_camera()
    cl = sc.find_corners(left, S.pattern)
    cr = cl if same else sc.find_corners(right, S.pattern)
    if cl is None or cr is None:
        return None
    cl = sc.canonical_corners(left, cl, S.pattern)
    cr = cl if same else sc.canonical_corners(right, cr, S.pattern)
    cams, lens = [], {}
    for side, img, c in (("left", left, cl), ("right", right, cr)):
        size = (img.shape[1], img.shape[0])
        cam = S.lens_for("left" if same else side, size)
        lens[side] = "saved" if cam is not None else "estimated"
        cams.append(cam if cam is not None else sc.estimate_camera(c, size, S.pattern, S.square))
    if same:
        cams[1] = cams[0]
    S.calib, check = sc.stereo_from_table(cl, cr, *cams, S.pattern, S.square, same_camera=same)
    S.calib.cameras = None if S.settings["test_mode"] else {"left": S.settings["left"], "right": S.settings["right"]}
    S.save_calib()
    if save_images:
        cv2.imwrite(str(S.data / "table_left.png"), left)
        cv2.imwrite(str(S.data / "table_right.png"), right)
    return check, lens


def table_text(check, lens):
    lines = [f"Lens: left {lens['left']}, right {lens['right']}"
             + (" (estimated from this photo, no distortion: calibrate the lenses once for full accuracy)"
                if "estimated" in lens.values() else ""),
             f"Board reprojection: left {check['left_px']:.2f} px, right {check['right_px']:.2f} px"]
    if check["mean_mm"] is not None:
        lines.append(f"Stereo check, board corners: {check['mean_mm']:.2f} mm mean, {check['max_mm']:.2f} mm max "
                     f"· baseline {S.calib.baseline_mm:.0f} mm")
    return "\n".join(lines + camera_pose_text().split(" · "))


def detect_both(left, right, mask_color=None):
    """Both cameras' detections, paired. Pieces only one camera sees are drawn grey with a "?": the runner ignores
    them (that's where false detections come from)."""
    vl, vr = S.views
    params = S.params
    if one_camera():   # one camera: one detection, and no disparity to triangulate from: the table plane
        pl = pieces.detect_pieces(left, vl, params=params)
        pr, matched = pl, [(p, np.array(p.table_mm)) for p in pl]
    else:
        with ThreadPoolExecutor(2) as pool:
            pl, pr = pool.map(lambda iv: pieces.detect_pieces(iv[0], iv[1], params=params), ((left, vl), (right, vr)))
        matched = pieces.matched_pieces(S.calib, pl, pr)
    lone_l = [p for p, st in matched if st is None]
    paired_r = set()
    for p, st in matched:
        if st is not None:   # the right detection it was paired with: same colour, nearest on the table
            paired_r.add(min((q for q in pr if q.color == p.color),
                             key=lambda q: np.hypot(q.table_mm[0] - p.table_mm[0], q.table_mm[1] - p.table_mm[1])).table_mm)
    lone_r = [q for q in pr if q.table_mm not in paired_r] if not one_camera() else []
    if mask_color:
        left, right = pieces.mask_view(left, params, mask_color), pieces.mask_view(right, params, mask_color)
    return pl, pr, pieces.draw(left, pl, vl, lone=lone_l), pieces.draw(right, pr, vr, lone=lone_r), matched


# ---------- UI ----------

def cameras_found():
    cams = list_cameras()
    return "Cameras found: " + ", ".join(f"{i} = {name}" for i, name in cams) if cams else "No cameras found."


def camera_pose_text():
    """Height above the table and tilt from vertical for both cameras."""
    out = []
    for name in ("left", "right"):
        _, T = pieces.camera_table_pose(S.calib, name)
        Tinv = np.linalg.inv(T)
        z_axis = Tinv[:3, 2]  # camera optical axis in table frame
        tilt = np.degrees(np.arccos(np.clip(-z_axis[2], -1, 1)))
        out.append(f"{name}: height {Tinv[2, 3]:.0f} mm, tilt {tilt:.1f}°")
    return " · ".join(out)


async def open_app_window():
    """Open the UI in a standalone Chrome/Chromium "app" window (no tabs or address bar) and quit when it closes.

    pywebview's Qt backend can't create a WebGL context with the NVIDIA driver here, which breaks the 3D view;
    Chromium with SwiftShader can. Falls back to the default browser.
    """
    url = f"http://localhost:{PORT}"
    exe = next(filter(None, map(shutil.which, BROWSERS)), None)
    if exe is None:
        webbrowser.open(url)
        return
    profile = Path.home() / ".cache" / "calib-gui-window"  # own profile = own process, so wait() returns on close
    proc = subprocess.Popen([exe, f"--app={url}", "--window-size=1280,720", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--force-dark-mode",
                             # Chrome can't get a GPU WebGL context from the NVIDIA driver here either; SwiftShader
                             # (software WebGL) renders the small 3D scene fine
                             "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    await run.io_bound(proc.wait)
    app.shutdown()


def root():
    ui.page_title("Stereo calibration")
    status = {}
    next_buttons = []  # (button, hint label, is_ready)
    view = {"stage": "calib", "frames": (None, None)}  # per window: which image set this window shows, last frames

    with ui.header().classes("items-center bg-slate-800"):
        ui.label("Stereo calibration").classes("text-lg font-medium")
        ui.space()
        status["src"] = ui.label().classes("text-sm opacity-80")
        status["chip_cal"] = ui.chip("not calibrated", icon="photo_camera").props("dense")
        status["chip_table"] = ui.chip("no table pose", icon="table_restaurant").props("dense")

    def refresh_status():
        left, right = S.pair(view["stage"])
        prefix = f"test · {STAGES[view['stage']]} · " if S.settings["test_mode"] else ""
        status["src"].text = "" if left is None else f"{prefix}L: {left.label()}   R: {right.label()}"
        sides = ("left",) if one_camera() else ("left", "right")
        have = [S.lens_key(sd) in S.lens for sd in sides]
        status["chip_cal"].text = "lens: saved" if all(have) else "lens: saved for one camera" if any(have) \
            else "lens: estimated from the table photo"
        status["chip_cal"].props(f"color={'positive' if all(have) else 'grey'} text-color=white")
        has_table = S.calib is not None and S.calib.T_left_table is not None
        status["chip_table"].text = "table pose ✓" if has_table else "no table pose"
        status["chip_table"].props(f"color={'positive' if has_table else 'grey'} text-color=white")
        for btn, hint, ready in next_buttons:
            ok = ready()
            btn.set_enabled(ok)
            hint.set_visibility(not ok)

    def next_button(target, title, ready, todo):
        """'Next' button at the bottom of a tab, enabled once the step is done."""
        with ui.row().classes("w-full items-center justify-end gap-3 mt-4"):
            hint = ui.label(todo).classes("text-sm opacity-60")
            btn = ui.button(f"Next: {title}", on_click=lambda: tabs.set_value(target)).props("icon-right=arrow_forward")
        next_buttons.append((btn, hint, ready))

    with ui.tabs().classes("w-full") as tabs:
        t_setup = ui.tab("1 · Setup")
        t_stereo = ui.tab("2 · Lens (once)")
        t_table = ui.tab("3 · Table pose")
        t_3d = ui.tab("4 · 3D & detection")

    with ui.tab_panels(tabs, value=t_setup).classes("w-full"):
        # ---- 1 Setup ----
        with ui.tab_panel(t_setup):
            with ui.row().classes("w-full no-wrap gap-4 items-stretch"):
                with ui.card().classes("flex-[3] min-w-0"):
                    with ui.row().classes("items-center w-full"):
                        ui.label("Sources").classes("text-base font-medium")
                        ui.space()
                        ui.switch("Test mode: image files instead of cameras").bind_value(S.settings, "test_mode")
                    with ui.column().classes("w-full gap-1").bind_visibility_from(S.settings, "test_mode", value=False):
                        found = ui.label(cameras_found()).classes("text-sm opacity-70")

                        def use_basler():
                            found.text = cameras_found()
                            cams = [spec for spec, _ in list_cameras() if spec.startswith(BASLER)]
                            if cams:
                                S.settings.update(left=cams[0], right=cams[-1])
                            ui.notify(f"{len(cams)} Basler camera(s) found", type="positive" if len(cams) == 2 else "warning")
                        ui.button("Use the Basler cameras", icon="photo_camera", on_click=use_basler).props("outline dense")
                        ui.switch("One camera (testing only): uses the left camera for both views") \
                            .bind_value(S.settings, "one_camera")
                        async def swap():
                            S.settings["left"], S.settings["right"] = S.settings["right"], S.settings["left"]
                            if S.sources and not S.settings["test_mode"]:
                                await do_connect()
                            ui.notify("Cameras swapped" + (": capture the table pose again" if S.calib is not None else ""))
                        with ui.row().classes("w-full items-center no-wrap"):
                            ui.input("Left camera").bind_value(S.settings, "left").classes("flex-1")
                            ui.button(icon="swap_horiz", on_click=swap).props("flat round") \
                                .tooltip("Swap the left and right camera")
                            ui.input("Right camera").bind_value(S.settings, "right").classes("flex-1")
                        with ui.row():
                            ui.number("Width", format="%d").bind_value(S.settings, "width").classes("w-28")
                            ui.number("Height", format="%d").bind_value(S.settings, "height").classes("w-28")
                    with ui.column().classes("w-full gap-1").bind_visibility_from(S.settings, "test_mode"):
                        ui.label("Each step uses its own image files, picked by the active tab. Paths are relative to the "
                                 "project folder; a glob (*.png) gives several images, step through them with ◀ ▶.") \
                            .classes("text-sm opacity-70")
                        for st, title in STAGES.items():
                            with ui.row().classes("w-full items-center no-wrap"):
                                ui.label(title).classes("w-36 text-sm")
                                for i, side in enumerate(("left", "right")):
                                    ui.input(f"{side.capitalize()} images", placeholder=TEST_FILES[st][i]) \
                                        .bind_value(S.settings, f"test_{st}_{side}").classes("flex-1")

                        def use_synthetic():
                            S.settings.update({f"test_{st}_{side}": TEST_FILES[st][i]
                                               for st in STAGES for i, side in enumerate(("left", "right"))})
                            S.settings.update(cols=9, rows=6, square_mm=25.0, board_mm=0.8)
                            ui.notify("Synthetic test set selected (9 x 6 board, 25 mm squares, 0.8 mm board)")
                        ui.button("Use the synthetic test set", icon="science", on_click=use_synthetic).props("outline dense")
                    async def do_connect():
                        try:
                            errors = await run.io_bound(S.connect)
                        except Exception as e:
                            ui.notify(str(e), type="negative")
                            return
                        found.text = cameras_found()
                        for e in errors:
                            ui.notify(e, type="warning")
                        if S.sources:
                            ui.notify("Test images loaded" if S.settings["test_mode"] else "Cameras connected", type="positive")
                        refresh_pairs()
                        refresh_status()
                        draw_scene()
                        if S.calib is not None:
                            refresh_result()
                    ui.space()
                    ui.button("Connect", icon="link", on_click=do_connect)

                with ui.card().classes("flex-[2] min-w-0"):
                    ui.label("Checkerboard").classes("text-base font-medium")
                    ui.label("Inner corners, i.e. squares minus one. The OpenCV sample board is 9 x 6.").classes("text-sm opacity-70")
                    with ui.grid(columns=2).classes("w-full gap-x-4"):
                        ui.number("Inner corners (cols)", format="%d").bind_value(S.settings, "cols")
                        ui.number("Inner corners (rows)", format="%d").bind_value(S.settings, "rows")
                        ui.number("Square size (mm)", format="%.2f").bind_value(S.settings, "square_mm")
                        ui.number("Board thickness (mm)", format="%.1f").bind_value(S.settings, "board_mm") \
                            .tooltip("Thickness of the board when it lies on the table for the table pose; pieces stand this much lower")
                    ui.label("Measure the square size with a caliper on the printed board; printers rarely print at exactly 100%.") \
                        .classes("text-sm text-amber-400")
                    ui.label("Board thickness: the board lies on the table for the table pose, so the pieces stand this "
                             "much lower than the board surface.").classes("text-sm opacity-70")
            next_button(t_table, "Table pose", lambda: bool(S.sources), "Connect the sources first")

        # ---- 2 Stereo ----
        with ui.tab_panel(t_stereo):
            with ui.row().classes("w-full no-wrap gap-4 items-start"):
                with ui.column().classes("flex-[2] min-w-0 gap-2"):
                    with ui.row().classes("w-full no-wrap gap-2") as st_live:
                        st_left = ui.interactive_image().classes("w-1/2")
                        st_right = ui.interactive_image().classes("w-1/2")
                    st_rect = ui.interactive_image().classes("w-full")
                    st_rect.visible = False
                    with ui.row().classes("items-center gap-2"):
                        btn_capture = ui.button("Capture pair", icon="photo_camera")
                        auto = ui.switch("Auto-capture").tooltip("Capture when the board is seen by both cameras at a new position")
                        ui.button(icon="chevron_left", on_click=lambda: S.step(view["stage"], -1)).props("flat dense").tooltip("Previous image")
                        ui.button(icon="chevron_right", on_click=lambda: S.step(view["stage"], 1)).props("flat dense").tooltip("Next image")
                        btn_import = ui.button("Import all test images", icon="download").props("outline") \
                            .bind_visibility_from(S.settings, "test_mode")
                    with ui.row().classes("items-center gap-4"):
                        st_found = ui.label().classes("text-sm")
                        rect_sw = ui.switch("Show rectified").tooltip("After calibration: matching points must lie on the same yellow line")
                    ui.label("Once per camera, kept across sessions (moving a camera doesn't change its lens; refocusing "
                             "or another resolution does). Move the board through each camera's view: near/far, tilted "
                             "~30° in different directions, corners of the image too. One camera at a time is fine. "
                             "15-25 views per camera. Green = coverage so far.").classes("text-sm opacity-70")
                with ui.card().classes("flex-1 min-w-0 self-stretch"):
                    with ui.row().classes("items-center w-full"):
                        pairs_title = ui.label().classes("font-medium")
                        ui.space()
                        btn_clear = ui.button(icon="delete_sweep").props("flat dense color=negative").tooltip("Delete all pairs")
                    pairs_list = ui.column().classes("w-full gap-0 overflow-auto max-h-[calc(100vh-420px)]")
                    ui.space()
                    btn_calib = ui.button("Calibrate lenses", icon="tune").classes("w-full")
                    next_button(t_table, "Table pose", lambda: True, "")
            with ui.card().classes("w-full"):
                ui.label("Result").classes("font-medium")
                result_md = ui.markdown("Not calibrated yet.")
                err_chart = ui.echart({
                    "tooltip": {"trigger": "axis"}, "legend": {"data": ["left", "right"]},
                    "xAxis": {"type": "category", "data": []}, "yAxis": {"type": "value", "name": "px"},
                    "series": [{"name": n, "type": "bar", "data": []} for n in ("left", "right")],
                }).classes("w-full h-64")

            def refresh_pairs():
                pairs_title.text = f"Captured pairs: {len(S.pairs)}"
                pairs_list.clear()
                with pairs_list:
                    for i, p in enumerate(S.pairs):
                        with ui.row().classes("items-center w-full no-wrap text-sm min-h-[30px]"):
                            ui.label(f"#{i + 1}").classes("w-8")
                            fmt = lambda c, e: "–" if c is None else "✓" if e is None else f"{e:.2f} px"
                            errs = f"L {fmt(p.left, p.err_l)} · R {fmt(p.right, p.err_r)}"
                            bad = max(p.err_l or 0, p.err_r or 0) > 1.0
                            ui.label(errs).classes("flex-1" + (" text-red-400" if bad else ""))
                            ui.button(icon="close", on_click=lambda p=p: delete_pair(p)).props("flat dense size=sm")
                if any(p.err_l is not None or p.err_r is not None for p in S.pairs):
                    err_chart.options["xAxis"]["data"] = [f"#{i + 1}" for i in range(len(S.pairs))]
                    for s, key in zip(err_chart.options["series"], ("err_l", "err_r")):
                        s["data"] = [round(getattr(p, key) or 0, 3) for p in S.pairs]
                    err_chart.update()
                refresh_result()

            def refresh_result():
                rows = ["| camera | fx | fy | cx | cy | RMS px | views | saved | distortion |", "|---|---|---|---|---|---|---|---|---|"]
                for side in ("left", "right"):
                    e = S.lens.get(S.lens_key(side))
                    if e is None:
                        rows.append(f"| {side} ({S.lens_key(side)}) | not calibrated: the table photo estimates it |||||||")
                        continue
                    K, d = np.asarray(e["K"]), np.ravel(e["dist"])
                    rows.append(f"| {side} ({S.lens_key(side)}) | {K[0, 0]:.1f} | {K[1, 1]:.1f} | {K[0, 2]:.1f} | {K[1, 2]:.1f} "
                                f"| {e['rms']:.3f} | {e['views']} | {e['saved']} | {', '.join(f'{x:.3f}' for x in d[:5])} |")
                result_md.content = "\n".join(rows + ["", "Good: RMS under 0.5 px."])

            def delete_pair(p):
                S.pairs.remove(p)
                for side in ("left", "right"):
                    (S.pairs_dir / f"{p.name}_{side}.png").unlink(missing_ok=True)
                refresh_pairs()

            def clear_pairs():
                for p in list(S.pairs):
                    delete_pair(p)

            async def do_capture(_=None, quiet=False):
                left, right = view["frames"]
                if left is None:
                    ui.notify("No frames; connect the sources first", type="warning")
                    return
                p = await run.io_bound(capture_pair, left, right)
                if p is None:
                    if not quiet:
                            ui.notify("Board not found in either image", type="warning")
                    return
                S.pairs.append(p)
                refresh_pairs()
                ui.notify(f"Captured pair #{len(S.pairs)}", type="positive")

            async def do_import():
                left_src, right_src = S.pair("calib")
                if left_src is None or not (left_src.is_folder and right_src.is_folder):
                    ui.notify("Only in test mode: switch it on in Setup", type="warning")
                    return
                n = ui.notification("Detecting corners...", spinner=True, timeout=None)
                added, total = await run.io_bound(import_all)
                n.dismiss()
                ui.notify(f"Imported {added} of {total} pairs (board found in both)", type="positive")
                refresh_pairs()

            async def do_calibrate():
                n = ui.notification("Calibrating...", spinner=True, timeout=None)
                try:
                    out = await run.io_bound(calibrate_lens)
                    if not out:
                        ui.notify("Each camera needs at least 5 views of the board (15+ recommended)", type="warning")
                    else:
                        ui.notify("Lens saved: " + ", ".join(f"{sd} RMS {r:.3f} px ({v} views)" for sd, (r, v) in out.items())
                                  + (" · table pose updated" if S.calib is not None else ""), type="positive")
                except Exception as e:
                    ui.notify(f"Calibration failed: {e}", type="negative")
                finally:
                    n.dismiss()
                refresh_pairs()
                refresh_status()
                draw_scene()

            btn_capture.on_click(do_capture)
            btn_import.on_click(do_import)
            btn_clear.on_click(clear_pairs)
            btn_calib.on_click(do_calibrate)

        # ---- 3 Table pose ----
        with ui.tab_panel(t_table):
            with ui.row().classes("w-full no-wrap gap-4 items-start"):
                with ui.column().classes("flex-[2] min-w-0 gap-2"):
                    with ui.row().classes("w-full no-wrap gap-2"):
                        tb_left = ui.interactive_image().classes("w-1/2")
                        tb_right = ui.interactive_image().classes("w-1/2")
                    with ui.row().classes("items-center gap-2"):
                        tb_found = ui.label().classes("text-sm")
                        ui.button(icon="chevron_left", on_click=lambda: S.step(view["stage"], -1)).props("flat dense")
                        ui.button(icon="chevron_right", on_click=lambda: S.step(view["stage"], 1)).props("flat dense")
                with ui.card().classes("flex-1 min-w-0 self-stretch"):
                    ui.label("Table pose").classes("font-medium")
                    ui.label("The one step per session: lay the board flat on the table in the robot's workspace, seen by "
                             "both cameras, and capture. That one photo gives both cameras' poses and the stereo geometry. "
                             "The origin is the inner corner next to the board's dark corner square (the axes show it), "
                             "x along the rows, z up. Set the board thickness in Setup. Redo it whenever a camera moved.") \
                        .classes("text-sm opacity-70")
                    btn_table = ui.button("Capture table pose", icon="table_restaurant").classes("w-full")
                    tb_result = ui.label().classes("text-sm whitespace-pre-line")
                    ui.space()
                    card_bottom = ui.column().classes("w-full")

            async def do_table():
                left, right = view["frames"]
                if left is None:
                    return
                res = await run.io_bound(capture_table, left, right)
                if res is None:
                    ui.notify("Board not found in both images", type="warning")
                    return
                tb_result.text = "Saved.\n" + table_text(*res)
                ui.notify("Table pose saved", type="positive")
                refresh_status()
                draw_scene()
            btn_table.on_click(do_table)
            with card_bottom:
                next_button(t_3d, "3D & detection", lambda: S.calib is not None and S.calib.T_left_table is not None,
                            "Capture the table pose first")

        # ---- 4 3D & detection ----
        with ui.tab_panel(t_3d):
            with ui.row().classes("w-full no-wrap gap-4 items-start"):
                with ui.column().classes("w-1/2 min-w-0"):
                    scene = ui.scene(grid=(2, 40), background_color="#18181b").classes("w-full h-[calc(100vh-235px)]")
                    no_webgl = ui.label("The 3D view needs WebGL, which this browser can't provide. Start the app with "
                                        "`python calib_gui.py` (its window uses software WebGL).") \
                        .classes("text-amber-400 text-sm")
                    no_webgl.visible = False
                    ui.label("Grid: 5 cm cells. Table frame axes: x red, y green, z blue. Units in the view are metres.") \
                        .classes("text-xs opacity-60")
                with ui.column().classes("w-1/2 min-w-0"):
                    with ui.row().classes("items-center"):
                        det_sw = ui.switch("Live detection")
                        mask_sel = ui.select({"": "no mask", **{c: f"{c} mask" for c in pieces.COLORS}}, value="") \
                            .props("dense outlined").classes("w-32").tooltip("Show only the pixels that pass a colour's thresholds")
                        ui.button(icon="chevron_left", on_click=lambda: S.step(view["stage"], -1)).props("flat")
                        ui.button(icon="chevron_right", on_click=lambda: S.step(view["stage"], 1)).props("flat")
                    with ui.row().classes("w-full no-wrap gap-2"):
                        det_left = ui.interactive_image().classes("w-1/2")
                        det_right = ui.interactive_image().classes("w-1/2")
                    det_table = ui.table(columns=[
                        {"name": "color", "label": "colour", "field": "color"},
                        {"name": "plane", "label": "x, y (plane, left cam) mm", "field": "plane"},
                        {"name": "stereo", "label": "x, y, z (stereo) mm", "field": "stereo"},
                        {"name": "diff", "label": "Δ mm", "field": "diff"},
                    ], rows=[]).props("dense flat virtual-scroll").classes("w-full").style("max-height: calc(100vh - 490px)")
                    with ui.expansion("Sensitivity (colour thresholds)", icon="tune").classes("w-full"):
                        tuning = ui.column().classes("w-full gap-1")

    def build_tuning():
        """Sliders for pieces.py's thresholds. They act at once; Save keeps them for the runner too."""
        tuning.clear()
        P = S.params
        def changed(*_):
            det_cache.clear()
        with tuning:
            ui.label("Grey pieces with a ? in the images: only one camera sees them. Missing pieces: widen the hue, "
                     "lower S / V min (use the mask to see what passes). Too many: the reverse, or raise min fill / rim.") \
                .classes("text-xs opacity-70")
            for name, c in P["colors"].items():
                with ui.row().classes("w-full items-center no-wrap gap-3"):
                    ui.label(name).classes("w-12 text-sm").style(f"color: {HEX[name]}")
                    lo, hi = (-60, 60) if name == "red" else (0, 180)
                    hue = ui.range(min=lo, max=hi, value={"min": c["h"][0], "max": c["h"][1]}).props("label").classes("flex-1")
                    hue.on_value_change(lambda e, c=c: (c.update(h=(int(e.value["min"]), int(e.value["max"]))), changed()))
                    ui.label("S≥").classes("text-xs")
                    ui.slider(min=0, max=255, value=c["s"], on_change=lambda e, c=c: (c.update(s=int(e.value)), changed())) \
                        .props("label").classes("w-24")
                    ui.label("V≥").classes("text-xs")
                    ui.slider(min=0, max=255, value=c["v"], on_change=lambda e, c=c: (c.update(v=int(e.value)), changed())) \
                        .props("label").classes("w-24")
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                for key, label in (("min_fill", "min fill"), ("min_rim", "min rim")):
                    ui.label(label).classes("text-sm")
                    ui.slider(min=0.2, max=0.95, step=0.05, value=P[key],
                              on_change=lambda e, key=key: (P.update({key: float(e.value)}), changed())).props("label").classes("flex-1")
            with ui.row().classes("gap-2"):
                def save():
                    pieces.save_params(S.params_path, S.params)
                    ui.notify(f"Saved to {S.params_path.relative_to(Path(__file__).parent)}: the runner uses it", type="positive")
                def reset():
                    S.params = pieces.load_params()
                    changed()
                    build_tuning()
                ui.button("Save", icon="save", on_click=save).props("dense")
                ui.button("Reset to defaults", icon="restart_alt", on_click=reset).props("dense flat")

    scene_objs = {"static": [], "pieces": []}

    def draw_scene():
        """Table frame, board, camera frustums."""
        if S.calib is None or S.calib.T_left_table is None:
            return
        for o in scene_objs["static"]:
            o.delete()
        scene_objs["static"].clear()
        m = 1e-3
        cols, rows = S.pattern
        sq = S.square
        with scene:
            add = scene_objs["static"].append
            add(scene.axes_helper(0.1))
            w, h = (cols - 1) * sq * m, (rows - 1) * sq * m
            for a, b in (((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))):
                add(scene.line([a[0], a[1], 0], [b[0], b[1], 0]).material("#d4d4d8"))
            for name, color in (("left", "#d97706"), ("right", "#7c3aed")):
                cam, T = pieces.camera_table_pose(S.calib, name)
                Tinv = np.linalg.inv(T)
                C = Tinv[:3, 3] * m
                Kinv = np.linalg.inv(np.asarray(cam.K))
                wpx, hpx = cam.image_size
                depth = 80.0
                corners = [Tinv[:3, :3] @ (Kinv @ np.array([u, v, 1.0]) * depth) * m + C
                           for u, v in ((0, 0), (wpx, 0), (wpx, hpx), (0, hpx))]
                for i in range(4):
                    add(scene.line(C.tolist(), corners[i].tolist()).material(color))
                    add(scene.line(corners[i].tolist(), corners[(i + 1) % 4].tolist()).material(color))
                add(scene.sphere(0.008).move(*map(float, C)).material(color))
                add(scene.text(name, "background: rgba(0,0,0,.6); color: #fff; padding: 1px 4px; border-radius: 3px").move(*map(float, C)))
        Cl = np.linalg.inv(pieces.camera_table_pose(S.calib, "left")[1])[:3, 3] * m
        scene.move_camera(x=float(Cl[0]) + 0.4, y=float(Cl[1]) - 0.6, z=float(Cl[2]) + 0.2, look_at_x=0.1, look_at_y=0.1, look_at_z=0)

    def draw_pieces(rows):
        for o in scene_objs["pieces"]:
            o.delete()
        scene_objs["pieces"].clear()
        m = 1e-3
        with scene:
            for p, st in rows:
                x, y, z = p.table_mm if st is None else (st[0], st[1], p.table_mm[2])
                cyl = scene.cylinder(pieces.PIECE_D_MM / 2 * m, pieces.PIECE_D_MM / 2 * m, pieces.PIECE_H_MM * m, 32)
                cyl.rotate(np.pi / 2, 0, 0).move(float(x) * m, float(y) * m, float(z - pieces.PIECE_H_MM / 2) * m) \
                    .material(HEX[p.color], opacity=1.0 if st is not None else 0.25)
                scene_objs["pieces"].append(cyl)

    # ---- live loop ----
    busy = {"v": False}
    coverage_cache = {}
    det_cache = {}

    tab_stage = {t_stereo.props["name"]: "calib", t_table.props["name"]: "table", t_3d.props["name"]: "scene",
}

    async def tick():
        tab = tabs.value if isinstance(tabs.value, str) else tabs.value.props["name"]
        view["stage"] = tab_stage.get(tab, view["stage"])  # in test mode every step has its own images
        refresh_status()
        if busy["v"] or S.pair(view["stage"])[0] is None:
            return
        busy["v"] = True
        try:
            frames = await run.io_bound(S.read, view["stage"])  # None while shutting down
            if frames is None or frames[0] is None or frames[1] is None:
                return
            left, right = view["frames"] = frames
            if tab in (t_stereo.props["name"], t_table.props["name"]):
                stereo_tab = tab == t_stereo.props["name"]
                if stereo_tab and rect_sw.value and S.calib is not None:
                    both = await run.io_bound(rectified_pair, left, right)
                    st_rect.set_source(to_url(both, 1280))
                    st_rect.visible, st_live.visible = True, False
                    return
                st_rect.visible, st_live.visible = False, True
                corners = await run.io_bound(lambda: (quick_corners(left, S.pattern), quick_corners(right, S.pattern)))
                if corners is None:
                    return
                cl, cr = corners
                found = f"board: left {'✓' if cl is not None else '✗'} · right {'✓' if cr is not None else '✗'}"
                if stereo_tab:
                    key = len(S.pairs)
                    if coverage_cache.get("key") != key:
                        coverage_cache.update(key=key, l=[p.left for p in S.pairs if p.left is not None],
                                              r=[p.right for p in S.pairs if p.right is not None])
                    vl = draw_board(left, cl, S.pattern, coverage_cache["l"])
                    vr = draw_board(right, cr, S.pattern, coverage_cache["r"])
                    st_left.set_source(to_url(vl))
                    st_right.set_source(to_url(vr))
                    st_found.text = found
                    if auto.value and (cl is not None or cr is not None) and time.time() - S.last_auto > 1.5:
                        side, c = ("left", cl) if cl is not None else ("right", cr)
                        centre = c.mean(0)
                        if all(getattr(p, side) is None or np.linalg.norm(centre - getattr(p, side).mean(0)) > 0.08 * left.shape[1]
                               for p in S.pairs):
                            S.last_auto = time.time()
                            await do_capture(quiet=True)
                else:
                    vl, vr = draw_board(left, cl, S.pattern), draw_board(right, cr, S.pattern)
                    if S.calib is not None and S.calib.T_left_table is not None:
                        for name, img in (("left", vl), ("right", vr)):
                            cam, T = pieces.camera_table_pose(S.calib, name)
                            img[:] = sc.draw_axes(img, T, cam, length_mm=3 * S.square)
                    tb_left.set_source(to_url(vl))
                    tb_right.set_source(to_url(vr))
                    tb_found.text = found
            elif tab == t_3d.props["name"]:
                if det_sw.value and S.views is not None:
                    key = hash(left[::16, ::16].tobytes() + right[::16, ::16].tobytes()) + hash(mask_sel.value)
                    if key == det_cache.get("key"):
                        return  # same frames as last time (e.g. image files): nothing to redo
                    result = await run.io_bound(detect_both, left, right, mask_sel.value or None)
                    if result is None:
                        return
                    _, _, vl, vr, rows = result
                    det_cache["key"] = key
                    det_left.set_source(to_url(vl))
                    det_right.set_source(to_url(vr))
                    table_rows = []
                    for p, st in rows:
                        x, y, _ = p.table_mm
                        diff = "" if st is None else f"{np.linalg.norm(np.array([x, y, p.table_mm[2]]) - st):.1f}"
                        stereo = "left camera only (ignored)" if st is None else f"{st[0]:.1f}, {st[1]:.1f}, {st[2]:.1f}"
                        table_rows.append({"color": p.color, "plane": f"{x:.1f}, {y:.1f}", "stereo": stereo, "diff": diff})
                    det_table.rows = table_rows
                    det_table.update()
                    sig = [(p.color, round(p.table_mm[0]), round(p.table_mm[1])) for p, _ in rows]
                    if sig != det_cache.get("sig"):
                        det_cache["sig"] = sig
                        draw_pieces(rows)
                else:
                    det_cache.clear()
                    det_left.set_source(to_url(left))
                    det_right.set_source(to_url(right))
        finally:
            busy["v"] = False

    async def check_webgl():
        ok = await ui.run_javascript(
            "(() => { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); })()")
        no_webgl.visible = not ok
        scene.visible = bool(ok)

    build_tuning()
    ui.timer(0.25, tick)
    ui.timer(1.0, check_webgl, once=True)
    refresh_pairs()
    refresh_status()
    draw_scene()
    if S.calib is not None:
        refresh_result()


if __name__ in {"__main__", "__mp_main__"}:
    if "--one-camera" in sys.argv:
        S.settings["one_camera"] = True
    S.load_data()
    try:
        S.connect()  # reconnect the sources from the last session
    except Exception as e:
        print(f"sources not connected: {e}")
    browser = "--browser" in sys.argv  # only serve on http://localhost:8080, don't open a window
    if not browser:
        app.on_startup(open_app_window)
    ui.run(root, title="Stereo calibration", port=PORT, reload=False, dark=True, show=False)
