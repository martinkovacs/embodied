"""Calibration and detection app, version 2 (NiceGUI). Replaces calib_gui.py; same data files, so either one works.

Tabs: Setup (cameras, board) -> Calibration (1: lens views, the board held up at different angles, 6+ per camera,
which fit each lens with its distortion and, where both cameras see the board, the stereo geometry; 2: the table
capture, the board flat on the table, which sets the table frame) -> Detection (live view, colour mask, threshold
sliders, click a piece to tune its colour) -> 3D (cameras, board and pieces in the table frame).

How it is built, so the UI stays simple:
  * Engine owns the cameras. One thread grabs frames and finds the board in them, another runs the piece detection
    on the newest frames. The UI only shows their latest results and calls Engine methods.
  * Pictures are served at full resolution from /v2img/<name>.jpg, so a click on a picture gives camera (or top-view)
    pixels directly.
  * The maths lives in stereo_calib.py and pieces.py; the files in calib_data/ are the ones runner.py reads:
    calib.json (calibration), detection.json (thresholds), settings.json (cameras, board), pairs/ (captures).

Run:  ./lab.sh python calib_app.py                 (then open http://localhost:8080)
      ./lab.sh python calib_app.py --snapshot calib_data/debug/20261123_171500.zip
                                           (a debug snapshot, from the Detection tab's bug button: its pictures,
                                           calibration and settings, as they were)
      ./lab.sh python calib_app.py --files 'synthetic/table_left.png,synthetic/scene_left.png' \\
                                           'synthetic/table_right.png,synthetic/scene_right.png'
                                           (image files instead of cameras, for development; < > step through them)
"""
import argparse
import copy
import glob
import json
import shutil
import subprocess
import threading
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.responses import Response
from nicegui import app, run, ui

import pieces
import stereo_calib as sc
from cameras import DATA, DETECTION_PATH, ROOT, SETTINGS_PATH, TEST_DATA, Source, list_cameras, load_settings

PORT = 8080
FILES_DATA = TEST_DATA / "v2"     # --files: development data, never the lab's calib_data
HEX = {"red": "#e53935", "blue": "#1e40af", "green": "#2e9d4f"}
SIDES = ("left", "right")
LIVE_W = 640            # the live board search runs on a copy this wide


# ---------------------------------------------------------------- sources

class Files:
    """Image files standing in for a camera (--files): comma-separated globs, stepped through with step()."""

    def __init__(self, patterns):
        self.files = []
        for pat in patterns.split(","):
            pat = pat.strip()
            self.files += sorted(glob.glob(pat if Path(pat).is_absolute() else str(ROOT / pat)))
        if not self.files:
            raise RuntimeError(f"no images match {patterns}")
        self.idx, self.cache = 0, None

    def grab(self):
        pass

    def retrieve(self):
        if self.cache is None or self.cache[0] != self.idx:
            self.cache = self.idx, cv2.imread(self.files[self.idx])
        return self.cache[1]

    def release(self):
        pass

    def label(self):
        return f"{self.idx + 1}/{len(self.files)} {Path(self.files[self.idx]).name}"


# ---------------------------------------------------------------- engine

@dataclass
class Capture:
    """One capture: the board's corners in each camera (canonical order), None where it wasn't seen."""
    name: str
    left: np.ndarray | None
    right: np.ndarray | None
    size: tuple[int, int]
    err_l: float | None = None
    err_r: float | None = None
    err_s: float | None = None      # stereo error, for a capture both cameras saw

    @property
    def both(self):
        return self.left is not None and self.right is not None


def live_corners(img, pattern):
    """Quick board search on a small copy, for the live view; corners in full-resolution pixels."""
    scale = min(1.0, LIVE_W / img.shape[1])
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, c = cv2.findChessboardCorners(gray, pattern, flags=flags)
    return c.reshape(-1, 2) / scale if ok else None


def full_corners(img, pattern):
    """The board's corners at full resolution in canonical order (stereo_calib.canonical_corners), or None. The
    accurate detector can miss a board the quick one finds (soft images): then the quick corners, refined."""
    c = sc.find_corners(img, pattern)
    if c is None:
        q = live_corners(img, pattern)
        if q is None:
            return None
        win = int(np.clip(np.linalg.norm(q[1] - q[0]) / 4, 3, 15))
        c = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), q.reshape(-1, 1, 2).astype(np.float32),
                             (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        c = c.reshape(-1, 2)
    return sc.canonical_corners(img, c, pattern)


class Engine:
    """Cameras, captures, calibration and detection. Thread-safe: the UI calls in from its event loop (heavy calls
    through run.io_bound), the grab and detect threads run on their own."""

    def __init__(self, files=None, data=None):
        self.settings = load_settings()
        self.files = files                  # (left patterns, right patterns) instead of cameras
        self.data = Path(data) if data else FILES_DATA if files else DATA
        if data and (self.data / "settings.json").exists():     # a snapshot: its board and cameras, not this PC's
            self.settings |= json.loads((self.data / "settings.json").read_text())
        self.pairs = self.data / "pairs"
        self.table_file, self.calib_file = self.data / "table_pair.txt", self.data / "calib.json"
        self.params_file = self.data / "detection.json" if files else DETECTION_PATH
        self.lock = threading.RLock()
        self.cams = None
        self.cam_error = None
        self.frames, self.frame_id = (None, None), 0
        self.corners = (None, None)         # live board corners per camera
        self.captures: list[Capture] = []
        self.loading = False
        self.table_name = self.table_file.read_text().strip() if self.table_file.exists() else None
        self.calib = sc.StereoCalib.load(self.calib_file) if self.calib_file.exists() else None
        self.calib_id = 0                   # bumps whenever the calibration changes
        self.views = None
        self.result = None                  # the last calibration's numbers, see calibrate()
        self.params = pieces.load_params(self.params_file)
        self.params_id = 0                  # bumps whenever the thresholds change
        self.want_detect = False            # set by the UI: the detection or 3D tab is open and live detection on
        self.want_corners = True            # set by the UI: the calibration tab is open (the live board search costs
                                            # a lot of CPU on a slow PC, and only that tab shows it)
        self.det = None                     # the latest detection, see detect()
        self.det_error = None
        self._build_views()
        threading.Thread(target=self._grab_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    # -- settings
    @property
    def pattern(self):
        return int(self.settings["cols"]), int(self.settings["rows"])

    @property
    def square(self):
        return float(self.settings["square_mm"])

    def save_settings(self):
        if self.files:          # development: the lab's settings stay as they are
            return
        # this app is two cameras, always: the runner and lab_check read these too
        self.settings.update(test_mode=False, one_camera=False)
        DATA.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(self.settings, indent=2))

    # -- cameras
    def connect(self):
        """(Re)open both cameras with the current settings. Raises with what's wrong."""
        with self.lock:
            self.disconnect()
            if self.files:
                cams = tuple(Files(p) for p in self.files)
            else:
                w, h = int(self.settings["width"]), int(self.settings["height"])
                if str(self.settings["left"]) == str(self.settings["right"]):
                    raise RuntimeError("left and right are the same camera")
                left = Source(str(self.settings["left"]), w, h)
                try:
                    right = Source(str(self.settings["right"]), w, h)
                except Exception:
                    left.release()
                    raise
                cams = (left, right)
            self.cams, self.cam_error = cams, None
            self.frames, self.corners, self.det = (None, None), (None, None), None
        self.save_settings()

    def disconnect(self):
        with self.lock:
            if self.cams:
                for c in self.cams:
                    c.release()
            self.cams = None

    def step(self, d):
        """--files: the next / previous image pair."""
        with self.lock:
            if self.cams and self.files:
                for c in self.cams:
                    c.idx = (c.idx + d) % len(c.files)

    def source_label(self):
        if not self.cams:
            return "not connected"
        if self.files:
            return f"files: {self.cams[0].label()} | {self.cams[1].label()}"
        return f"left {self.settings['left']} · right {self.settings['right']}"

    def _grab_loop(self):
        while True:
            with self.lock:
                cams = self.cams
            if not cams:
                time.sleep(0.2)
                continue
            try:
                for c in cams:
                    c.grab()
                fl, fr = (c.retrieve() for c in cams)
            except Exception as e:
                fl = fr = None
                self.cam_error = f"camera error: {e}"
            if fl is None or fr is None:
                self.cam_error = self.cam_error or "a camera gives no image"
                time.sleep(0.5)
                continue
            pattern = self.pattern
            cl, cr = (live_corners(fl, pattern), live_corners(fr, pattern)) if self.want_corners else (None, None)
            with self.lock:
                if self.cams is not cams:     # reconnected meanwhile
                    continue
                same = self.files and self.frames[0] is fl and self.frames[1] is fr
                if not same:
                    self.frames, self.frame_id, self.cam_error = (fl, fr), self.frame_id + 1, None
                self.corners = (cl, cr)
            time.sleep(0.2 if self.files else 0.03)

    # -- captures and calibration
    def load_captures(self):
        """The captures saved in calib_data/pairs (from earlier sessions too), corners found again (slow)."""
        self.loading = True
        try:
            caps = []
            for f in sorted(self.pairs.glob("*_left.png")):
                name = f.name.removesuffix("_left.png")
                left, right = cv2.imread(str(f)), cv2.imread(str(self.pairs / f"{name}_right.png"))
                if left is None or right is None:
                    continue
                cl, cr = full_corners(left, self.pattern), full_corners(right, self.pattern)
                if cl is not None or cr is not None:
                    caps.append(Capture(name, cl, cr, (left.shape[1], left.shape[0])))
            with self.lock:
                self.captures = caps
            self.calibrate()
        finally:
            self.loading = False

    def capture(self, table=False):
        """Capture the current frames: a lens view (the board held up at some angle; one camera seeing it is
        enough), or with table=True the table capture (the board flat on the table, both cameras must see it; it
        gets the star). Returns the Capture, or None if the board isn't seen (by both cameras, for the table)."""
        with self.lock:
            fl, fr = self.frames
        if fl is None:
            raise RuntimeError("no frames: connect the cameras first")
        cl, cr = full_corners(fl, self.pattern), full_corners(fr, self.pattern)
        if cl is None and cr is None or table and (cl is None or cr is None):
            return None
        name = f"{int(time.time() * 1000) % 10**9:09d}"
        self.pairs.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.pairs / f"{name}_left.png"), fl)
        cv2.imwrite(str(self.pairs / f"{name}_right.png"), fr)
        cap = Capture(name, cl, cr, (fl.shape[1], fl.shape[0]))
        with self.lock:
            self.captures.append(cap)
        if table:
            self.set_table(name, recalibrate=False)
        self.calibrate()
        return cap

    def delete(self, name):
        with self.lock:
            self.captures = [c for c in self.captures if c.name != name]
            if name == self.table_name:
                self.table_name = None
        for side in SIDES:
            (self.pairs / f"{name}_{side}.png").unlink(missing_ok=True)
        self.calibrate()

    def delete_all(self):
        with self.lock:
            names = [c.name for c in self.captures]
            self.captures = []
            self.table_name = None
        for name in names:
            for side in SIDES:
                (self.pairs / f"{name}_{side}.png").unlink(missing_ok=True)
        self.calibrate()

    def set_table(self, name, recalibrate=True):
        self.table_name = name
        self.data.mkdir(parents=True, exist_ok=True)
        self.table_file.write_text(name)
        if recalibrate:
            self.calibrate()

    def table_capture(self):
        """The capture that sets the table pose: the one with the star (Capture table sets it)."""
        with self.lock:
            return next((c for c in self.captures if c.both and c.name == self.table_name), None)

    def calibrate(self):
        """Each lens from every capture that camera saw the board in (stereo_calib.calibrate_lens: the full model,
        distortion too, from FULL_MODEL_VIEWS views on), then, with a table capture, the table pose and the stereo
        geometry (stereo_calib.stereo_from_table: from every capture both cameras saw). Saves calib.json only with a
        table capture. E.result = {"lens": side -> (model, views, rms), "check": stereo_from_table's or None}."""
        tc = self.table_capture()
        with self.lock:
            if not self.captures:
                self.result = None
                return
            size = tc.size if tc else self.captures[-1].size
            caps = [c for c in self.captures if c.size == size]
        cams, models = {}, {}
        for side in SIDES:
            views = [c for c in caps if getattr(c, side) is not None]
            if not views:
                continue
            cam, errs, model = sc.calibrate_lens([getattr(c, side) for c in views], size, self.pattern, self.square)
            for c, e in zip(views, errs):
                setattr(c, "err_l" if side == "left" else "err_r", e)
            cams[side] = cam
            models[side] = (model, len(views), cam.rms)
        for c in caps:
            c.err_s = None
        if tc is None:
            with self.lock:
                self.result = {"lens": models, "check": None}
            return
        others = [c for c in caps if c.both and c is not tc]
        calib, check = sc.stereo_from_table(tc.left, tc.right, cams["left"], cams["right"], self.pattern, self.square,
                                            pairs=[(c.left, c.right) for c in others])
        if check["pair_px"]:
            for c, e in zip([tc] + others, check["pair_px"]):
                c.err_s = e
        calib.cameras = None if self.files else {"left": self.settings["left"], "right": self.settings["right"]}
        self.data.mkdir(parents=True, exist_ok=True)
        calib.save(self.calib_file)
        with self.lock:
            self.calib, self.result = calib, {"lens": models, "check": check}
            self.calib_id += 1
            self.det = None
        self._build_views()

    def calib_status(self):
        """(chip text, colour) for the header: calibrated only with both lenses fully fitted and the table captured."""
        if self.loading:
            return "loading captures…", "grey"
        if self.calib is None or self.calib.T_left_table is None:
            return "not calibrated", "grey"
        r = self.result
        if r is not None and r["check"] is not None:
            full = all(r["lens"][s][0] == "full" for s in SIDES)
        else:   # the saved calib.json, from earlier: a lens without distortion was fitted from too few views
            full = all(np.any(np.asarray(getattr(self.calib, s).dist) != 0) for s in SIDES)
        return ("calibrated", "positive") if full else (f"rough: under {sc.FULL_MODEL_VIEWS} lens views", "warning")

    def _build_views(self):
        views = None
        if self.calib is not None and self.calib.T_left_table is not None:
            b = float(self.settings["board_mm"])
            views = tuple(pieces.view_from_calib(self.calib, side, b) for side in SIDES)
        with self.lock:
            self.views = views

    def board_outline(self):
        """The board's four outer inner corners in the table frame, mm (4, 2): where the left camera saw them in the
        table capture, on the table plane (the table frame's y may run either way along the board)."""
        tc = self.table_capture()
        if tc is None or self.calib is None:
            return None
        cols, rows = self.pattern
        n = cols * rows
        cam, T = pieces.camera_table_pose(self.calib, "left")
        px = tc.left[[0, cols - 1, n - 1, n - cols]]
        rays = cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), np.asarray(cam.K),
                                   np.asarray(cam.dist)).reshape(-1, 2)
        R, t = T[:3, :3], T[:3, 3]
        c = -R.T @ t
        d = (R.T @ np.c_[rays, np.ones(4)].T).T
        return (c + (-c[2] / d[:, 2])[:, None] * d)[:, :2]

    # -- debug snapshot
    def snapshot(self, note=""):
        """Everything about the detection as it is now, to look at away from the lab: both camera pictures, the
        calibration, the settings (saved or not), the detection drawn on the pictures, the colour masks, and a report
        of every piece found, unpaired, outside the other camera's view, or nearly found. A folder in <data>/debug/
        and a zip of it. Returns (zip path, one-line summary). Reopen it with --snapshot."""
        with self.lock:
            frames, calib, views = self.frames, self.calib, self.views
            params = copy.deepcopy(self.params)
        if frames[0] is None or frames[1] is None:
            raise RuntimeError("no pictures: connect the cameras first")
        if views is None or calib is None:
            raise RuntimeError("calibrate first")
        out = self.data / "debug" / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        for side, img in zip(SIDES, frames):
            cv2.imwrite(str(out / f"{side}.png"), img)      # lossless: the detection runs on exactly these again
        t0 = time.monotonic()
        det = detect(frames, views, calib, params)          # on the saved pictures, not the live result
        seconds = time.monotonic() - t0
        calib.save(out / "calib.json")
        pieces.save_params(out / "detection.json", params)
        (out / "settings.json").write_text(json.dumps(self.settings, indent=2))

        def info(p, why=None):
            d = {"color": p.color, "table_mm": [round(float(v), 1) for v in p.table_mm],
                 "image_px": [round(float(v), 1) for v in p.image_uv],
                 "top_view_px": [round(float(v), 1) for v in p.view_uv],
                 "rim": round(float(p.rim), 3), "fill": round(float(p.fill), 3), "around": round(float(p.around), 3)}
            return d | ({"why": why} if why else {})
        jpg = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cams = {}
        for i, side in enumerate(SIDES):
            img, view = frames[i], views[i]
            found, alone, misses = det["found"][i], det["lone"][i], det["misses"][i]
            outside = [p for p in det["raw"][i] if not any(p is q for q in found)]
            notes = det["notes"] | {id(p): "outside the other camera's view" for p in outside}
            everything = found + outside
            vis = pieces.draw_misses(pieces.draw(img, everything, view, lone=alone + outside, notes=notes), misses, view)
            cv2.imwrite(str(out / f"{side}_detection.jpg"), vis, jpg)
            cv2.imwrite(str(out / f"{side}_top.jpg"),
                        pieces.draw_top(det["tops"][i], everything, misses, view.ppm, alone + outside), jpg)
            cv2.imwrite(str(out / f"{side}_edges.jpg"),
                        pieces.draw_edges(det["tops"][i], view, params, everything, misses, alone + outside), jpg)
            for color in params["colors"]:
                cv2.imwrite(str(out / f"{side}_mask_{color}.jpg"), pieces.mask_view(img, params, color), jpg)
            cams[side] = {"found_paired": [info(p) for p in found if not any(p is q for q in alone)],
                          "unpaired": [info(p, det["notes"].get(id(p))) for p in alone],
                          "outside_other_view": [info(p) for p in outside],
                          "near_misses": [info(p) for p in misses]}
        both = [{"color": p.color, "table_mm": [round(float(v), 1) for v in st]} for p, st in det["matched"] if st is not None]
        summary = (f"{len(both)} piece(s) both cameras agree on; " + "; ".join(
            f"{side}: {len(c['found_paired']) + len(c['unpaired'])} found, {len(c['unpaired'])} unpaired, "
            f"{len(c['outside_other_view'])} outside the other view, {len(c['near_misses'])} near misses"
            for side, c in cams.items()))
        report = {"time": datetime.now().isoformat(timespec="seconds"), "note": note, "code": code_version(),
                  "detection_seconds": round(seconds, 2), "summary": summary,
                  "image_size": [int(frames[0].shape[1]), int(frames[0].shape[0])],
                  "top_view": {"ppm": views[0].ppm, "plane_z_mm": views[0].z},
                  "settings": {k: self.settings.get(k) for k in ("left", "right", "width", "height", "cols", "rows",
                                                                "square_mm", "board_mm")},
                  "params": params, "pieces_both_cameras": both, "cameras": cams}
        (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
        (out / "README.txt").write_text(
            f"Debug snapshot {out.name}\n{note}\n\n{summary}\n\n"
            "left.png / right.png: the camera pictures. *_detection.jpg: coloured = both cameras agree, grey ? = one "
            "camera only (why), thin grey with scores = near misses. *_top.jpg: the top view the detector works on. "
            "*_edges.jpg: the edges the detector sees (white; yellow = from the colour mask only) and each piece's "
            "rim test (green dot = the ray met an edge, red = not). *_mask_<colour>.jpg: what passes each colour's "
            "thresholds. report.json: all of it in numbers.\n"
            f"Reopen: ./lab.sh python calib_app.py --snapshot {out}\n")
        zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
        return Path(zip_path), summary

    # -- detection
    def _detect_loop(self):
        done = None
        while True:
            with self.lock:
                frames, fid, views, calib = self.frames, self.frame_id, self.views, self.calib
                key = (fid, self.params_id, self.calib_id)
                params = copy.deepcopy(self.params)
                want = self.want_detect
            if not want or views is None or frames[0] is None or key == done:
                time.sleep(0.1)
                continue
            try:
                t0 = time.monotonic()
                det = detect(frames, views, calib, params)
                det.update(frames=frames, key=key, seconds=time.monotonic() - t0)
                with self.lock:
                    if self.calib_id == key[2]:
                        self.det, self.det_error = det, None
            except Exception as e:
                self.det_error = f"detection failed: {e}"
            done = key

    def set_param(self, fn):
        """Change the thresholds: fn(params) edits them in place."""
        with self.lock:
            fn(self.params)
            self.params_id += 1

    def reset_params(self):
        with self.lock:
            self.params = pieces.load_params()
            self.params_id += 1

    def save_params(self):
        with self.lock:
            pieces.save_params(self.params_file, self.params)

    def teach(self, side, kind, u, v, color=None):
        """A click on a piece's top in a picture (kind "cam": camera pixel, "top": top-view pixel): widen its
        colour's thresholds so it passes, lower the rim / fill minimum that keeps that piece out (not below 0.3), and
        say what still does (pieces.probe). The detection thread shows the result. Returns a message; raises
        ValueError if the spot isn't a coloured piece."""
        with self.lock:
            views, frames = self.views, self.frames
            params = copy.deepcopy(self.params)
        if views is None:
            raise ValueError("calibrate first")
        i = SIDES.index(side)
        img, view = frames[i], views[i]
        if img is None:
            raise ValueError("no picture yet")
        if kind == "top":
            r = pieces.PIECE_D_MM / 2 * view.ppm
            color, changed = pieces.teach(view.warp(img), u, v, r, params, color)
        else:
            x, y = view.view_to_table(*view.image_to_view(u, v))
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = view.project(np.c_[x + pieces.PIECE_D_MM / 2 * np.cos(ang), y + pieces.PIECE_D_MM / 2 * np.sin(ang),
                                      np.full(16, view.z)])
            r = float(np.clip(np.hypot(ring[:, 0] - u, ring[:, 1] - v).min(), 4, 150)) if np.isfinite(ring).all() else 15.0
            color, changed = pieces.teach(img, u, v, r, params, color)
        # this piece's own scores with the widened colours: lower the rim / fill minimum it misses (not below 0.3),
        # and say what still keeps it out
        top = view.warp(img)
        tu, tv = (u, v) if kind == "top" else view.image_to_view(u, v)
        pr = pieces.probe(top, view, tu, tv, params)
        if pr is not None and pr["fill"] >= 0.4:
            for key, score in (("min_rim", pr["rim"]), ("min_fill", pr["fill"])):
                new = max(0.3, round(score - 0.05, 2))
                if new < params[key]:
                    changed.append(f"{key} {params[key]:.2f} → {new:.2f}")
                    params[key] = new
            pr = pieces.probe(top, view, tu, tv, params)
        with self.lock:
            self.params = params
            self.params_id += 1
        msg = f"{color}: {', '.join(changed) or 'no change needed'}"
        if pr is None:
            return msg + ". No piece colour found around the click"
        msg += (f". This piece in the {side} camera: fill {pr['fill']:.2f}, rim {pr['rim']:.2f}, around "
                f"{pr['around']:.2f}: " + ("passes" if not pr["fails"] else "still out, " + ", ".join(pr["fails"])))
        if any(f.startswith("rim") for f in pr["fails"]):
            msg += (". Its outline is too weak in the picture: lower 'edges ≥' (now the strongest "
                    f"{100 - params['edge_pct']:.0f} % of edges count), or switch on 'Outline from the mask too'")
        if any(f.startswith("around") for f in pr["fails"]):
            msg += ". Too much of its colour around it: raise 'around ≤', or narrow the colour (S, V, hue)"
        return msg


def seen_by(view, xyz):
    """Is the table point (mm) inside this camera's top view, i.e. in its picture?"""
    u, v = (xyz[0] - view.x0) * view.ppm, (view.y1 - xyz[1]) * view.ppm
    h, w = view.maps.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return False
    mx, my = view.maps[int(v), int(u)]     # where the top-view pixel comes from in the camera image
    iw, ih = view.image_size
    return bool(0 <= mx < iw and 0 <= my < ih)


def detect(frames, views, calib, params):
    """Both cameras' pieces, paired: {"matched": [(left Piece, table xyz or None)], and per camera (left, right):
    "found", "misses" (near misses), "tops" (top views), "lone" (pieces only that camera sees), "raw" (found, before
    dropping the ones outside the other camera's view)}."""
    def one(i):
        top, misses = views[i].warp(frames[i]), []
        return pieces.detect_pieces(frames[i], views[i], params=params, rejected=misses, top=top), misses, top
    with ThreadPoolExecutor(2) as pool:
        (pl, ml, tl), (pr, mr, tr) = pool.map(one, (0, 1))
    raw = (pl, pr)
    # only where both cameras look: a detection the other camera can't see can never be confirmed (that's where
    # the background, walls and cloth come in)
    pl = [p for p in pl if seen_by(views[1], p.table_mm)]
    pr = [q for q in pr if seen_by(views[0], q.table_mm)]
    matched = pieces.matched_pieces(calib, pl, pr, max_mm=params["match_mm"], max_dz=params["max_dz_mm"])
    paired = set()
    for p, st in matched:
        if st is not None:      # its right partner: same colour, nearest on the table
            q = min((q for q in pr if q.color == p.color),
                    key=lambda q: np.hypot(q.table_mm[0] - p.table_mm[0], q.table_mm[1] - p.table_mm[1]))
            paired.add(id(q))
    lone = ([p for p, st in matched if st is None], [q for q in pr if id(q) not in paired])
    # why a piece wasn't paired: how far the other camera's nearest piece of that colour is, on the table
    notes = {}
    for mine, other, name in ((lone[0], pr, "R"), (lone[1], pl, "L")):
        for p in mine:
            d = [np.hypot(q.table_mm[0] - p.table_mm[0], q.table_mm[1] - p.table_mm[1]) for q in other if q.color == p.color]
            notes[id(p)] = f"{name} {min(d):.0f} mm away" if d else f"{name} sees none"
    return {"matched": matched, "found": (pl, pr), "misses": (ml, mr), "tops": (tl, tr), "lone": lone, "notes": notes,
            "raw": raw}


def code_version():
    """The git commit this runs, + "with changes" if files differ from it: which code a snapshot came from."""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (rev or "unknown") + (" with changes" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- pictures

_edges_cache = {}      # side -> (key, picture): the edge view, made once per detection (it takes a moment)


def picture(E: Engine, name, mask=None, misses=False, lone=False):
    """One picture for the UI as BGR: calib_<side> (live, board corners, coverage, table axes), cam_<side>
    (detection on the camera image), top_<side> (detection on the top view), edges_<side> (the edges the detector
    sees on the top view, with each piece's rim test: pieces.draw_edges). None if there's nothing to show."""
    kind, side = name.split("_")
    i = SIDES.index(side)
    with E.lock:
        frames, corners, det, views, calib = E.frames, E.corners, E.det, E.views, E.calib
        caps = [getattr(c, side) for c in E.captures if getattr(c, side) is not None]
        params = copy.deepcopy(E.params)
    pattern = E.pattern
    if kind == "calib":
        img = frames[i]
        if img is None:
            return None
        vis = img.copy()
        if caps:
            layer = vis.copy()
            for c in caps:
                cv2.fillConvexPoly(layer, cv2.convexHull(c.astype(np.int32)), (0, 160, 0))
            vis = cv2.addWeighted(layer, 0.25, vis, 0.75, 0)
        if corners[i] is not None:
            cv2.drawChessboardCorners(vis, pattern, corners[i].reshape(-1, 1, 2).astype(np.float32), True)
        if calib is not None and calib.T_left_table is not None:
            cam, T = pieces.camera_table_pose(calib, side)
            vis = sc.draw_axes(vis, T, cam, length_mm=3 * E.square)
        return vis
    base = frames[i]       # always the live frame: the latest detection is drawn on it, a moment old at most
    if base is None:
        return None
    view = views[i] if views else None
    if kind == "edges":
        if view is None:
            return None
        # on the frame the detection ran on, so the rim dots match its pieces; without a detection, the live frame
        # every 2 s
        key = (det["key"], E.params_id) if det else ("live", E.params_id, int(time.monotonic() // 2))
        hit = _edges_cache.get(side)
        if hit and hit[0] == key:
            return hit[1]
        img = det["frames"][i] if det else base
        vis = pieces.draw_edges(view.warp(img), view, params, det["found"][i] if det else [],
                                det["misses"][i] if det else [], det["lone"][i] if det else [])
        _edges_cache[side] = (key, vis)
        return vis
    if kind == "top":
        if view is None:
            return None
        pic = view.warp(base)
    else:
        pic = base
    if mask:
        pic = pieces.mask_view(pic, params, mask)
    if det:
        alone = det["lone"][i]
        found = det["found"][i] if lone else [p for p in det["found"][i] if not any(p is q for q in alone)]
        near = det["misses"][i] if misses else []
        if kind == "top":
            pic = pieces.draw_top(pic, found, near, view.ppm, alone)
        else:
            pic = pieces.draw_misses(pieces.draw(pic, found, view, lone=alone, notes=det["notes"]), near, view)
    return pic


def placeholder(text):
    img = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(img, text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- UI

def build_ui(E: Engine):
    @app.get("/v2img/{name}.jpg")
    def image(name: str, mask: str = "", misses: int = 0, lone: int = 0):    # FastAPI runs a plain def in a thread
        try:
            img = picture(E, name, mask or None, bool(misses), bool(lone))
        except Exception as e:
            img = placeholder(f"error: {e}"[:60])
        if img is None:
            img = placeholder("no picture yet")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @ui.page("/")
    def page():
        ui.page_title("Calibration v2")
        ui.query("body").style("background: #121212")

        # ---- header
        with ui.header().classes("items-center bg-slate-800 gap-3"):
            ui.label("Calibration v2").classes("text-lg font-medium")
            if E.files:
                ui.button(icon="chevron_left", on_click=lambda: E.step(-1)).props("flat dense color=white")
                ui.button(icon="chevron_right", on_click=lambda: E.step(1)).props("flat dense color=white")
            src = ui.label().classes("text-sm opacity-80")
            ui.space()
            chip_cam = ui.chip(icon="videocam").props("dense")
            chip_cal = ui.chip(icon="straighten").props("dense")
            chip_det = ui.chip(icon="category").props("dense")

        with ui.tabs().classes("w-full") as tabs:
            t_setup = ui.tab("Setup", icon="settings")
            t_cal = ui.tab("Calibration", icon="grid_on")
            t_det = ui.tab("Detection", icon="center_focus_weak")
            t_3d = ui.tab("3D", icon="view_in_ar")

        with ui.tab_panels(tabs, value=t_setup if E.cams is None else t_cal).classes("w-full bg-transparent"):
            with ui.tab_panel(t_setup):
                setup_panel(E, tabs, t_cal)
            with ui.tab_panel(t_cal).classes("p-2"):
                cal = calibration_panel(E)
            with ui.tab_panel(t_det).classes("p-2"):
                det = detection_panel(E)
            with ui.tab_panel(t_3d).classes("p-2"):
                d3 = scene_panel(E)

        def tick():
            tab = tabs.value if isinstance(tabs.value, str) else tabs.value.props["name"]
            src.text = E.source_label()
            if E.cam_error:
                chip_cam.text, color = E.cam_error, "negative"
            elif E.cams:
                chip_cam.text, color = "cameras on", "positive"
            else:
                chip_cam.text, color = "no cameras", "grey"
            chip_cam.props(f"color={color} text-color=white")
            chip_cal.text, color = E.calib_status()
            chip_cal.props(f"color={color} text-color=white")
            d = E.det
            if E.det_error:
                chip_det.text, color = E.det_error[:50], "negative"
            elif d:
                n = sum(st is not None for _, st in d["matched"])
                chip_det.text, color = f"{n} piece{'s' if n != 1 else ''} · {d['seconds']:.1f} s", "positive"
            else:
                chip_det.text, color = "detection off", "grey"
            chip_det.props(f"color={color} text-color=white")
            E.want_detect = tab in (t_det.props["name"], t_3d.props["name"]) and det.live.value
            E.want_corners = tab == t_cal.props["name"]
            if tab == t_cal.props["name"]:
                cal.tick()
            elif tab == t_det.props["name"]:
                det.tick()
            elif tab == t_3d.props["name"]:
                d3.tick()
        ui.timer(0.2, tick)


def setup_panel(E, tabs, t_cal):
    st = E.settings
    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.card().classes("flex-1 min-w-[420px]"):
            ui.label("Cameras").classes("text-base font-medium")
            found = ui.label().classes("text-sm opacity-70")
            options = {}

            def refresh():
                cams = list_cameras()
                options.clear()
                options.update({spec: f"{spec} · {name}" for spec, name in cams})
                for side in SIDES:          # keep the saved ones selectable even if unplugged
                    options.setdefault(str(st[side]), f"{st[side]} (not found)")
                for s in (left, right):
                    s.set_options(dict(options))
                found.text = f"{len(cams)} camera(s) found" if cams else "No cameras found: plugged in? (LAB_GUIDE.md)"
            with ui.row().classes("w-full items-center no-wrap"):
                left = ui.select({}, label="Left camera").bind_value(st, "left").classes("flex-1")
                ui.button(icon="swap_horiz", on_click=lambda: swap()).props("flat round").tooltip("Swap left and right")
                right = ui.select({}, label="Right camera").bind_value(st, "right").classes("flex-1")
            with ui.row().classes("items-center gap-3"):
                ui.number("Width", format="%d").bind_value(st, "width").classes("w-28")
                ui.number("Height", format="%d").bind_value(st, "height").classes("w-28")
                ui.button("Find cameras", icon="refresh", on_click=refresh).props("flat")
            ui.label("Swapping the cameras makes the calibration invalid: capture the board on the table again.") \
                .classes("text-xs opacity-60")

            async def swap():
                st["left"], st["right"] = st["right"], st["left"]
                if E.cams:
                    await connect()

            async def connect():
                try:
                    await run.io_bound(E.connect)
                    ui.notify("Cameras connected", type="positive")
                except Exception as e:
                    ui.notify(f"Can't connect: {e}", type="negative", multi_line=True)
            with ui.row().classes("gap-2 mt-2"):
                ui.button("Connect", icon="link", on_click=connect)
                ui.button("Disconnect", icon="link_off", on_click=lambda: E.disconnect()).props("flat")
            if E.files:
                ui.label("Running on image files (--files): the camera settings aren't used.").classes("text-amber-400 text-sm")
            refresh()
        with ui.card().classes("flex-1 min-w-[420px]"):
            ui.label("Checkerboard").classes("text-base font-medium")
            ui.label("Inner corners = squares minus one (the OpenCV board is 9 x 6). Measure the squares with a caliper: "
                     "printers rarely print at exactly 100%.").classes("text-sm opacity-70")
            with ui.grid(columns=2).classes("w-full gap-x-4"):
                ui.number("Inner corners (cols)", format="%d").bind_value(st, "cols")
                ui.number("Inner corners (rows)", format="%d").bind_value(st, "rows")
                ui.number("Square size (mm)", format="%.2f").bind_value(st, "square_mm")
                ui.number("Board thickness (mm)", format="%.1f").bind_value(st, "board_mm") \
                    .tooltip("The board lies on the table for the calibration: the pieces stand this much lower")

            async def apply_board():
                E.save_settings()
                n = ui.notification("Finding the board in the captures again…", spinner=True, timeout=None)
                await run.io_bound(E.load_captures)
                n.dismiss()
                ui.notify("Board settings applied", type="positive")
            ui.button("Apply board settings", icon="check", on_click=apply_board).props("flat")
    with ui.row().classes("w-full justify-end mt-4"):
        ui.button("Next: Calibration", icon="arrow_forward", on_click=lambda: tabs.set_value(t_cal))


def calibration_panel(E):
    with ui.row().classes("w-full no-wrap gap-3 items-start"):
        with ui.column().classes("flex-1 min-w-0 gap-2"):
            with ui.row().classes("w-full no-wrap gap-2"):
                imgs = [ui.interactive_image().classes("w-1/2") for _ in SIDES]
            seen = ui.label().classes("text-sm")
            with ui.row().classes("w-full no-wrap gap-3 items-stretch"):
                with ui.card().classes("flex-1 gap-1"):
                    ui.label("1 · Lens views").classes("font-medium")
                    ui.label(f"Hold the board up in front of the cameras and press Capture view, {sc.FULL_MODEL_VIEWS} "
                             "times or more per camera: tilted 20 to 45° different ways, near and far, the middle and "
                             "the corners of the picture (the green area shows where it has been). Hold it still. "
                             "Views both cameras see at once also fit where one camera is from the other.") \
                        .classes("text-sm opacity-70")
                    with ui.row().classes("items-center gap-3 mt-auto"):
                        btn_view = ui.button("Capture view", icon="photo_camera")
                        counts = ui.label().classes("text-sm")
                with ui.card().classes("flex-1 gap-1"):
                    ui.label("2 · The table").classes("font-medium")
                    ui.label("Last, lay the board flat on the table in front of the arm, seen by both cameras, its "
                             "short side with the two dark corner squares against the arm's base, and press Capture "
                             "table. The axes then start at (0, 0), next to the arm: x red, y green, z blue. Leave the "
                             "board there for robot_frame.py, and don't move the cameras afterwards.") \
                        .classes("text-sm opacity-70")
                    with ui.row().classes("items-center gap-3 mt-auto"):
                        btn_table = ui.button("Capture table", icon="table_restaurant")
        with ui.card().classes("w-[380px] shrink-0 gap-1"):
            title = ui.label().classes("font-medium")
            ui.label("★ = the table capture · L / R = lens error per camera · S = stereo error (both cameras) · "
                     "red = over 1 px: delete it (✕)").classes("text-xs opacity-60")
            listing = ui.column().classes("w-full gap-0 overflow-auto").style("max-height: 40vh")
            with ui.row().classes("w-full"):
                ui.space()
                clear = ui.button("Delete all", icon="delete_sweep").props("flat dense color=negative")
            result = ui.label().classes("text-sm whitespace-pre-line mt-2")

    shown = {"list": None, "id": None}

    def refresh_list():
        tc = E.table_capture()
        with E.lock:
            caps = list(E.captures)
        title.text = f"Captures: {len(caps)}"
        listing.clear()
        with listing:
            for i, c in enumerate(caps):
                with ui.row().classes("items-center w-full no-wrap text-sm min-h-[30px]"):
                    star = ui.button(icon="star" if c is tc else "star_border",
                                     on_click=lambda c=c: act(E.set_table, c.name)).props("flat dense size=sm")
                    star.set_enabled(c.both)
                    ui.label(f"#{i + 1}").classes("w-8")
                    f = lambda corners, e: "–" if corners is None else "✓" if e is None else f"{e:.2f}"
                    bad = max(c.err_l or 0, c.err_r or 0, c.err_s or 0) > 1.0
                    stereo = f" · S {c.err_s:.2f}" if c.err_s is not None else ""
                    ui.label(f"L {f(c.left, c.err_l)} · R {f(c.right, c.err_r)}{stereo} px") \
                        .classes("flex-1" + (" text-red-400" if bad else ""))
                    ui.button(icon="close", on_click=lambda c=c: act(E.delete, c.name)).props("flat dense size=sm")
        result.text = result_text(E)

    async def act(fn, *args):
        try:
            await run.io_bound(fn, *args)
        except Exception as e:
            ui.notify(f"Calibration failed: {e}", type="negative", multi_line=True)

    async def capture(table):
        btn = btn_table if table else btn_view
        btn.props("loading")
        try:
            cap = await run.io_bound(E.capture, table)
        except Exception as e:
            ui.notify(f"Calibration failed: {e}", type="negative", multi_line=True)
            return
        finally:
            btn.props(remove="loading")
        if cap is None:
            ui.notify(("Both cameras must see the whole board for the table capture" if table else
                       "The board isn't in either picture") + " (the corner count in Setup must match the board)",
                      type="warning")
            return
        short = [s for s, n in lens_views().items() if n < sc.FULL_MODEL_VIEWS]
        if table and short:
            ui.notify(f"Table captured, but the {' and '.join(short)} lens has fewer than {sc.FULL_MODEL_VIEWS} views: "
                      "its distortion isn't fitted, so positions can be centimetres off. Capture more lens views.",
                      type="warning", multi_line=True)
        else:
            ui.notify("Captured" + ("" if table or cap.both else f" ({'left' if cap.left is not None else 'right'} "
                                                                  "camera only)"), type="positive")
    btn_view.on_click(lambda: capture(False))
    btn_table.on_click(lambda: capture(True))

    def lens_views():
        with E.lock:
            return {s: sum(getattr(c, s) is not None for c in E.captures) for s in SIDES}
    clear.on_click(lambda: act(E.delete_all))

    class Panel:
        def tick(self):
            key = (E.frame_id, len(E.captures), E.calib_id)
            if key != shown["id"]:
                shown["id"] = key
                for img, side in zip(imgs, SIDES):
                    img.set_source(f"/v2img/calib_{side}.jpg?k={E.frame_id}-{E.calib_id}-{len(E.captures)}")
                cl, cr = E.corners
                seen.text = f"board: left {'✓' if cl is not None else '✗'} · right {'✓' if cr is not None else '✗'}"
                counts.text = " · ".join(f"{s} {n}/{sc.FULL_MODEL_VIEWS}" + (" ✓" if n >= sc.FULL_MODEL_VIEWS else "")
                                         for s, n in lens_views().items())
            sig = (tuple((c.name, c.err_l, c.err_r) for c in E.captures), E.table_name, E.calib_id)
            if sig != shown["list"]:
                shown["list"] = sig
                refresh_list()
    return Panel()


def result_text(E):
    if E.result is None:
        return (f"Not calibrated: capture {sc.FULL_MODEL_VIEWS} or more lens views per camera, then the table "
                "(the board flat on the table, seen by both cameras).")
    lines = []
    models, check = E.result["lens"], E.result["check"]
    for side in SIDES:
        if side not in models:
            lines.append(f"{side} lens: no view yet")
            continue
        model, n, rms = models[side]
        note = (f"  ⚠ distortion not fitted: {sc.FULL_MODEL_VIEWS - n} more view{'s' if sc.FULL_MODEL_VIEWS - n != 1 else ''}"
                if model != "full" else "  ⚠ over 1 px: delete the red captures" if rms > 1.0 else "")
        lines.append(f"{side} lens: {model}, {n} view{'s' if n != 1 else ''}, RMS {rms:.2f} px{note}")
    if check is None:
        lines.append("Table: not captured yet (step 2)")
        return "\n".join(lines)
    lines.append(f"Board on the table: reprojection left {check['left_px']:.2f} px, right {check['right_px']:.2f} px")
    lines.append(f"Stereo: from {check['stereo_views']} view{'s' if check['stereo_views'] != 1 else ''} both cameras saw"
                 + ("" if check["stereo_views"] > 1 else " (the table only: hold the board where both see it too)")
                 + f" · baseline {E.calib.baseline_mm:.0f} mm")
    lines.append(f"Stereo check: corners {check['mean_mm']:.2f} mm mean, {check['max_mm']:.2f} mm max")
    for side in SIDES:
        _, T = pieces.camera_table_pose(E.calib, side)
        Tinv = np.linalg.inv(T)
        tilt = np.degrees(np.arccos(np.clip(-Tinv[2, 2], -1, 1)))
        lines.append(f"{side} camera: {Tinv[2, 3]:.0f} mm above the table, tilted {tilt:.0f}°")
    return "\n".join(lines)


def detection_panel(E):
    with ui.row().classes("w-full no-wrap gap-3 items-start"):
        with ui.column().classes("flex-1 min-w-0 gap-2"):
            with ui.row().classes("w-full items-center gap-2"):
                kind = ui.toggle({"cam": "Camera", "top": "Top view", "edges": "Edges"}, value="cam").props("dense") \
                    .tooltip("Edges: the top view as the detector sees it. White = an edge in the picture, yellow = "
                             "an edge only the colour mask has. Around each piece and near miss, a dot per rim ray: "
                             "green = it met an edge, red = not (rim = share of green). The line points to where "
                             "its side shows, towards the camera: of a piece found twice, the circle on that line "
                             "is on its side")
                side = ui.toggle({"left": "Left", "right": "Right", "both": "Both"}, value="left").props("dense")
                mask = ui.select({"": "no mask", **{c: f"{c} mask" for c in pieces.COLORS}}, value="") \
                    .props("dense outlined").classes("w-36").tooltip("Only the pixels that pass a colour's thresholds")
                ui.space()
                show_lone = ui.switch("One-camera", value=True).tooltip(
                    "Also show what only one camera sees (grey, with ?). The runner ignores those: only pieces both "
                    "cameras agree on count")
                show_misses = ui.switch("Near misses").tooltip(
                    "Grey circles with scores where something piece-like was rejected: which minimum keeps it out")
                live = ui.switch("Live detection", value=True)
                ui.button(icon="fullscreen", on_click=lambda: open_full()).props("flat dense").tooltip("Full screen")
                ui.button(icon="bug_report", on_click=lambda: snap_dlg.open()).props("flat dense") \
                    .tooltip("Debug snapshot: pictures, settings and detection in one zip, to send")
            with ui.row().classes("w-full no-wrap gap-2"):
                imgs = {s: ui.interactive_image(events=["click"]).classes("w-full cursor-crosshair") for s in SIDES}
            ui.label("Coloured: a piece both cameras agree on (what the runner uses). One-camera (switch): grey with ?, "
                     "seen by one camera only, ignored. Near misses (switch): thin grey with scores, the one past its "
                     "limit keeps it out. Click the middle of a piece's top to tune its colour to it.") \
                .classes("text-xs opacity-60")
        with ui.card().classes("w-[400px] shrink-0 gap-2 overflow-auto").style("max-height: calc(100vh - 140px)"):
            ui.label("Thresholds").classes("font-medium")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                teach_on = ui.switch("Click to tune", value=True)
                teach_color = ui.select({"": "colour: auto", **{c: c for c in pieces.COLORS}}, value="") \
                    .props("dense outlined").classes("w-36")
            sliders = threshold_sliders(E)
            with ui.row().classes("gap-2"):
                def save():
                    E.save_params()
                    ui.notify("Saved: the runner uses these now", type="positive")

                def reset():
                    E.reset_params()
                    sliders.load()
                ui.button("Save", icon="save", on_click=save).props("dense")
                ui.button("Defaults", icon="restart_alt", on_click=reset).props("dense flat")
            table = ui.table(columns=[{"name": "c", "label": "colour", "field": "c"},
                                      {"name": "xyz", "label": "table x, y, z mm", "field": "xyz"}],
                             rows=[]).props("dense flat").classes("w-full")

    with ui.dialog() as snap_dlg, ui.card().classes("w-[560px] gap-2"):
        ui.label("Debug snapshot").classes("text-base font-medium")
        ui.label("Saves both camera pictures, the calibration, the current settings (saved or not), the detection drawn "
                 "on the pictures, the colour masks and a report of every piece, as one zip to send.") \
            .classes("text-sm opacity-70")
        snap_note = ui.textarea("What's on the table, and what goes wrong?",
                                placeholder="e.g. 12 pieces: 4 red, 4 blue, 4 green. Only 3 found; the blue ones never.") \
            .classes("w-full").props("autogrow outlined")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=snap_dlg.close).props("flat")
            snap_btn = ui.button("Save snapshot", icon="save")

    async def save_snapshot():
        snap_btn.props("loading")
        try:
            zip_path, summary = await run.io_bound(E.snapshot, snap_note.value or "")
        except Exception as e:
            ui.notify(f"No snapshot: {e}", type="negative", multi_line=True)
            return
        finally:
            snap_btn.props(remove="loading")
        snap_dlg.close()
        snap_note.value = ""
        ui.notify(f"Saved {zip_path}. {summary}", type="positive", multi_line=True, timeout=0, close_button=True)
        ui.download(zip_path)       # into the browser's downloads too, ready to attach
    snap_btn.on_click(save_snapshot)

    full = {"open": False, "pct": 100.0}
    with ui.dialog().props("maximized") as dlg, ui.card().classes("w-full h-full p-2 gap-2"):
        with ui.row().classes("w-full items-center gap-2"):
            full_title = ui.label().classes("font-medium")
            ui.space()
            ui.button(icon="zoom_out", on_click=lambda: zoom(full["pct"] / 1.25)).props("flat dense")
            pct = ui.label().classes("w-12 text-center text-sm")
            ui.button(icon="zoom_in", on_click=lambda: zoom(full["pct"] * 1.25)).props("flat dense")
            ui.button(icon="fit_screen", on_click=lambda: zoom(100)).props("flat dense")
            ui.button(icon="close", on_click=dlg.close).props("flat dense")
        with ui.element("div").classes("w-full grow overflow-auto") as box:
            full_img = ui.interactive_image(events=["click"]).classes("cursor-crosshair")
        ui.label("Ctrl + wheel zooms, the scroll bars pan. Click a piece's top to tune its colour to it.") \
            .classes("text-xs opacity-60")
    dlg.on("hide", lambda: full.update(open=False))
    box.on("wheel", lambda e: zoom(full["pct"] * (1.15 if e.args["deltaY"] < 0 else 1 / 1.15))
           if e.args.get("ctrlKey") else None, ["deltaY", "ctrlKey"], throttle=0.05)

    def zoom(p):
        full["pct"] = float(np.clip(p, 25, 800))
        full_img.style(f"width: {full['pct']:.0f}%; max-width: none")
        pct.text = f"{full['pct']:.0f}%"

    def open_full():
        full["open"] = True
        zoom(100)
        shown["key"] = None
        dlg.open()

    async def click(e, s):
        if not teach_on.value:
            return
        try:
            msg = await run.io_bound(E.teach, s, "cam" if kind.value == "cam" else "top", e.image_x, e.image_y,
                                     teach_color.value or None)
        except ValueError as err:
            ui.notify(str(err), type="warning")
            return
        sliders.load()
        ui.notify(msg + ". The detection updates in a moment; Save keeps it.", type="positive", multi_line=True)
    for s, img in imgs.items():
        img.on_mouse(lambda e, s=s: click(e, s))
    full_img.on_mouse(lambda e: click(e, "right" if side.value == "right" else "left"))

    shown = {"key": None, "rows": None, "t": 0.0}

    class Panel:
        def tick(self):
            d = E.det
            dkey = d["key"] if d else None
            key = (E.frame_id, dkey, kind.value, side.value, mask.value, show_misses.value, show_lone.value, full["open"],
                   E.params_id)
            if key == shown["key"] or time.monotonic() - shown["t"] < 0.15:
                return
            shown["key"], shown["t"] = key, time.monotonic()
            k = f"{E.frame_id}-{dkey[0] if dkey else 0}-{dkey[1] if dkey else 0}-{E.params_id}"
            q = f"?k={k}&mask={mask.value}&misses={int(show_misses.value)}&lone={int(show_lone.value)}"
            for s, img in imgs.items():
                img.set_visibility(side.value in (s, "both"))
                img.set_source(f"/v2img/{kind.value}_{s}.jpg{q}")
            if full["open"]:
                s = "right" if side.value == "right" else "left"
                full_title.text = f"{s} camera · {dict(cam='image', top='top view', edges='edges')[kind.value]}"
                full_img.set_source(f"/v2img/{kind.value}_{s}.jpg{q}")
            if d:
                rows = [{"c": p.color, "xyz": f"{st[0]:.0f}, {st[1]:.0f}, {st[2]:.0f}"} for p, st in d["matched"]
                        if st is not None]
                for side_name, alone in zip(("left", "right"), d["lone"]):
                    for p in alone[:8]:
                        rows.append({"c": f"{p.color} ?", "xyz": f"{side_name} only: {d['notes'].get(id(p), '')}"})
                if rows != shown["rows"]:
                    shown["rows"] = rows
                    table.rows = rows
                    table.update()
    p = Panel()
    p.live = live
    return p


def threshold_sliders(E):
    """Sliders bound to E.params through E.set_param. load() puts the current values back (after a click-to-tune)."""
    widgets = {}
    with ui.column().classes("w-full gap-1"):
        for name, c in E.params["colors"].items():
            ui.label(name).classes("text-sm font-medium mt-1").style(f"color: {HEX[name]}")
            lo, hi = (-60, 60) if name == "red" else (0, 180)
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label("hue").classes("text-xs w-8")
                widgets[name, "h"] = ui.range(min=lo, max=hi, value={"min": c["h"][0], "max": c["h"][1]}) \
                    .props("label").classes("flex-1")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label("S ≥").classes("text-xs w-8")
                widgets[name, "s"] = ui.slider(min=0, max=255, value=c["s"]).props("label").classes("flex-1")
                ui.label("V ≥").classes("text-xs w-8")
                widgets[name, "v"] = ui.slider(min=0, max=255, value=c["v"]).props("label").classes("flex-1")
        ui.label("shape").classes("text-sm font-medium mt-1")
        for key, label in (("min_fill", "fill ≥"), ("min_rim", "rim ≥"), ("max_around", "around ≤")):
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label(label).classes("text-xs w-14")
                widgets[key] = ui.slider(min=0.2, max=1.0, step=0.05, value=E.params[key]).props("label").classes("flex-1")
        ui.label("outline (rim)").classes("text-sm font-medium mt-1")
        widgets["mask_edges"] = ui.switch("Outline from the mask too", value=E.params["mask_edges"]).props("dense") \
            .tooltip("The edge of the colour mask counts as the piece's outline too: a piece the mask shows well is "
                     "found even where the picture has no sharp edge")
        for key, label, lo, hi, step in (("edge_pct", "edges ≥ %", 50, 98, 1), ("rim_band", "band ±", 0.04, 0.3, 0.02),
                                         ("size", "size ×", 0.8, 1.25, 0.01)):
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label(label).classes("text-xs w-14")
                widgets[key] = ui.slider(min=lo, max=hi, step=step, value=E.params[key]).props("label").classes("flex-1")
        ui.label("Edges ≥: an edge counts as outline if it is stronger than this share of all edges in the picture (90: "
                 "the strongest 10 %). A busy background (carpet) raises the bar for every piece: lower finds softer "
                 "outlines. Band: how far from the expected radius an edge may lie. Size: the top's radius the detector "
                 "looks for (1 = 35 mm); try it if the circles sit too big or too small on the pieces.") \
            .classes("text-xs opacity-60")
        ui.label("stereo").classes("text-sm font-medium mt-1")
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            ui.label("match ≤ mm").classes("text-xs w-14")
            widgets["match_mm"] = ui.slider(min=5, max=40, step=1, value=E.params["match_mm"]).props("label").classes("flex-1")
        with ui.row().classes("w-full items-center no-wrap gap-2"):
            ui.label("height ± mm").classes("text-xs w-14")
            widgets["max_dz_mm"] = ui.slider(min=5, max=45, step=1, value=E.params["max_dz_mm"]).props("label") \
                .classes("flex-1").tooltip("How far from the top-face height the two cameras' sighting may be")
        ui.label("How far apart the two cameras may put the same piece and still pair it. Grey pieces with ? say how "
                 "far the other camera's sighting is: if that is more than a few mm everywhere, the calibration is off "
                 "(capture the board on the table again, add lens views); raising this is the quick fix.") \
            .classes("text-xs opacity-60")
        ui.label("Fill: share of the top that has the colour (lower finds more). Rim: share of the top's edge that is "
                 "visible (lower finds more). Around: share of a ring just outside the top with the same colour; a "
                 "piece has other things around it, a blue-ish wall or cloth doesn't (lower rejects more surfaces).") \
            .classes("text-xs opacity-60")
    loading = {"on": False}

    def changed(key, value):
        if loading["on"]:
            return

        def put(p):
            if isinstance(key, tuple):
                name, k = key
                p["colors"][name][k] = (int(value["min"]), int(value["max"])) if k == "h" else int(value)
            elif key == "mask_edges":
                p[key] = bool(value)
            else:
                p[key] = float(value)
        E.set_param(put)
    for key, w in widgets.items():
        w.on_value_change(lambda e, key=key: changed(key, e.value))

    class Sliders:
        def load(self):
            loading["on"] = True
            try:
                for key, w in widgets.items():
                    if isinstance(key, tuple):
                        name, k = key
                        v = E.params["colors"][name][k]
                        w.value = {"min": v[0], "max": v[1]} if k == "h" else v
                    else:
                        w.value = E.params[key]
            finally:
                loading["on"] = False
    return Sliders()


def scene_panel(E):
    with ui.column().classes("w-full gap-1"):
        scene = ui.scene(grid=(2, 40), background_color="#18181b").classes("w-full h-[calc(100vh-200px)]")
        no_webgl = ui.label("The 3D view needs WebGL, which this browser doesn't provide.").classes("text-amber-400 text-sm")
        no_webgl.visible = False
        with ui.row().classes("items-center gap-1"):
            ui.button(icon="zoom_in", on_click=lambda: zoom(0.7)).props("flat dense").tooltip("Zoom in")
            ui.button(icon="zoom_out", on_click=lambda: zoom(1.4)).props("flat dense").tooltip("Zoom out")
            ui.button(icon="center_focus_strong", on_click=lambda: fit()).props("flat dense").tooltip("Fit")
            ui.label("Drag: rotate · right-drag: pan · wheel: zoom. Grid: 5 cm. Axes: x red, y green, z blue. "
                     "Solid pieces: both cameras see them; faint: one camera only.").classes("text-xs opacity-60 ml-2")
    objs = {"static": [], "pieces": []}
    shown = {"calib": None, "pieces": None}
    m = 1e-3

    def draw_static():
        for o in objs["static"]:
            o.delete()
        objs["static"].clear()
        if E.calib is None or E.calib.T_left_table is None:
            return
        with scene:
            add = objs["static"].append
            add(scene.axes_helper(0.1))
            outline = E.board_outline()
            if outline is not None:
                for a, b in zip(outline * m, np.roll(outline, -1, axis=0) * m):
                    add(scene.line([float(a[0]), float(a[1]), 0], [float(b[0]), float(b[1]), 0]).material("#d4d4d8"))
            for name, color in (("left", "#d97706"), ("right", "#7c3aed")):
                cam, T = pieces.camera_table_pose(E.calib, name)
                Tinv = np.linalg.inv(T)
                C = Tinv[:3, 3] * m
                Kinv = np.linalg.inv(np.asarray(cam.K))
                w, h = cam.image_size
                corners = [Tinv[:3, :3] @ (Kinv @ np.array([u, v, 1.0]) * 80.0) * m + C
                           for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
                for i in range(4):
                    add(scene.line(C.tolist(), corners[i].tolist()).material(color))
                    add(scene.line(corners[i].tolist(), corners[(i + 1) % 4].tolist()).material(color))
                add(scene.sphere(0.008).move(*map(float, C)).material(color))
                add(scene.text(name, "background: rgba(0,0,0,.6); color: #fff; padding: 1px 4px; border-radius: 3px")
                    .move(*map(float, C)))
        fit()

    def draw_pieces(matched):
        for o in objs["pieces"]:
            o.delete()
        objs["pieces"].clear()
        with scene:
            for p, st in matched:
                x, y = (st[0], st[1]) if st is not None else p.table_mm[:2]
                z = p.table_mm[2]
                cyl = scene.cylinder(pieces.PIECE_D_MM / 2 * m, pieces.PIECE_D_MM / 2 * m, pieces.PIECE_H_MM * m, 32)
                cyl.rotate(np.pi / 2, 0, 0).move(float(x) * m, float(y) * m, float(z - pieces.PIECE_H_MM / 2) * m) \
                    .material(HEX[p.color], opacity=1.0 if st is not None else 0.3)
                objs["pieces"].append(cyl)

    def fit():
        d = E.det
        pts = [((st if st is not None else p.table_mm)[0], (st if st is not None else p.table_mm)[1])
               for p, st in (d["matched"] if d else [])]
        outline = E.board_outline()
        if outline is not None:
            pts += [tuple(p) for p in outline]
        if not pts:
            pts = [(0, 0), (200, 125)]
        pts = np.array(pts, float) * m
        mid, size = pts.mean(0), max(float(np.ptp(pts, 0).max()), 0.15)
        scene.move_camera(float(mid[0]) + 0.5 * size, float(mid[1]) - 1.3 * size, 1.3 * size,
                          float(mid[0]), float(mid[1]), 0.0, duration=0.4)

    async def zoom(factor):
        cam = await scene.get_camera()
        pos = np.array([cam["position"][k] for k in ("x", "y", "z")])
        c = scene.camera
        at = np.array([c.look_at_x, c.look_at_y, c.look_at_z])
        scene.move_camera(*map(float, at + (pos - at) * factor), *map(float, at), duration=0.3)

    async def check_webgl():
        ok = await ui.run_javascript(
            "(() => { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); })()")
        no_webgl.visible = not ok
    ui.timer(1.0, check_webgl, once=True)

    class Panel:
        def tick(self):
            if shown["calib"] != E.calib_id:
                shown["calib"] = E.calib_id
                draw_static()
            d = E.det
            sig = None if not d else [(p.color, round(p.table_mm[0]), round(p.table_mm[1]), st is not None)
                                      for p, st in d["matched"]]
            if d and sig != shown["pieces"]:
                shown["pieces"] = sig
                draw_pieces(d["matched"])
    return Panel()


def add_source_args(ap):
    ap.add_argument("--files", nargs=2, metavar=("LEFT", "RIGHT"),
                    help="image files instead of cameras: comma-separated globs per camera (development)")
    ap.add_argument("--snapshot", metavar="DIR_OR_ZIP",
                    help="open a debug snapshot (Detection tab, bug button): its pictures, calibration and settings")


def engine_from_args(args):
    if args.snapshot:
        d = Path(args.snapshot).resolve()
        if d.suffix == ".zip":                  # unpacked next to it, once
            zipped, d = d, d.with_suffix("")
            if not d.exists():
                with zipfile.ZipFile(zipped) as z:
                    z.extractall(d)
        if not (d / "left.png").exists() or not (d / "calib.json").exists():
            raise SystemExit(f"{d} isn't a debug snapshot (no left.png / calib.json)")
        return Engine((str(d / "left.png"), str(d / "right.png")), data=d)
    return Engine(tuple(args.files) if args.files else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    E = engine_from_args(args)
    try:
        E.connect()
    except Exception as e:
        print(f"cameras not connected: {e}")
    threading.Thread(target=E.load_captures, daemon=True).start()
    build_ui(E)
    app.on_shutdown(E.disconnect)
    ui.run(title="Calibration v2", port=args.port, reload=False, dark=True, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
