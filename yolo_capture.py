"""Take pictures for YOLO training (NiceGUI): both cameras live, Capture saves one picture from each, Download zip
gives them all at the end.

The cameras are the ones calib_app.py set up (calib_data/settings.json: left, right, width, height), unless --left /
--right say otherwise. Close calib_app.py first: a camera can only be open in one program.

Pictures go to yolo_data/<session>/ as 0001_left.jpg, 0001_right.jpg, ... (a new session folder per start, or
--session NAME to add to an existing one). Capture: the button or the space bar. Delete last: the last pair, if it
came out wrong.

Run:  ./lab.sh python yolo_capture.py                   (then open http://localhost:8081)
      ./lab.sh python yolo_capture.py --left 0 --right 2 --png
"""
import argparse
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi.responses import Response
from nicegui import app, run, ui

from cameras import ROOT, Source, fresh_pair, list_cameras, load_settings

PORT = 8081                 # calib_app.py has 8080
OUT = ROOT / "yolo_data"
SIDES = ("left", "right")
PREVIEW_W = 960             # the live view is sent this wide; the saved pictures are full size


class Capturer:
    """Both cameras, a grab thread for the live view, and the session folder the captures go to."""

    def __init__(self, left, right, width, height, folder: Path, ext):
        self.spec = {"left": left, "right": right}
        self.size = width, height
        self.folder, self.ext = folder, ext
        self.lock = threading.RLock()
        self.cams = None
        self.error = None
        self.frames = (None, None)
        self.frame_id = 0
        self.folder.mkdir(parents=True, exist_ok=True)
        self.n = self._last_index()
        threading.Thread(target=self._grab_loop, daemon=True).start()

    def connect(self):
        with self.lock:
            self.disconnect()
            if str(self.spec["left"]) == str(self.spec["right"]):
                raise RuntimeError("left and right are the same camera")
            left = Source(str(self.spec["left"]), *self.size)
            try:
                right = Source(str(self.spec["right"]), *self.size)
            except Exception:
                left.release()
                raise
            self.cams, self.error = (left, right), None

    def disconnect(self):
        with self.lock:
            if self.cams:
                for c in self.cams:
                    c.release()
            self.cams = None

    def _grab_loop(self):
        while True:
            with self.lock:
                cams = self.cams
                if cams:
                    try:
                        for c in cams:
                            c.grab()
                        fl, fr = (c.retrieve() for c in cams)
                    except Exception as e:
                        fl = fr = None
                        self.error = f"camera error: {e}"
                    if fl is not None and fr is not None:
                        self.frames, self.frame_id, self.error = (fl, fr), self.frame_id + 1, None
                    else:
                        self.error = self.error or "a camera gives no image"
            time.sleep(0.03 if cams else 0.2)

    # -- captures
    def files(self):
        return sorted(self.folder.glob("*_left.*")) + sorted(self.folder.glob("*_right.*"))

    def pairs(self):
        """The capture numbers in the folder, in order."""
        return sorted({int(p.name.split("_")[0]) for p in self.files() if p.name.split("_")[0].isdigit()})

    def _last_index(self):
        return max(self.pairs(), default=0)

    def capture(self):
        """A fresh frame from each camera (taken after the click, not the one on screen), saved. Returns the number."""
        with self.lock:
            if not self.cams:
                raise RuntimeError("cameras not connected")
            fl, fr = fresh_pair(*self.cams)
            if fl is None or fr is None:
                raise RuntimeError("a camera gives no image")
            self.n += 1
            n = self.n
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if self.ext == "jpg" else []
        for side, img in zip(SIDES, (fl, fr)):
            if not cv2.imwrite(str(self.folder / f"{n:04d}_{side}.{self.ext}"), img, params):
                raise RuntimeError(f"cannot write to {self.folder}")
        return n

    def delete_last(self):
        nums = self.pairs()
        if not nums:
            return None
        n = nums[-1]
        for p in self.folder.glob(f"{n:04d}_*"):
            p.unlink()
        with self.lock:
            self.n = max(self.pairs(), default=0)
        return n

    def make_zip(self):
        """All of this session's pictures in one zip next to the folder; returns its path."""
        files = self.files()
        if not files:
            raise RuntimeError("no pictures yet")
        path = self.folder.with_suffix(".zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:     # jpg/png are compressed already
            for f in files:
                z.write(f, f"{self.folder.name}/{f.name}")
        return path

    def preview(self, side):
        img = self.frames[SIDES.index(side)]
        if img is None:
            img = np.full((480, 640, 3), 40, np.uint8)
            cv2.putText(img, self.error or "no picture yet", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2, cv2.LINE_AA)
        elif img.shape[1] > PREVIEW_W:
            img = cv2.resize(img, (PREVIEW_W, round(img.shape[0] * PREVIEW_W / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def build_ui(C: Capturer):
    @app.get("/live/{side}.jpg")
    def live(side: str):        # FastAPI runs a plain def in a thread
        if side not in SIDES:
            return Response(status_code=404)
        return Response(C.preview(side), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @ui.page("/")
    def page():
        ui.page_title("YOLO capture")
        with ui.header().classes("items-center bg-slate-800 gap-3"):
            ui.label("YOLO capture").classes("text-lg font-medium")
            ui.label(f"→ {C.folder.relative_to(ROOT)}").classes("text-sm opacity-80")
            ui.space()
            status = ui.label().classes("text-sm")

        with ui.row().classes("w-full no-wrap gap-2"):
            imgs = {}
            for side in SIDES:
                with ui.column().classes("w-1/2 gap-1"):
                    ui.label(f"{side} · camera {C.spec[side]}").classes("text-sm opacity-70")
                    imgs[side] = ui.interactive_image().classes("w-full")

        with ui.row().classes("items-center gap-3"):
            cap_btn = ui.button("Capture", icon="photo_camera").props("size=lg color=primary")
            ui.button("Delete last", icon="undo", on_click=lambda: delete_last()).props("flat")
            zip_btn = ui.button("Download zip", icon="download").props("outline")
            ui.button("Reconnect", icon="refresh", on_click=lambda: reconnect()).props("flat")
            count = ui.label().classes("text-lg")
            ui.label("space = capture").classes("text-sm opacity-60")

        def refresh_count():
            n = len(C.pairs())
            count.text = f"{n} pair{'s' if n != 1 else ''} saved"

        busy = {"capture": False}

        async def capture():
            if busy["capture"]:
                return
            busy["capture"] = True
            cap_btn.props("loading")
            try:
                n = await run.io_bound(C.capture)
                ui.notify(f"Saved pair {n:04d}", type="positive", timeout=800)
            except Exception as e:
                ui.notify(f"Not saved: {e}", type="negative")
            finally:
                busy["capture"] = False
                cap_btn.props(remove="loading")
                refresh_count()
        cap_btn.on_click(capture)

        def delete_last():
            n = C.delete_last()
            ui.notify(f"Deleted pair {n:04d}" if n else "Nothing to delete")
            refresh_count()

        async def download():
            zip_btn.props("loading")
            try:
                path = await run.io_bound(C.make_zip)
                ui.download.file(path)
            except Exception as e:
                ui.notify(f"No zip: {e}", type="negative")
            finally:
                zip_btn.props(remove="loading")
        zip_btn.on_click(download)

        async def reconnect():
            try:
                await run.io_bound(C.connect)
                ui.notify("Cameras connected", type="positive")
            except Exception as e:
                ui.notify(f"Not connected: {e}", type="negative", multi_line=True)

        def on_key(e):
            if e.key == " " and e.action.keydown and not e.action.repeat:
                return capture()
        ui.keyboard(on_key=on_key, ignore=["input", "select", "button", "textarea"])

        def tick():
            for side, img in imgs.items():
                img.set_source(f"/live/{side}.jpg?k={C.frame_id}")
            status.text = C.error or ("live" if C.cams else "not connected")
            status.classes(replace="text-sm " + ("text-red-400" if C.error or not C.cams else "text-green-400"))
        ui.timer(0.15, tick)
        refresh_count()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = load_settings()
    ap.add_argument("--left", default=str(s["left"]), help="left camera: OpenCV index or basler:<serial>")
    ap.add_argument("--right", default=str(s["right"]), help="right camera: OpenCV index or basler:<serial>")
    ap.add_argument("--width", type=int, default=int(s["width"]))
    ap.add_argument("--height", type=int, default=int(s["height"]))
    ap.add_argument("--session", help="folder name under yolo_data/ (default: the date and time)")
    ap.add_argument("--png", action="store_true", help="save PNG (lossless, big) instead of JPEG quality 95")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    folder = OUT / (args.session or datetime.now().strftime("%Y%m%d_%H%M%S"))
    C = Capturer(args.left, args.right, args.width, args.height, folder, "png" if args.png else "jpg")
    try:
        C.connect()
    except Exception as e:
        print(f"cameras not connected: {e}\ncameras found: {list_cameras() or 'none'}")
    build_ui(C)
    app.on_shutdown(C.disconnect)
    ui.run(title="YOLO capture", port=args.port, reload=False, dark=True, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
