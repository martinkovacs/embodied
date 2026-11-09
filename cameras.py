"""Camera sources + shared settings for the stereo calibration GUI and the runner.

Source is a camera (spec is an index) or a sorted list of image files (spec is a glob), so image
files can stand in for the cameras in test mode. Paths are anchored at this folder, so callers don't
depend on the working directory.
"""
import fcntl
import glob
import json
import struct
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "calib_data"
CALIB_PATH = DATA / "calib.json"
SETTINGS_PATH = DATA / "settings.json"
SYNTHETIC_CALIB = ROOT / "synthetic" / "calib.json"     # goes with the synthetic renders, whatever the lab calibrated
SYNTHETIC_BOARD_MM = 0.8                                 # the rendered board's thickness
TEST_DATA = ROOT / "synthetic" / "gui"                   # calib_gui.py in test mode: never mixes with the lab's DATA

STAGES = {"calib": "Calibration pairs", "table": "Table pose", "scene": "Detection scene"}
# Test mode: image files stand in for the cameras, one left/right set per step (paths relative to ROOT).
TEST_FILES = {"calib": ("synthetic/calib/left_*.png", "synthetic/calib/right_*.png"),
              "table": ("synthetic/table_left.png", "synthetic/table_right.png"),
              "scene": ("synthetic/scene_left.png", "synthetic/scene_right.png")}
DEFAULTS = {"left": "0", "right": "1", "width": 1280, "height": 960,
            "cols": 9, "rows": 6, "square_mm": 25.0, "board_mm": 0.0, "test_mode": False,
            **{f"test_{st}_{side}": TEST_FILES[st][i] for st in STAGES for i, side in enumerate(("left", "right"))}}

# grab() past the frames a webcam queued while nobody read; a single grab() after a pause returns a stale frame
FLUSH_FRAMES = 5


def list_cameras():
    """[(index, name)] of the V4L2 nodes that capture video. A UVC camera also has a metadata node that OpenCV can't
    read, so its index in /dev/video* isn't simply the next free number. Empty where there is no V4L2."""
    VIDIOC_QUERYCAP, CAP_VIDEO_CAPTURE = 0x80685600, 0x1
    out = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d.removeprefix("/dev/video"))):
        try:
            with open(dev, "rb") as f:
                buf = fcntl.ioctl(f, VIDIOC_QUERYCAP, bytes(104))
        except OSError:
            continue
        if struct.unpack_from("<I", buf, 88)[0] & CAP_VIDEO_CAPTURE:    # device_caps: this node, not the whole device
            out.append((int(dev.removeprefix("/dev/video")), buf[16:48].split(b"\0")[0].decode(errors="replace")))
    return out


def fresh_pair(left, right):
    """Both frames captured after this call, alternating grabs so the two are as close in time as possible."""
    for _ in range(FLUSH_FRAMES):
        left.grab()
        right.grab()
    return left.retrieve(), right.retrieve()


def load_settings(path=SETTINGS_PATH) -> dict:
    """Defaults overridden by the saved settings, or just the defaults if there is no file yet."""
    return {**DEFAULTS, **(json.loads(path.read_text()) if path.exists() else {})}


class Source:
    """A camera (spec is an index) or a sorted list of image files (spec is a glob)."""

    def __init__(self, spec: str, width: int, height: int):
        self.spec = spec.strip()
        self.cap, self.files, self.idx = None, [], 0
        if self.spec.isdigit():
            self.cap = cv2.VideoCapture(int(self.spec))
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera {self.spec}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # before the size; two raw streams
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)                       # this large don't fit on one USB bus
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; some backends ignore it
        else:
            pattern = self.spec if Path(self.spec).is_absolute() else str(ROOT / self.spec)
            self.files = sorted(glob.glob(pattern))
            if not self.files:
                raise RuntimeError(f"no images match {self.spec}")

    @property
    def is_folder(self):
        return self.cap is None

    def grab(self):
        if self.cap is not None:
            self.cap.grab()

    def retrieve(self):
        if self.cap is not None:
            ok, frame = self.cap.retrieve()
            return frame if ok else None
        return cv2.imread(self.files[self.idx])

    def label(self):
        return f"camera {self.spec}" if self.cap is not None else f"{self.idx + 1}/{len(self.files)} {Path(self.files[self.idx]).name}"

    def release(self):
        if self.cap is not None:
            self.cap.release()
