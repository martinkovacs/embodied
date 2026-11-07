"""Camera sources + shared settings for the stereo calibration GUI and the runner.

Source is a camera (spec is an index) or a sorted list of image files (spec is a glob), so image
files can stand in for the cameras in test mode. Paths are anchored at this folder, so callers don't
depend on the working directory.
"""
import glob
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "calib_data"
CALIB_PATH = DATA / "calib.json"
SETTINGS_PATH = DATA / "settings.json"

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
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
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
