"""Camera sources + shared settings for the stereo calibration GUI and the runner.

Source is a camera (spec is an OpenCV index, or "basler:<serial number>") or a sorted list of image files
(spec is a glob), so image files can stand in for the cameras in test mode. Paths are anchored at this folder, so callers don't
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
            "cols": 9, "rows": 6, "square_mm": 25.0, "board_mm": 0.0, "test_mode": False, "one_camera": False,
            **{f"test_{st}_{side}": TEST_FILES[st][i] for st in STAGES for i, side in enumerate(("left", "right"))}}

BASLER = "basler:"              # Basler cameras are USB3 / GigE Vision, not webcams: OpenCV can't see them, pypylon can

# grab() past the frames a webcam queued while nobody read; a single grab() after a pause returns a stale frame
FLUSH_FRAMES = 5


def basler_devices():
    """pypylon's device list; empty if pypylon or the cameras aren't there."""
    try:
        from pypylon import pylon
        return pylon.TlFactory.GetInstance().EnumerateDevices()
    except Exception:
        return ()


def list_cameras():
    """[(spec, name)] of the cameras Source can open: Basler cameras, then the V4L2 nodes that capture video. A UVC
    camera also has a metadata node that OpenCV can't read, so its index in /dev/video* isn't simply the next free
    number."""
    VIDIOC_QUERYCAP, CAP_VIDEO_CAPTURE = 0x80685600, 0x1
    out = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d.removeprefix("/dev/video"))):
        try:
            with open(dev, "rb") as f:
                buf = fcntl.ioctl(f, VIDIOC_QUERYCAP, bytes(104))
        except OSError:
            continue
        if struct.unpack_from("<I", buf, 88)[0] & CAP_VIDEO_CAPTURE:    # device_caps: this node, not the whole device
            out.append((dev.removeprefix("/dev/video"), buf[16:48].split(b"\0")[0].decode(errors="replace")))
    return [(BASLER + d.GetSerialNumber(), d.GetModelName()) for d in basler_devices()] + out


def fresh_pair(left, right):
    """Both frames captured after this call, alternating grabs so the two are as close in time as possible."""
    for _ in range(FLUSH_FRAMES):
        left.grab()
        right.grab()
    return left.retrieve(), right.retrieve()


def load_settings(path=SETTINGS_PATH) -> dict:
    """Defaults overridden by the saved settings, or just the defaults if there is no file yet."""
    return {**DEFAULTS, **(json.loads(path.read_text()) if path.exists() else {})}


class BaslerCamera:
    """One Basler camera behind OpenCV's grab / retrieve / release: free-running at full sensor size, grab() waits for
    the newest frame, retrieve() gives it as BGR. Exposure and white balance run automatically."""

    GRAB_TIMEOUT_MS = 2000

    def __init__(self, serial):
        from pypylon import pylon
        self.pylon, self.result = pylon, None
        dev = next((d for d in basler_devices() if d.GetSerialNumber() == serial), None)
        if dev is None:
            raise RuntimeError(f"no Basler camera with serial {serial}. Cameras found: {list_cameras() or 'none'}")
        try:
            self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev))
            self.cam.Open()
        except Exception as e:
            raise RuntimeError(f"cannot open Basler camera {serial} (another program using it?): {e}") from None
        self.configure()
        self.convert = pylon.ImageFormatConverter()
        self.convert.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.convert.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def configure(self):
        """Best effort, per setting: not every model has every one (a mono camera has no white balance)."""
        def attempt(fn):
            try:
                fn()
            except Exception:
                pass
        c = self.cam
        for name in ("OffsetX", "OffsetY"):
            attempt(lambda: setattr(getattr(c, name), "Value", 0))
        for name in ("Width", "Height"):
            attempt(lambda: setattr(getattr(c, name), "Value", getattr(c, name).Max))
        for name in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
            attempt(lambda: setattr(getattr(c, name), "Value", "Continuous"))
        # a colour camera may default to mono output, and the pieces are told apart by colour
        def colour():
            have = c.PixelFormat.GetSymbolics()
            c.PixelFormat.Value = next(f for f in ("BayerRG8", "BayerBG8", "BayerGB8", "BayerGR8", "BGR8", "RGB8") if f in have)
        attempt(colour)

    def isOpened(self):
        return self.cam.IsOpen()

    def set(self, *_):      # the OpenCV settings Source makes: a Basler camera is configured above
        return False

    def grab(self):
        if self.result is not None:
            self.result.Release()
        try:
            self.result = self.cam.RetrieveResult(self.GRAB_TIMEOUT_MS, self.pylon.TimeoutHandling_Return)
        except Exception:   # unplugged: reads as "no image", like a webcam
            self.result = None
        return self.result is not None

    def retrieve(self):
        r = self.result
        if r is None or not r.IsValid() or not r.GrabSucceeded():
            return False, None
        return True, self.convert.Convert(r).GetArray().copy()

    def release(self):
        if self.result is not None:
            self.result.Release()
            self.result = None
        self.cam.StopGrabbing()
        self.cam.Close()


class Source:
    """A camera (spec is an index, or BASLER + serial number) or a sorted list of image files (spec is a glob)."""

    def __init__(self, spec: str, width: int, height: int):
        self.spec = spec.strip()
        self.cap, self.files, self.idx = None, [], 0
        if self.spec.startswith(BASLER) or self.spec.isdigit():
            self.cap = BaslerCamera(self.spec.removeprefix(BASLER)) if self.spec.startswith(BASLER) \
                else cv2.VideoCapture(int(self.spec))
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open camera {self.spec}. Cameras found: {list_cameras() or 'none'}")
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
