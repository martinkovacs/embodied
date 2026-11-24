"""Piece detection with a trained YOLO model: a box per piece in the camera image, with its colour and confidence.

The model was trained on camera pictures (yolo_capture.py) with three classes, red_cylinder, blue_cylinder and
green_cylinder; each box holds the whole piece as the camera sees it (top and side). No calibration needed, and no
table position yet: only where the pieces are in each picture.

ultralytics (and with it torch) is imported on first use only, so the rest of the app runs without it.

Usage:
    venv/bin/python yolo_pieces.py samples/pieces_01.jpg [--weights best.pt] [--conf 0.5]
"""
import argparse
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# the trained model; YOLO_WEIGHTS (environment) or --yolo-weights (calib_app.py) points elsewhere
WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", "/home/martin/Dokumentumok/yolo/runs/detect/train/weights/best.pt"))
IMGSZ = 1280            # the size it was trained at
CONF = 0.5
DRAW = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 180, 0)}


@dataclass
class Box:
    color: str                                  # red / blue / green
    conf: float
    xyxy: tuple[float, float, float, float]     # camera pixels

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2


class Detector:
    """The model, loaded on the first detect() (a few seconds). Thread-safe."""

    def __init__(self, weights=WEIGHTS, imgsz=IMGSZ):
        self.weights, self.imgsz = Path(weights), imgsz
        self._model = None
        self._lock = threading.Lock()

    @property
    def device(self):
        m = self._model
        pred = getattr(m, "predictor", None) if m is not None else None
        return str(pred.device) if pred is not None else None

    def _load(self):
        if self._model is None:
            if not self.weights.exists():
                raise FileNotFoundError(f"no YOLO model at {self.weights}")
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError("ultralytics is not installed (venv/bin/python -m pip install ultralytics)") from e
            self._model = YOLO(str(self.weights), task="detect")
        return self._model

    def detect(self, images, conf=CONF):
        """Boxes in each BGR image: [[Box], ...], one list per image, most confident first. Images that are None
        give []."""
        idx = [i for i, im in enumerate(images) if im is not None]
        out = [[] for _ in images]
        if not idx:
            return out
        with self._lock:
            model = self._load()
            results = model.predict([images[i] for i in idx], imgsz=self.imgsz, conf=conf, verbose=False)
        for i, r in zip(idx, results):
            b = r.boxes
            for xyxy, c, k in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                name = r.names[k].split("_")[0]          # red_cylinder -> red
                out[i].append(Box(name, float(c), tuple(float(v) for v in xyxy)))
            out[i].sort(key=lambda bx: -bx.conf)
        return out


def draw(img, boxes):
    """The boxes on a copy of the image, each with its colour and confidence."""
    vis = img.copy()
    t = max(2, round(img.shape[1] / 800))
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b.xyxy)
        col = DRAW.get(b.color, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, t)
        label = f"{b.color} {b.conf:.2f}"
        s = 0.3 * t
        org = (x1, max(y1 - 6, 14))
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, t, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--out", help="folder for the pictures with the boxes drawn on")
    a = ap.parse_args()
    det = Detector(a.weights)
    imgs = [cv2.imread(p) for p in a.images]
    for path, img, boxes in zip(a.images, imgs, det.detect(imgs, a.conf)):
        print(f"{path}: " + (", ".join(f"{b.color} {b.conf:.2f} @ ({b.center[0]:.0f}, {b.center[1]:.0f})"
                                       for b in boxes) or "nothing"))
        if a.out and img is not None:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(a.out) / (Path(path).stem + "_yolo.jpg")), draw(img, boxes))
    print(f"device: {det.device}")


if __name__ == "__main__":
    main()
