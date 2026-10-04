"""Detect red and blue balls and draw bounding boxes.

Two stages:
  1. YOLOE (open-vocabulary, text prompt "ball") proposes class-agnostic ball boxes + masks.
  2. A per-object classifier labels each proposal. For now: hue voting inside the mask.
     Swap `classify` for e.g. a DINOv2 kNN on crops when balls differ by symbol/shape.

Usage:
    .venv/bin/python detect.py renders/ --out detections/
    .venv/bin/python detect.py img1.png img2.jpg --show
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

# OpenCV hue is 0..179. Red wraps around 0.
HUE_RANGES = {
    "red": [(0, 10), (170, 180)],
    "blue": [(95, 135)],
}
DRAW_COLORS = {"red": (0, 0, 255), "blue": (255, 0, 0), "unknown": (128, 128, 128)}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def classify(img_bgr, mask, min_sat=80, min_val=40, min_frac=0.5):
    """Return (label, score): majority hue class among saturated pixels inside the mask."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[mask & (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val)]
    if len(px) == 0:
        return "unknown", 0.0
    h = px[:, 0]
    fracs = {name: sum(((h >= lo) & (h < hi)).sum() for lo, hi in ranges) / len(h)
             for name, ranges in HUE_RANGES.items()}
    label = max(fracs, key=fracs.get)
    return (label, fracs[label]) if fracs[label] >= min_frac else ("unknown", fracs[label])


def detect(model, img_bgr, imgsz, conf):
    """Return list of dicts: box (x1, y1, x2, y2), label, det_conf, cls_score."""
    r = model.predict(img_bgr, imgsz=imgsz, conf=conf, retina_masks=True, quantize=16, verbose=False)[0]
    if r.masks is None:
        return []
    masks = r.masks.data.cpu().numpy().astype(bool)
    out = []
    for box, det_conf, mask in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), masks):
        label, score = classify(img_bgr, mask)
        out.append({"box": box.round().astype(int).tolist(), "label": label,
                    "det_conf": float(det_conf), "cls_score": float(score)})
    return out


def draw(img_bgr, dets):
    vis = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = DRAW_COLORS[d["label"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d['label']} {d['det_conf']:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def load_model(base, prompts):
    """Load YOLOE with the text prompts baked into the weights.

    The text encoder (mobileclip2_b.ts, ~250 MB VRAM) is only needed to embed the prompts, so the
    first run embeds them and saves e.g. yoloe-26m-seg_ball.pt; later runs load that file directly
    and never touch the encoder. Deploy only the baked file.
    """
    baked = Path(base).with_name(f"{Path(base).stem}_{'+'.join(prompts).replace(' ', '-')}.pt")
    if baked.exists():
        return YOLOE(str(baked))
    model = YOLOE(base)
    model.set_classes(prompts)
    model.save(str(baked))
    print(f"baked prompts {prompts} into {baked}")
    return YOLOE(str(baked))  # reload so the encoder is freed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or directories")
    ap.add_argument("--out", default="detections", help="output directory for annotated images")
    ap.add_argument("--model", default="yoloe-26m-seg.pt", help="base YOLOE weights (prompts get baked in)")
    ap.add_argument("--prompt", default="ball", help="comma-separated YOLOE text prompts")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; keep high, balls are small")
    ap.add_argument("--conf", type=float, default=0.05, help="YOLOE text-prompt scores run low")
    ap.add_argument("--show", action="store_true", help="also display results in a window")
    args = ap.parse_args()

    paths = []
    for p in map(Path, args.inputs):
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]

    model = load_model(args.model, [s.strip() for s in args.prompt.split(",")])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        img = cv2.imread(str(path))
        dets = detect(model, img, args.imgsz, args.conf)
        counts = {k: sum(d["label"] == k for d in dets) for k in DRAW_COLORS}
        print(f"{path.name}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for d in dets:
            print(f"  {d['label']:7s} box={d['box']} det={d['det_conf']:.2f} cls={d['cls_score']:.2f}")
        vis = draw(img, dets)
        cv2.imwrite(str(out_dir / path.name), vis)
        if args.show:
            cv2.imshow("detections", vis)
            if cv2.waitKey(0) == 27:
                break


if __name__ == "__main__":
    main()
