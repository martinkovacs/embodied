"""Detect cylindrical pieces (red / blue / green / white, 35 mm diameter, 30 mm tall) by colour + known size.

Calibrated mode (--calib): the camera image is remapped to a top-down view of the plane at piece-top
height, where every top face is an exact circle of known pixel radius. Candidates (fixed-radius Hough +
colour-disk peaks) are scored by rim edges and colour fill. Positions come out in table mm.

Uncalibrated mode (--radius-px): same detector run directly on the image. Only sensible for a camera
looking roughly straight down; with an oblique view the tops are ellipses and detection degrades.

Usage:
    venv/bin/python pieces.py synthetic/scene_left.png --calib calib.json --camera left
    venv/bin/python pieces.py samples/pieces_01.jpg --radius-px 60
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PIECE_D_MM = 35.0
PIECE_H_MM = 30.0

# HSV ranges, OpenCV convention (H 0..179). Red wraps around 0.
COLORS = {
    "red": {"h": [(0, 10), (170, 180)], "s": (100, 255), "v": (40, 255)},
    "blue": {"h": [(100, 135)], "s": (100, 255), "v": (20, 255)},
    "green": {"h": [(40, 90)], "s": (50, 255), "v": (40, 255)},
    "white": {"h": [(0, 180)], "s": (0, 45), "v": (150, 255), "min_rim": 0.75},  # white table: demand a clear rim
}
DRAW = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 180, 0), "white": (255, 255, 255)}


@dataclass
class Piece:
    color: str
    view_uv: tuple[float, float]        # centre in the detection image (top view or raw image)
    image_uv: tuple[float, float]       # top-face centre in the original camera image
    table_mm: tuple[float, float, float] | None  # top-face centre in the table frame
    rim: float                          # fraction of the rim with a matching edge
    fill: float                         # fraction of the top disk with the piece colour


class TopView:
    """Remap a camera image onto the table plane at height z_mm (table frame, z up)."""

    def __init__(self, K, dist, T_cam_table, z_mm=PIECE_H_MM, ppm=2.0, bounds=None, image_size=None):
        self.K, self.dist, self.z, self.ppm = np.asarray(K, float), np.asarray(dist, float), z_mm, ppm
        self.rvec, _ = cv2.Rodrigues(np.asarray(T_cam_table, float)[:3, :3])
        self.tvec = np.asarray(T_cam_table, float)[:3, 3]
        self.T = np.asarray(T_cam_table, float)
        x0, x1, y0, y1 = bounds or self.visible_bounds(image_size)
        self.x0, self.y1 = x0, y1
        xs = np.arange(x0, x1, 1 / ppm)
        ys = np.arange(y1, y0, -1 / ppm)  # image rows go down = table y decreasing, so the view isn't mirrored
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx, gy, np.full_like(gx, z_mm)], -1).reshape(-1, 3)
        self.maps = self.project(pts).reshape(len(ys), len(xs), 2).astype(np.float32)

    def visible_bounds(self, image_size, limit=400.0):
        """Table-plane bbox (x0, x1, y0, y1) seen by the camera, clipped to +-limit mm."""
        w, h = image_size
        border = np.array([(u, v) for t in np.linspace(0, 1, 20)
                           for u, v in ((t * w, 0), (t * w, h), (0, t * h), (w, t * h))], np.float32)
        rays = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        R_tc, origin = self.T[:3, :3].T, -self.T[:3, :3].T @ self.tvec  # camera->table
        d = (R_tc @ np.c_[rays, np.ones(len(rays))].T).T
        ok = d[:, 2] < -1e-6
        s = (self.z - origin[2]) / d[ok, 2]
        p = origin[:2] + s[:, None] * d[ok, :2]
        p = np.clip(p, -limit, limit)
        return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()

    def project(self, pts_table):
        uv, _ = cv2.projectPoints(np.asarray(pts_table, float).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    def warp(self, img):
        return cv2.remap(img, self.maps[..., 0], self.maps[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def view_to_table(self, u, v):
        return self.x0 + u / self.ppm, self.y1 - v / self.ppm


def color_masks(bgr, colors=COLORS):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    out = {}
    for name, c in colors.items():
        hm = np.zeros(H.shape, bool)
        for lo, hi in c["h"]:
            hm |= (H >= lo) & (H < hi)
        out[name] = hm & (S >= c["s"][0]) & (S <= c["s"][1]) & (V >= c["v"][0]) & (V <= c["v"][1])
    return out


def _disk(r):
    r = max(int(round(r)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def detect(bgr, r_px, colors=COLORS, min_rim=0.6, min_fill=0.55, hough_votes=None):
    """Find top-face circles of radius r_px. Returns [(color, u, v, rim, fill)]."""
    sigma = max(r_px / 25, 1.0)
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), sigma)
    # Colour gradient: per pixel, the BGR channel with the strongest gradient. Grey alone misses
    # e.g. red/green boundaries, which have about the same brightness.
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma).astype(np.float32)
    gxs, gys = cv2.Sobel(blur, cv2.CV_32F, 1, 0), cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    best = (gxs ** 2 + gys ** 2).argmax(axis=2)[..., None]
    gx, gy = np.take_along_axis(gxs, best, 2)[..., 0], np.take_along_axis(gys, best, 2)[..., 0]
    mag = cv2.magnitude(gx, gy)
    edge_thr = max(np.percentile(mag, 90), 20.0)

    masks = color_masks(bgr, colors)
    inner = _disk(0.8 * r_px).astype(np.float32)
    inner /= inner.sum()
    fills = {c: cv2.filter2D(m.astype(np.float32), -1, inner, borderType=cv2.BORDER_CONSTANT) for c, m in masks.items()}
    # Mean brightness of the disk: the top face is lit from above, the piece's side (smeared outwards
    # in the top view) is darker, so among overlapping candidates the brightest one is the top.
    bright = cv2.filter2D(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255, -1, inner,
                          borderType=cv2.BORDER_CONSTANT)

    # Candidates: colour-disk peaks + fixed-radius Hough on the grey image.
    cands = []
    for f in fills.values():
        peaks = ((f >= cv2.dilate(f, _disk(0.3 * r_px))) & (f >= min_fill)).astype(np.uint8)
        # one candidate per peak region (flat plateaus, e.g. a white table, would give millions of pixels)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(peaks)
        cands += [tuple(c) for c, st in zip(centroids[1:], stats[1:]) if st[cv2.CC_STAT_AREA] < (2 * r_px) ** 2]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=0.5 * r_px, param1=60,
                               param2=hough_votes or max(8, int(0.5 * r_px)),
                               minRadius=int(r_px * 0.9), maxRadius=int(np.ceil(r_px * 1.1)))
    if circles is not None:
        cands += [(float(x), float(y)) for x, y, _ in circles[0]]

    # Score = rim * fill * brightness. Rim: on how many of 72 rays around the circle there is an edge
    # with a radial gradient. Each candidate is refined by a grid search within +-r, because Hough and
    # colour peaks often land on the piece's side instead of its top face.
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    radii = r_px + np.arange(-0.12, 0.121, 0.04) * r_px
    cos, sin = np.cos(ang), np.sin(ang)
    h, w = gray.shape
    names = list(fills)
    fill_stack = np.stack([fills[c] for c in names])

    def score(xs, ys):
        px = np.clip((xs[:, None, None] + radii[None, :, None] * cos).round().astype(int), 0, w - 1)
        py = np.clip((ys[:, None, None] + radii[None, :, None] * sin).round().astype(int), 0, h - 1)
        m = mag[py, px]
        radial = np.abs(gx[py, px] * cos + gy[py, px] * sin) / (m + 1e-6)
        rim = ((m > edge_thr) & (radial > 0.8)).any(axis=1).mean(axis=1)
        xi, yi = np.clip(xs.round().astype(int), 0, w - 1), np.clip(ys.round().astype(int), 0, h - 1)
        f = fill_stack[:, yi, xi]
        return rim, f.argmax(0), f.max(0), bright[yi, xi]

    def grid(step, n):
        offs = np.arange(-n, n + 1) * step
        return [a.ravel() for a in np.meshgrid(offs, offs)]

    coarse, fine = grid(r_px / 8, 8), grid(r_px / 48, 6)
    results = []
    for x, y in cands:
        for ox, oy in (coarse, fine):  # coarse search within +-r, then fine within +-r/8
            xs, ys = x + ox, y + oy
            rim, ci, fill, br = score(xs, ys)
            s = rim * fill * br * ((rim >= min_rim) & (fill >= min_fill))
            k = int(s.argmax())
            x, y = xs[k], ys[k]
        name = names[ci[k]]
        if s[k] > 0 and rim[k] >= colors[name].get("min_rim", min_rim):
            results.append((name, float(x), float(y), float(rim[k]), float(fill[k]), float(br[k])))

    # Non-max suppression: touching pieces are 2r apart, duplicates (e.g. on a piece's side) are closer.
    results.sort(key=lambda r: r[3] * r[4] * r[5], reverse=True)
    kept = []
    for r in results:
        if all(np.hypot(r[1] - k[1], r[2] - k[2]) > 1.4 * r_px for k in kept):
            kept.append(r)
    return [r[:5] for r in kept]


def detect_pieces(img, view: TopView | None = None, radius_px=None, colors=COLORS):
    """Detect pieces in a camera image. With a TopView, positions are also returned in table mm."""
    if view is None:
        return [Piece(c, (u, v), (u, v), None, rim, fill) for c, u, v, rim, fill in detect(img, radius_px, colors)]
    top = view.warp(img)
    pieces = []
    for c, u, v, rim, fill in detect(top, PIECE_D_MM / 2 * view.ppm, colors):
        x, y = view.view_to_table(u, v)
        iu, iv = view.project([[x, y, view.z]])[0]
        pieces.append(Piece(c, (u, v), (float(iu), float(iv)), (x, y, view.z), rim, fill))
    return pieces


def draw(img, pieces, view: TopView | None = None, radius_px=None):
    """Draw top-face outlines + bounding boxes on the original image."""
    vis = img.copy()
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for p in pieces:
        col = DRAW[p.color]
        if view is not None:
            r = PIECE_D_MM / 2
            x, y, z = p.table_mm
            top = view.project(np.c_[x + r * np.cos(ang), y + r * np.sin(ang), np.full(48, z)])
            bottom = view.project(np.c_[x + r * np.cos(ang), y + r * np.sin(ang), np.full(48, z - PIECE_H_MM)])
            bx, by, bw, bh = cv2.boundingRect(np.vstack([top, bottom]).astype(np.float32))
            cv2.polylines(vis, [top.round().astype(np.int32)], True, col, 2, cv2.LINE_AA)
            label = f"{p.color} ({x:.0f}, {y:.0f})"
        else:
            u, v = p.image_uv
            bx, by, bw, bh = int(u - radius_px), int(v - radius_px), int(2 * radius_px), int(2 * radius_px)
            label = p.color
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 2)
        cv2.circle(vis, tuple(int(round(c)) for c in p.image_uv), 3, col, -1)
        cv2.putText(vis, label, (bx, max(by - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (bx, max(by - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return vis


def camera_table_pose(calib, camera="left"):
    """(CameraCalib, 4x4 table->camera) for one camera of a StereoCalib with a table pose."""
    if calib.T_left_table is None:
        raise ValueError("calibration has no table pose; calibrate the table first")
    T = np.asarray(calib.T_left_table, float)
    if camera == "left":
        return calib.left, T
    T_rl = np.eye(4)
    T_rl[:3, :3], T_rl[:3, 3] = calib.R, np.ravel(calib.T)
    return calib.right, T_rl @ T


def view_from_calib(calib, camera="left", table_offset_mm=0.0, ppm=2.0):
    """TopView from a StereoCalib (or its JSON path). table_offset_mm = thickness of the board used for the table pose."""
    if isinstance(calib, (str, Path)):
        from stereo_calib import StereoCalib
        calib = StereoCalib.load(calib)
    cam, T = camera_table_pose(calib, camera)
    return TopView(cam.K, cam.dist, T, z_mm=PIECE_H_MM - table_offset_mm, ppm=ppm, image_size=cam.image_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--calib", help="stereo calibration JSON (from the calibration GUI)")
    ap.add_argument("--camera", default="left", choices=["left", "right"])
    ap.add_argument("--table-offset-mm", type=float, default=0.0, help="board thickness if the table pose was taken on a board")
    ap.add_argument("--radius-px", type=float, help="top-face radius in pixels (uncalibrated mode)")
    ap.add_argument("--out", default="detections")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.calib and not args.radius_px:
        ap.error("give --calib or --radius-px")

    view = view_from_calib(args.calib, args.camera, args.table_offset_mm) if args.calib else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.images):
        img = cv2.imread(str(path))
        pieces = detect_pieces(img, view, args.radius_px)
        counts = {c: sum(p.color == c for p in pieces) for c in COLORS}
        print(f"{path.name}: " + ", ".join(f"{c}={n}" for c, n in counts.items()))
        for p in pieces:
            pos = f"table=({p.table_mm[0]:7.1f}, {p.table_mm[1]:7.1f}) mm" if p.table_mm else ""
            print(f"  {p.color:6s} img=({p.image_uv[0]:6.1f}, {p.image_uv[1]:6.1f}) {pos} rim={p.rim:.2f} fill={p.fill:.2f}")
        vis = draw(img, pieces, view, args.radius_px)
        cv2.imwrite(str(out / path.name), vis)
        if view is not None:
            top = view.warp(img)
            r = PIECE_D_MM / 2 * view.ppm
            for p in pieces:
                cv2.circle(top, tuple(int(round(c)) for c in p.view_uv), int(r), DRAW[p.color], 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"{path.stem}_top{path.suffix}"), top)
        if args.show:
            cv2.imshow("pieces", vis)
            if cv2.waitKey(0) == 27:
                break


if __name__ == "__main__":
    main()
