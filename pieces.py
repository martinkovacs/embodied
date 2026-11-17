"""Detect cylindrical pieces (red / blue / green, 35 mm diameter, 30 mm tall) by colour + known size.

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
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PIECE_D_MM = 35.0
PIECE_H_MM = 30.0

# HSV thresholds, OpenCV convention (H 0..179): hue range [lo, hi), minimum saturation and value. A negative lo
# wraps around 0 (red: -10 = 170). The navy blue pieces' tops lose saturation where the light hits them (S ~100 on
# a bright top, 150-200 elsewhere), hence blue's lower S floor; nothing grey on the table comes near hue 100-135.
COLORS = {
    "red": {"h": (-10, 10), "s": 100, "v": 40},
    "blue": {"h": (100, 135), "s": 60, "v": 20},
    "green": {"h": (40, 90), "s": 50, "v": 40},
}
# Everything the detection GUI can tune. min_fill: share of the top disk that must have the colour; min_rim: share
# of the rim that must have an edge. Lower = more sensitive (more pieces found, and more false ones).
# max_around: share of a ring just outside the top that may have the same colour. A piece top has something else
# around it (its own side covers ~30% of that ring, a group of same-coloured pieces more); a blue-ish wall or cloth is
# blue all around. 0.9 keeps tight groups of same-coloured pieces.
# match_mm: how far apart the two cameras may put the same piece on the table and still pair it (a calibration
# that isn't perfect needs more; pieces are >= 40 mm apart, so not much over 20).
DEFAULT_PARAMS = {"colors": COLORS, "min_fill": 0.55, "min_rim": 0.6, "max_around": 0.9, "match_mm": 20.0}
DRAW = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 180, 0)}


def load_params(path=None):
    """Detection parameters: the defaults, overridden by what the calibration GUI saved to path (if it exists)."""
    params = copy.deepcopy(DEFAULT_PARAMS)
    if path is not None and Path(path).exists():
        saved = json.loads(Path(path).read_text())
        for name, c in saved.get("colors", {}).items():
            if name in params["colors"]:
                params["colors"][name] |= {k: tuple(v) if k == "h" else v for k, v in c.items() if k in ("h", "s", "v")}
        params |= {k: float(saved[k]) for k in ("min_fill", "min_rim", "max_around", "match_mm") if k in saved}
    return params


def save_params(path, params):
    Path(path).write_text(json.dumps(params, indent=2))


@dataclass
class Piece:
    color: str
    view_uv: tuple[float, float]        # centre in the detection image (top view or raw image)
    image_uv: tuple[float, float]       # top-face centre in the original camera image
    table_mm: tuple[float, float, float] | None  # top-face centre in the table frame
    rim: float                          # fraction of the rim with a matching edge
    fill: float                         # fraction of the top disk with the piece colour
    around: float = 0.0                 # fraction of a ring just outside the top with the piece colour


class TopView:
    """Remap a camera image onto the table plane at height z_mm (table frame, z up)."""

    def __init__(self, K, dist, T_cam_table, z_mm=PIECE_H_MM, ppm=2.0, bounds=None, image_size=None):
        self.K, self.dist, self.z, self.ppm = np.asarray(K, float), np.asarray(dist, float), z_mm, ppm
        self.rvec, _ = cv2.Rodrigues(np.asarray(T_cam_table, float)[:3, :3])
        self.tvec = np.asarray(T_cam_table, float)[:3, 3]
        self.T = np.asarray(T_cam_table, float)
        self.cam_c = -self.T[:3, :3].T @ self.tvec  # camera centre in the table frame
        self.image_size = image_size
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

    def image_to_view(self, u, v):
        """Camera image pixel -> top-view pixel: where its ray meets the top-face plane."""
        ray = cv2.undistortPoints(np.array([[[u, v]]], np.float32), self.K, self.dist).reshape(2)
        d = self.T[:3, :3].T @ np.array([ray[0], ray[1], 1.0])
        c = self.cam_c
        x, y = (c + (self.z - c[2]) / d[2] * d)[:2]
        return (x - self.x0) * self.ppm, (self.y1 - y) * self.ppm

    def smear(self, u, v):
        """Where a piece whose top face is centred at view (u, v) smears its side, in view px: the vector from the
        top centre to the bottom centre, seen from the camera along its ray onto the top-face plane. The camera sees
        the side facing it, so in the top view the side lies on the camera's side of the top face, as a band of the
        piece's colour this long; an oblique camera makes it longer than a radius."""
        x, y = self.view_to_table(u, v)
        c = self.cam_c
        k = (c[2] - self.z) / (c[2] - (self.z - PIECE_H_MM))
        bx, by = c[0] + (x - c[0]) * k, c[1] + (y - c[1]) * k
        return (bx - x) * self.ppm, -(by - y) * self.ppm


def color_masks(bgr, colors=COLORS):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    out = {}
    for name, c in colors.items():
        lo, hi = c["h"]
        hm = (H >= lo % 180) | (H < hi) if lo < 0 else (H >= lo) & (H < hi)
        out[name] = hm & (S >= c["s"]) & (V >= c["v"])
    return out


def mask_view(bgr, params, color):
    """The image dimmed to grey except the pixels that pass color's thresholds: shows what the detector sees."""
    m = color_masks(bgr, {color: params["colors"][color]})[color]
    grey = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 3
    grey[m] = bgr[m]
    return grey


def hue_in(h, lo):
    """Hue (0..179) in a colour's hue space: red's runs from -90, so its range doesn't wrap."""
    return np.where(h >= 90, h - 180, h) if lo < 0 else h


def teach(img, u, v, radius_px, params, color=None):
    """Widen color's thresholds (or the colour nearest in hue) so the piece top clicked at (u, v) in img (a camera
    image or a top view; radius_px: the top face's radius there) passes: hue range over its 2nd-98th percentile,
    S and V floors under its 10th percentile, with some margin. Only ever widens. Returns (colour, what changed),
    or raises ValueError when the spot isn't a coloured piece."""
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    disk = (xx - u) ** 2 + (yy - v) ** 2 <= (0.6 * radius_px) ** 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[disk].astype(float)
    if len(hsv) < 5:
        raise ValueError(f"the click at ({u:.0f}, {v:.0f}) is outside the {img.shape[1]} x {img.shape[0]} image")
    H, S, V = hsv.T
    if np.median(S) < 35:
        raise ValueError("no colour there (the table?): click the middle of a piece's top")
    cols = params["colors"]
    if color is None:          # the colour whose hue range is nearest, around the hue circle
        med = {n: np.median(hue_in(H, c["h"][0])) for n, c in cols.items()}
        color = min(cols, key=lambda n: max(cols[n]["h"][0] - med[n], 0, med[n] - cols[n]["h"][1]))
    c = cols[color]
    h = hue_in(H, c["h"][0])
    lim = (-60, 60) if c["h"][0] < 0 else (0, 180)
    lo, hi = int(np.floor(np.percentile(h, 2))) - 3, int(np.ceil(np.percentile(h, 98))) + 3
    new = {"h": (max(min(c["h"][0], lo), lim[0]), min(max(c["h"][1], hi), lim[1])),
           "s": max(0, min(c["s"], int(np.percentile(S, 10)) - 10)),
           "v": max(0, min(c["v"], int(np.percentile(V, 10)) - 10))}
    changed = [f"{k} {c[k]} -> {new[k]}" for k in ("h", "s", "v") if tuple(np.atleast_1d(new[k])) != tuple(np.atleast_1d(c[k]))]
    c.update(new)
    return color, changed


def same_piece(a, b, r_px, smear):
    """Are the candidates a, b ((color, x, y, ...), view px) the same piece? Closer than 1.4 r (touching pieces
    are 2 r apart), or of one colour with b in a's side band (smear: see TopView.smear), which an oblique camera
    stretches past 1.4 r: up to its end plus a little, and no wider than the piece."""
    d = np.subtract(b[1:3], a[1:3])
    if np.hypot(*d) < 1.4 * r_px:
        return True
    if smear is None or a[0] != b[0]:
        return False
    s = np.asarray(smear(*a[1:3]), float)
    n = np.hypot(*s)
    if n < 1e-6:
        return False
    along = d @ s / n
    return abs(along) < n + 0.5 * r_px and np.hypot(*(d - along * s / n)) < 0.5 * r_px


def toward_camera(a, b, smear):
    """How far b lies from a towards the camera, along a's side band (negative: away from it)."""
    s = np.asarray(smear(*a), float)
    n = np.hypot(*s)
    return 0.0 if n < 1e-6 else float(np.subtract(b, a) @ s / n)


def _disk(r):
    r = max(int(round(r)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def rim_votes(gx, gy, mag, edge_thr, r_px, min_rim):
    """Circle centres voted for by the colour edges: every strong edge pixel votes r away along its gradient, both
    ways. The grey Hough above misses rims with little brightness contrast (dark blue on its own side), and the
    colour peaks give nothing inside a group of same-coloured pieces, where the colour fill is one flat plateau."""
    h, w = mag.shape
    ys, xs = np.nonzero(mag > edge_thr)
    if len(xs) == 0:
        return []
    m = mag[ys, xs]
    ux, uy = gx[ys, xs] / m, gy[ys, xs] / m
    acc = np.zeros(h * w, np.float32)
    for sign in (1, -1):
        cx, cy = np.round(xs + sign * r_px * ux).astype(int), np.round(ys + sign * r_px * uy).astype(int)
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        acc += np.bincount(cy[ok] * w + cx[ok], minlength=h * w).astype(np.float32)
    acc = acc.reshape(h, w)
    acc = cv2.GaussianBlur(acc, (0, 0), max(r_px / 10, 1.0))
    # a full rim gives ~2 pi r votes, spread by the blur: ask for a fraction of what a rim passing min_rim would give
    need = 0.5 * min_rim * 2 * np.pi * r_px / (2 * np.pi * max(r_px / 10, 1.0) ** 2)
    peaks = ((acc >= cv2.dilate(acc, _disk(0.5 * r_px))) & (acc >= need)).astype(np.uint8)
    _, _, _, centroids = cv2.connectedComponentsWithStats(peaks)
    return [tuple(c) for c in centroids[1:]]


def detect(bgr, r_px, colors=COLORS, min_rim=0.6, min_fill=0.55, hough_votes=None, smear=None, rejected=None,
           max_around=0.9):
    """Find top-face circles of radius r_px. Returns [(color, u, v, rim, fill)].

    smear(u, v) -> (du, dv): where a piece's side shows in this image (TopView.smear). Without it, a candidate on
    the side of a piece seen at an angle can survive as a second detection of the same piece.
    rejected: a list to receive the near misses, (color, u, v, rim, fill) of circles that failed min_rim or
    min_fill, for tuning: they show which threshold keeps a piece out."""
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
    ring = _disk(1.45 * r_px).astype(np.float32)
    hole = _disk(1.1 * r_px).astype(np.float32)
    o = (ring.shape[0] - hole.shape[0]) // 2
    ring[o:o + hole.shape[0], o:o + hole.shape[1]] -= hole
    ring /= ring.sum()
    arounds = {c: cv2.filter2D(m.astype(np.float32), -1, ring, borderType=cv2.BORDER_CONSTANT) for c, m in masks.items()}
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
    cands += rim_votes(gx, gy, mag, edge_thr, r_px, min_rim)
    # candidates this close end up at the same top face after the grid search: refine one of them
    cell = max(r_px / 3, 1.0)
    cands = list({(round(x / cell), round(y / cell)): (x, y) for x, y in cands}.values())

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
            raw = rim * fill * br
            s = raw * ((rim >= min_rim) & (fill >= min_fill))
            k = int(s.argmax()) if s.max() > 0 else int(raw.argmax())   # nothing passes yet: follow the best try
            x, y = xs[k], ys[k]
        name = names[ci[k]]
        h_, w_ = arounds[name].shape
        around = float(arounds[name][min(max(int(round(y)), 0), h_ - 1), min(max(int(round(x)), 0), w_ - 1)])
        if s[k] > 0 and around <= max_around:
            results.append((name, float(x), float(y), float(rim[k]), float(fill[k]), float(br[k])))
        elif rejected is not None and fill[k] >= 0.4 and rim[k] >= 0.3:   # roughly piece-like only
            rejected.append((name, float(x), float(y), float(rim[k]), float(fill[k]), around, float(raw[k])))

    # Non-max suppression, best score first: one detection per piece. A candidate on the same piece as a kept one
    # (same_piece) is dropped, unless it lies further from the camera along the side band and scores at least half
    # as well: then the kept one was on the side and this one is the top face, which replaces it.
    def sc(r):
        return r[3] * r[4] * r[5]
    results.sort(key=sc, reverse=True)
    kept = []
    for r in results:
        dup = [i for i, k in enumerate(kept) if same_piece(k, r, r_px, smear)]
        if not dup:
            kept.append(r)
        elif len(dup) == 1 and smear is not None:
            k = kept[dup[0]]
            if toward_camera(k[1:3], r[1:3], smear) < -0.3 * r_px and sc(r) >= 0.5 * sc(k):
                kept[dup[0]] = r
    if rejected is not None:     # one near miss per place, and none on a detected piece
        misses, rejected[:] = sorted(rejected, key=lambda r: r[6], reverse=True), []
        for r in misses:
            if all(np.hypot(r[1] - k[1], r[2] - k[2]) > 1.4 * r_px for k in kept + rejected):
                rejected.append(r)
        rejected[:] = [r[:6] for r in rejected[:15]]
    return [r[:5] for r in kept]


def detect_pieces(img, view: TopView | None = None, radius_px=None, params=None, rejected=None, top=None):
    """Detect pieces in a camera image. With a TopView, positions are also returned in table mm.
    params: load_params(); the defaults if None. rejected: a list to receive the near misses as Pieces (see
    detect). top: view.warp(img) if the caller already has it."""
    p = params or DEFAULT_PARAMS
    kw = dict(colors=p["colors"], min_rim=p["min_rim"], min_fill=p["min_fill"],
              max_around=p.get("max_around", DEFAULT_PARAMS["max_around"]))
    misses = None if rejected is None else []
    if view is None:
        found = detect(img, radius_px, rejected=misses, **kw)
        if rejected is not None:
            rejected += [Piece(c, (u, v), (u, v), None, rim, fill, around) for c, u, v, rim, fill, around in misses]
        return [Piece(c, (u, v), (u, v), None, rim, fill) for c, u, v, rim, fill in found]
    top = view.warp(img) if top is None else top

    def piece(c, u, v, rim, fill, around=0.0):
        x, y = view.view_to_table(u, v)
        iu, iv = view.project([[x, y, view.z]])[0]
        return Piece(c, (u, v), (float(iu), float(iv)), (x, y, view.z), rim, fill, around)
    found = detect(top, PIECE_D_MM / 2 * view.ppm, smear=view.smear, rejected=misses, **kw)
    if rejected is not None:
        rejected += [piece(*m) for m in misses]
    return [piece(*f) for f in found]


def draw_misses(vis, misses, view: TopView):
    """Near misses (detect's rejected) on a camera image, thin grey with their rim and fill: which threshold to
    lower to let that piece in."""
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for p in misses:
        x, y, z = p.table_mm
        ring = view.project(np.c_[x + PIECE_D_MM / 2 * np.cos(ang), y + PIECE_D_MM / 2 * np.sin(ang), np.full(32, z)])
        cv2.polylines(vis, [ring.round().astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)
        label = f"{p.color}? rim {p.rim:.2f} fill {p.fill:.2f} around {p.around:.2f}"
        at = (int(ring[:, 0].min()), int(ring[:, 1].max()) + 14)
        cv2.putText(vis, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return vis


def draw_top(top, found, misses, ppm, lone=()):
    """The top view the detector works on, with what it found (colour) and its near misses (grey, with scores)."""
    vis = top.copy()
    r = int(round(PIECE_D_MM / 2 * ppm))
    for p in misses:
        c = tuple(int(round(v)) for v in p.view_uv)
        cv2.circle(vis, c, r, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(vis, f"rim {p.rim:.2f} fill {p.fill:.2f} around {p.around:.2f}", (c[0] - r, c[1] + r + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
    for p in found:
        alone = any(p is q for q in lone)
        c = tuple(int(round(v)) for v in p.view_uv)
        cv2.circle(vis, c, r, (150, 150, 150) if alone else DRAW[p.color], 2, cv2.LINE_AA)
    return vis


def draw(img, pieces, view: TopView | None = None, radius_px=None, lone=(), notes=None):
    """Draw top-face outlines + bounding boxes on the original image. Pieces in lone (seen by this camera only,
    which is where false detections come from) are drawn thin and grey, with a "?"."""
    vis = img.copy()
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    for p in pieces:
        alone = any(p is q for q in lone)
        col = (150, 150, 150) if alone else DRAW[p.color]
        if view is not None:
            r = PIECE_D_MM / 2
            x, y, z = p.table_mm
            top = view.project(np.c_[x + r * np.cos(ang), y + r * np.sin(ang), np.full(48, z)])
            bottom = view.project(np.c_[x + r * np.cos(ang), y + r * np.sin(ang), np.full(48, z - PIECE_H_MM)])
            bx, by, bw, bh = cv2.boundingRect(np.vstack([top, bottom]).astype(np.float32))
            cv2.polylines(vis, [top.round().astype(np.int32)], True, col, 1 if alone else 2, cv2.LINE_AA)
            label = f"{p.color} ({x:.0f}, {y:.0f})" + (f" ? {(notes or {}).get(id(p), '')}" if alone else "")
        else:
            u, v = p.image_uv
            bx, by, bw, bh = int(u - radius_px), int(v - radius_px), int(2 * radius_px), int(2 * radius_px)
            label = p.color
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 1 if alone else 2)
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


def matched_pieces(calib, left_pieces, right_pieces, max_mm=DEFAULT_PARAMS["match_mm"], max_dz=0.8 * PIECE_H_MM):
    """Pair left/right detections (same colour, close on the table) and triangulate their top centres.

    A pair must also triangulate near the top-face height (max_dz; 1 px of disparity is ~2.5 mm of height at 65 cm
    with a 15 cm baseline, so not tighter): a top face does, while a detection on a
    piece's side (each camera sees its own side) lands lower, so two side detections never pass for a piece.

    Returns [(left Piece, table xyz or None)]: the stereo position as an (x, y, z) ndarray in table
    mm, None for a left detection with no right partner.
    """
    rows, used = [], set()
    for p in left_pieces:
        best = None
        for j, q in enumerate(right_pieces):
            d = np.hypot(p.table_mm[0] - q.table_mm[0], p.table_mm[1] - q.table_mm[1])
            if q.color == p.color and j not in used and d < max_mm and (best is None or d < best[1]):
                X = calib.to_table(calib.triangulate(np.array([p.image_uv]), np.array([q.image_uv])))[0]
                if abs(X[2] - p.table_mm[2]) < max_dz:
                    best = (j, d, X)
        stereo = None
        if best is not None:
            used.add(best[0])
            stereo = best[2]
        rows.append((p, stereo))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--calib", help="stereo calibration JSON (from the calibration GUI)")
    ap.add_argument("--camera", default="left", choices=["left", "right"])
    ap.add_argument("--table-offset-mm", type=float, default=0.0, help="board thickness if the table pose was taken on a board")
    ap.add_argument("--radius-px", type=float, help="top-face radius in pixels (uncalibrated mode)")
    ap.add_argument("--params", help="detection thresholds as saved by the calibration GUI (calib_data/detection.json)")
    ap.add_argument("--out", default="detections")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.calib and not args.radius_px:
        ap.error("give --calib or --radius-px")

    view = view_from_calib(args.calib, args.camera, args.table_offset_mm) if args.calib else None
    params = load_params(args.params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.images):
        img = cv2.imread(str(path))
        pieces = detect_pieces(img, view, args.radius_px, params)
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
