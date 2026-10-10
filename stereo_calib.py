"""Stereo-camera calibration backend for a two-camera over-the-table picking rig.

Target: OpenCV checkerboard with PATTERN inner corners (10x7 squares). All
real-world units are millimetres; OpenCV camera conventions throughout. A
NiceGUI app drives this module's public API: find_corners / object_points /
calibrate_camera / calibrate_stereo / board_pose / table_pose_from_pair /
draw_axes plus the CameraCalib and StereoCalib dataclasses.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np

PATTERN = (9, 6)

_SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
_CORNERS_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def find_corners(img: np.ndarray, pattern=PATTERN) -> np.ndarray | None:
    """Locate checkerboard inner corners. Returns (N,2) float32 px, or None."""
    gray = _to_gray(img)
    use_small = gray.shape[1] > 1600
    scale = 1600.0 / gray.shape[1] if use_small else 1.0

    if use_small:  # fast exhaustive search on a downscaled copy first
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, corners = cv2.findChessboardCornersSB(small, pattern, _SB_FLAGS)
        if ok:
            corners = corners / scale
        else:  # then once at full resolution
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, _SB_FLAGS)
    else:
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, _SB_FLAGS)

    if not ok:  # fallback: classic detector + sub-pixel refinement
        probe = (cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                 if use_small else gray)
        ok, corners = cv2.findChessboardCorners(probe, pattern, _CORNERS_FLAGS)
        if not ok:
            return None
        corners = cv2.cornerSubPix(probe, corners.astype(np.float32), (11, 11), (-1, -1), _SUBPIX_CRIT)
        if use_small:
            corners = corners / scale
    elif use_small:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), _SUBPIX_CRIT)

    if corners is None:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def object_points(pattern=PATTERN, square_mm: float = 25.0) -> np.ndarray:
    """(N,3) float32 board-frame corners: origin at first corner, x along rows, z=0."""
    n, m = pattern
    pts = np.zeros((n * m, 3), np.float32)
    pts[:, :2] = np.mgrid[0:n, 0:m].T.reshape(-1, 2) * square_mm
    return pts


@dataclass
class CameraCalib:
    K: np.ndarray                 # 3x3
    dist: np.ndarray              # (5,) or (k,)
    image_size: tuple[int, int]   # (w, h)
    rms: float

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,2) normalized coords (no P)."""
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist)
        return out.reshape(-1, 2)


@dataclass
class StereoCalib:
    left: CameraCalib
    right: CameraCalib
    R: np.ndarray                 # 3x3, X_right = R @ X_left + T
    T: np.ndarray                 # (3,), mm
    rms: float
    T_left_table: np.ndarray | None = None   # 4x4, table frame -> LEFT camera frame
    pattern: tuple[int, int] = PATTERN
    square_mm: float = 25.0

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    def projection_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """P_l = [I|0], P_r = [R|T] in normalized coords (for undistorted points)."""
        pl = np.hstack([np.eye(3), np.zeros((3, 1))])
        pr = np.hstack([self.R, self.T.reshape(3, 1)])
        return pl, pr

    def triangulate(self, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
        """Pixel (N,2) each -> (N,3) mm in left camera frame."""
        pl, pr = self.projection_matrices()
        u = self.left.undistort_points(pts_left)
        v = self.right.undistort_points(pts_right)
        hom = cv2.triangulatePoints(pl, pr, u.T, v.T)
        return (hom[:3] / hom[3]).T

    def to_table(self, pts_left_cam: np.ndarray) -> np.ndarray:
        """(N,3) left-camera frame -> table frame."""
        if self.T_left_table is None:
            raise ValueError("T_left_table is None: run table_pose_from_pair first")
        T_wl = np.linalg.inv(self.T_left_table)
        hom = np.c_[pts_left_cam, np.ones(len(pts_left_cam))] @ T_wl.T
        return hom[:, :3]

    def rectify_maps(self, alpha: float = 0) -> tuple:
        """(map_lx, map_ly, map_rx, map_ry, Q) via stereoRectify + initUndistortRectifyMap."""
        w, h = self.left.image_size
        T_comp = self.T.reshape(3, 1)
        R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
            self.left.K, self.left.dist, self.right.K, self.right.dist,
            (w, h), self.R, T_comp, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        maps_l = cv2.initUndistortRectifyMap(self.left.K, self.left.dist, R1, P1, (w, h), cv2.CV_32FC1)
        maps_r = cv2.initUndistortRectifyMap(self.right.K, self.right.dist, R2, P2, (w, h), cv2.CV_32FC1)
        return maps_l[0], maps_l[1], maps_r[0], maps_r[1], Q

    def save(self, path) -> None:
        """Write human-readable JSON (indent=2, arrays as nested lists)."""
        data = {
            "pattern": list(self.pattern),
            "square_mm": self.square_mm,
            "image_size": list(self.left.image_size),
            "left": {"K": self.left.K.tolist(), "dist": self.left.dist.ravel().tolist(),
                     "rms": self.left.rms},
            "right": {"K": self.right.K.tolist(), "dist": self.right.dist.ravel().tolist(),
                      "rms": self.right.rms},
            "R": self.R.tolist(),
            "T": self.T.ravel().tolist(),
            "rms": self.rms,
            "T_left_table": None if self.T_left_table is None else self.T_left_table.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path) -> "StereoCalib":
        with open(path) as f:
            d = json.load(f)
        size = tuple(d["image_size"])
        left = CameraCalib(np.array(d["left"]["K"]), np.array(d["left"]["dist"]), size, d["left"]["rms"])
        right = CameraCalib(np.array(d["right"]["K"]), np.array(d["right"]["dist"]), size, d["right"]["rms"])
        return cls(left=left, right=right,
                   R=np.array(d["R"]), T=np.array(d["T"]), rms=d["rms"],
                   T_left_table=None if d["T_left_table"] is None else np.array(d["T_left_table"]),
                   pattern=tuple(d["pattern"]), square_mm=d["square_mm"])


def _rms_px(points2d: np.ndarray, observed: np.ndarray) -> float:
    """RMS reprojection error in px over a view: sqrt(mean(||p - x||^2))."""
    err = points2d.reshape(-1, 2) - observed.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def calibrate_camera(corners: list[np.ndarray], image_size, pattern=PATTERN,
                     square_mm=25.0) -> tuple[CameraCalib, list[float]]:
    """Intrinsics from >=3 views; second return = per-view RMS error in px."""
    if len(corners) < 3:
        raise ValueError(f"need at least 3 views of the board, got {len(corners)}")
    obj = object_points(pattern, square_mm)
    obj_views = [obj.reshape(-1, 1, 3).astype(np.float32)] * len(corners)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_views, [c.astype(np.float32) for c in corners], tuple(image_size), None, None)
    per_view = [_rms_px(cv2.projectPoints(obj, rv, tv, K, dist)[0], c)
                for rv, tv, c in zip(rvecs, tvecs, corners)]
    return CameraCalib(K, np.asarray(dist).ravel(), tuple(image_size), float(rms)), per_view


def _pose(corners: np.ndarray, cam: CameraCalib, pattern, square_mm: float) -> np.ndarray:
    """Board->camera via IPPE, refined iteratively; returns 4x4."""
    obj = object_points(pattern, square_mm).reshape(-1, 1, 3)
    _, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float32).reshape(-1, 1, 2),
                                 cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    cv2.solvePnP(obj, corners.astype(np.float32).reshape(-1, 1, 2), cam.K, cam.dist,
                 rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def calibrate_stereo(corners_left: list[np.ndarray], corners_right: list[np.ndarray],
                     left: CameraCalib, right: CameraCalib, pattern=PATTERN,
                     square_mm=25.0) -> tuple[StereoCalib, list[float]]:
    """Extrinsics from same-index synchronized pairs with fixed intrinsics.
    Second return = per-pair RMS error in px."""
    if len(corners_left) != len(corners_right):
        raise ValueError(f"list lengths differ: {len(corners_left)} vs {len(corners_right)}")
    if len(corners_left) < 1:
        raise ValueError("need at least one board pair")
    obj_views = [object_points(pattern, square_mm).reshape(-1, 1, 3).astype(np.float32)] * len(corners_left)
    rms, _K1, _K2, _d1, _d2, R, T, _E, _F = cv2.stereoCalibrate(
        obj_views, [c.astype(np.float32).reshape(-1, 1, 2) for c in corners_left],
        [c.astype(np.float32).reshape(-1, 1, 2) for c in corners_right],
        left.K, left.dist, right.K, right.dist, None, None,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    stereo = StereoCalib(left, right, np.asarray(R), np.asarray(T).ravel(), float(rms),
                         pattern=pattern, square_mm=square_mm)
    # per-pair check: board pose in left cam -> right via R,T -> project into both
    per_pair = []
    obj = object_points(pattern, square_mm)
    for cl, cr in zip(corners_left, corners_right):
        T_cb = _pose(cl, left, pattern, square_mm)
        X_left = (T_cb[:3, :3] @ obj.T).T + T_cb[:3, 3]
        X_right = (stereo.R @ X_left.T).T + stereo.T
        e_l = cv2.projectPoints(X_left.astype(np.float32).reshape(-1, 1, 3), np.zeros(3),
                                np.zeros(3), left.K, left.dist)[0].reshape(-1, 2) - cl.reshape(-1, 2)
        e_r = cv2.projectPoints(X_right.astype(np.float32).reshape(-1, 1, 3), np.zeros(3),
                                np.zeros(3), right.K, right.dist)[0].reshape(-1, 2) - cr.reshape(-1, 2)
        sq = np.r_[e_l, e_r] ** 2
        per_pair.append(float(np.sqrt(np.mean(np.sum(sq, axis=1)))))
    return stereo, per_pair


def board_pose(corners: np.ndarray, cam: CameraCalib, pattern=PATTERN,
               square_mm=25.0) -> np.ndarray:
    """4x4 board->camera pose via solvePnP."""
    return _pose(corners, cam, pattern, square_mm)


def table_pose_from_pair(corners_left, corners_right, stereo: StereoCalib,
                         pattern=PATTERN, square_mm=25.0) -> tuple[np.ndarray, float]:
    """Table frame from a flat board pair: origin = first inner corner, x along
    rows, z up (towards the cameras). Returns (T_left_table 4x4, rms_px in the
    RIGHT camera as a left/right consistency check)."""
    T_cb = _pose(corners_left, stereo.left, pattern, square_mm)
    if float(T_cb[:3, 2] @ (-T_cb[:3, 3])) < 0:  # board z axis points away from camera -> flip
        R, t = T_cb[:3, :3], T_cb[:3, 3]
        T_cb_flipped = np.eye(4)
        T_cb_flipped[:3, :3] = R @ np.diag([1.0, -1.0, -1.0])
        T_cb_flipped[:3, 3] = t
    else:
        T_cb_flipped = T_cb
    # consistency check via the SOLUTION pose (unflipped = actual board pose)
    X_left = (T_cb[:3, :3] @ object_points(pattern, square_mm).T).T + T_cb[:3, 3]
    X_right = (stereo.R @ X_left.T).T + stereo.T
    img_r = cv2.projectPoints(X_right.astype(np.float32).reshape(-1, 1, 3),
                              np.zeros(3), np.zeros(3), stereo.right.K, stereo.right.dist)[0]
    return T_cb_flipped, _rms_px(img_r, corners_right)


def draw_axes(img: np.ndarray, T_cam_frame: np.ndarray, cam: CameraCalib,
              length_mm=100) -> np.ndarray:
    """Copy of img with 3D axes (board/pose frame) drawn."""
    out = img.copy()
    cv2.drawFrameAxes(out, cam.K, cam.dist, cv2.Rodrigues(T_cam_frame[:3, :3])[0],
                      T_cam_frame[:3, 3], length_mm)
    return out
