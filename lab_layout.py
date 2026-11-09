"""Writes layouts/lab.yaml: where the pieces start and where they go, on the real table, and shows it on the cameras.

Run it after calib_gui.py and robot_frame.py. Every position is one the arm reaches, at least 40 mm from the others
and seen by both cameras. Slots get the middle of the arm's sweep, the colour zones the sides. The result goes
through the same check as a session start (runner.check_scenario), for every scenario asked for.

--view shows both cameras with the layout drawn on them: put each piece on its circle. A circle turns thick once
the cameras see the right colour there. Press q to quit.

Usage (from the repo root):
    ./lab.sh python lab_layout.py                       # room for every scenario
    ./lab.sh python lab_layout.py --conditions smoke correct     # less room needed
    ./lab.sh python lab_layout.py --bounds -50 400 -200 300      # keep inside these table x and y, mm
    ./lab.sh python lab_layout.py --view [--condition smoke]     # only the slots this scenario uses
    ./lab.sh python lab_layout.py --self-test
"""
import argparse
import sys
import tempfile
from math import atan2, cos, hypot, pi, sin
from pathlib import Path

import cv2
import numpy as np
import yaml

import cameras
import pieces
import robot_frame
import runner
from arm_node.kinematics import J1_X, R_MIN, ik  # noqa: E402  via robot_frame
from stereo_calib import StereoCalib

GRID_MM = 42.0                  # the check wants 40: a little over, so a rounded position still passes
R_RANGE = (max(R_MIN, 100.0), 205.0)    # from the arm's axis; the range kinematics.py's self-test proves works
IMAGE_MARGIN_PX = 30            # a piece this close to the image edge is easily cut off
PIECE_R_MM = pieces.PIECE_D_MM / 2
SHIFTS = 6                      # grid offsets tried along each axis: the best fills the ring
PREFIX = {"r": "red", "w": "white", "g": "green", "b": "blue"}
BGR = {"red": (60, 60, 230), "blue": (230, 120, 40), "green": (80, 200, 80), "white": (245, 245, 245)}
OUT = runner.ROOT / runner.LAB_LAYOUT
ALL_CONDITIONS = sorted(p.stem for p in (runner.ROOT / "scenarios").glob("*.yaml"))


def scenario(condition):
    return yaml.safe_load((runner.ROOT / "scenarios" / f"{condition}.yaml").read_text())


def demand(conditions):
    """(slot ids, first seen first; colour -> spots needed in the worst case of any of the conditions)."""
    slots, need = [], dict.fromkeys(runner.COLORS, 0)
    for name in conditions:
        sc = scenario(name)
        slots += [s for s in sc["order"] if s not in slots]
        script = {int(k): v for k, v in (sc.get("script") or {}).items()}
        got = runner.zone_need(sc["order"], script, {s: PREFIX[s[0]] for s in sc["order"]})
        need = {c: max(need[c], got[c]) for c in runner.COLORS}
    return slots, need


def visible(calib, x, y):
    """Is a piece at table (x, y), base to top face, inside both camera images?"""
    rim = [(x + PIECE_R_MM * cos(a), y + PIECE_R_MM * sin(a), z)
           for a in np.linspace(0, 2 * pi, 8, endpoint=False) for z in (0.0, pieces.PIECE_H_MM)]
    for side in ("left", "right"):
        cam, T = pieces.camera_table_pose(calib, side)
        if (T[:3, :3] @ np.array(rim).T + T[:3, 3:4])[2].min() <= 0:   # behind the camera
            return False
        px = project(calib, side, rim)
        w, h = cam.image_size
        if px[:, 0].min() < IMAGE_MARGIN_PX or px[:, 1].min() < IMAGE_MARGIN_PX \
                or px[:, 0].max() > w - IMAGE_MARGIN_PX or px[:, 1].max() > h - IMAGE_MARGIN_PX:
            return False
    return True


def project(calib, side, points):
    """Table points (N, 3) mm -> image pixels (N, 2)."""
    cam, T = pieces.camera_table_pose(calib, side)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return cv2.projectPoints(np.array(points, float), rvec, T[:3, 3], cam.K, cam.dist)[0].reshape(-1, 2)


def reachable(T, x, y):
    rx, ry, rz = robot_frame.to_robot(T, (x, y, 0.0))
    if not R_RANGE[0] <= hypot(rx - J1_X, ry) <= R_RANGE[1]:
        return False
    try:
        for z in runner.REACH_Z:
            ik(rx, ry, rz + z)
    except ValueError:
        return False
    return True


def candidates(calib, T, bounds):
    """The fullest hex grid of table (x, y) the arm reaches and both cameras see."""
    Tinv = np.linalg.inv(T)
    bx, by = (Tinv[:3, :3] @ np.array([J1_X, 0.0, 0.0]) + Tinv[:3, 3])[:2]
    row, span = GRID_MM * 3 ** 0.5 / 2, int(R_RANGE[1] / GRID_MM) + 3
    best = []
    for sx in np.linspace(0, GRID_MM, SHIFTS, endpoint=False):
        for sy in np.linspace(0, row * 2, SHIFTS, endpoint=False):
            pts = [(bx + sx + GRID_MM * (i + (j % 2) / 2), by + sy + row * j)
                   for j in range(-span, span + 1) for i in range(-span, span + 1)]
            if bounds:
                pts = [(x, y) for x, y in pts if bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]]
            pts = [p for p in pts if reachable(T, *p) and visible(calib, *p)]
            if len(pts) > len(best):
                best = pts
    return best, (float(bx), float(by))


def build(calib, T, conditions, frame_name, bounds=None):
    """The layout dict for the conditions, or SystemExit saying how far off it is."""
    slot_ids, need = demand(conditions)
    total = len(slot_ids) + sum(need.values())
    cands, (bx, by) = candidates(calib, T, bounds)
    if len(cands) < total:
        sys.exit(f"Room for {len(cands)} positions, but {', '.join(conditions)} need {total} "
                 f"({len(slot_ids)} slots, zones {need}).\n"
                 "Raise or angle the cameras so they see more around the arm, or for now ask for less, e.g. "
                 "--conditions smoke correct.")
    mid = atan2(np.mean([y - by for _, y in cands]), np.mean([x - bx for x, _ in cands]))
    off = {p: (atan2(p[1] - by, p[0] - bx) - mid + pi) % (2 * pi) - pi for p in cands}   # angle from the middle
    chosen = sorted(sorted(cands, key=lambda p: abs(off[p]))[:total], key=lambda p: off[p])

    def take(n):
        taken, chosen[:n] = chosen[:n], []
        return [[round(float(x), 1), round(float(y), 1)] for x, y in taken]
    zones = {"red": take(need["red"]), "green": take(need["green"])}
    slots = {s: [PREFIX[s[0]], *xy] for s, xy in zip(slot_ids, take(len(slot_ids)))}
    zones |= {"blue": take(need["blue"]), "white": take(need["white"])}
    return {"robot_frame": frame_name, "slots": slots, "zones": {c: zones[c] for c in runner.COLORS}}


def checked(lay, T, conditions):
    for name in conditions:
        try:
            runner.check_scenario(scenario(name), lay, T)
        except AssertionError as e:
            sys.exit(f"The generated layout doesn't pass for {name}: {e}")


def write(lay, path):
    lines = ["# Written by lab_layout.py for this table, camera and arm setup. All positions: table frame mm.",
             f"robot_frame: {lay['robot_frame']}", "slots:            # id: [colour, x, y]"]
    lines += [f"  {k}: [{c}, {x}, {y}]" for k, (c, x, y) in lay["slots"].items()]
    lines += ["zones:            # colour: spots in the order they fill up, [x, y]"]
    lines += [f"  {c}: {spots}" for c, spots in lay["zones"].items()]
    Path(path).write_text("\n".join(lines) + "\n")


def overlay(img, calib, side, lay, found, slot_ids):
    """One camera image with the slots and zone spots drawn at the height of the pieces' top face."""
    vis, ok = img.copy(), 0

    def circle(x, y, color, thick, label):
        rim = [(x + PIECE_R_MM * cos(a), y + PIECE_R_MM * sin(a), pieces.PIECE_H_MM)
               for a in np.linspace(0, 2 * pi, 32, endpoint=False)]
        px = project(calib, side, rim).astype(np.int32)
        cv2.polylines(vis, [px], True, color, thick, cv2.LINE_AA)
        at = tuple(px.mean(0).astype(int) - (14, -5))
        cv2.putText(vis, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for c, spots in lay["zones"].items():
        for i, (x, y) in enumerate(spots, 1):
            circle(x, y, tuple(v // 2 for v in BGR[c]), 1, c[0].upper() + str(i))
    for s in slot_ids:
        c, x, y = lay["slots"][s]
        hit = runner.nearest([f for f in found if f[0] == c], x, y)
        ok += hit is not None
        circle(x, y, BGR[c], 5 if hit else 2, s)
    return vis, ok


def view(condition, snapshot):
    if not OUT.exists():
        sys.exit(f"no {runner.LAB_LAYOUT}: run lab_layout.py first")
    lay = yaml.safe_load(OUT.read_text())
    slot_ids = scenario(condition)["order"] if condition else list(lay["slots"])
    vision = runner.Vision()
    title, shown = "Layout: q quits", -1
    try:
        while True:
            found, pair = vision.detect()
            (vl, ok), (vr, _) = (overlay(img, vision.calib, s, lay, found, slot_ids)
                                 for img, s in zip(pair, ("left", "right")))
            both = np.hstack([vl, vr])
            if ok != shown:
                print(f"{ok}/{len(slot_ids)} slots have the right piece", flush=True)
                shown = ok
            if snapshot:
                cv2.imwrite(snapshot, both)
                return
            cv2.imshow(title, cv2.resize(both, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                return
    finally:
        for s in vision.sources:
            s.release()
        if not snapshot:
            cv2.destroyAllWindows()


def self_test():
    """On the synthetic calibration: a small layout passes runner's own check; asking for too much is refused."""
    calib = StereoCalib.load(cameras.SYNTHETIC_CALIB)
    T = robot_frame.load(runner.ROOT / "synthetic" / "robot_frame.json")
    conditions = ["smoke", "correct"]
    lay = build(calib, T, conditions, "synthetic/robot_frame.json")
    checked(lay, T, conditions)
    assert set(lay["slots"]) == {"r1", "r2", "r3", "r4", "r5", "w1", "w2", "w3", "g1", "g2", "g3", "b1"}, lay["slots"]
    assert all(lay["slots"][s][0] == PREFIX[s[0]] for s in lay["slots"])
    with tempfile.TemporaryDirectory() as tmp:
        write(lay, Path(tmp) / "x.yaml")
        again = yaml.safe_load((Path(tmp) / "x.yaml").read_text())
    assert again == lay, again
    try:
        build(calib, T, conditions, "x", bounds=(0, 1, 0, 1))
    except SystemExit as e:
        assert "Room for 0 positions" in str(e), e
    else:
        raise AssertionError("an impossible layout passed")
    print(f"lab_layout self-test ok ({len(candidates(calib, T, None)[0])} positions fit the synthetic setup)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS,
                    help="which scenarios need room (default: all)")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="keep every position inside these table coordinates, mm (e.g. to stay on the table)")
    ap.add_argument("--view", action="store_true", help="live camera view with the layout drawn on it")
    ap.add_argument("--condition", choices=ALL_CONDITIONS, help="with --view: only the slots this scenario uses")
    ap.add_argument("--snapshot", help="with --view: save one image to this file and quit (no display needed)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.view:
        return view(args.condition, args.snapshot)
    frame = runner.ROOT / "calib_data" / "robot_frame.json"
    if not cameras.CALIB_PATH.exists() or not frame.exists():
        sys.exit("needs calib_data/calib.json and calib_data/robot_frame.json: do the calib_gui.py and robot_frame.py "
                 "steps first")
    calib, T = StereoCalib.load(cameras.CALIB_PATH), robot_frame.load(frame)
    if calib.T_left_table is None:
        sys.exit("calib.json has no table pose: capture it in calib_gui.py")
    lay = build(calib, T, args.conditions, "calib_data/robot_frame.json", args.bounds)
    checked(lay, T, args.conditions)
    write(lay, OUT)
    print(f"Wrote {runner.LAB_LAYOUT}: {len(lay['slots'])} slots, "
          + ", ".join(f"{len(s)} {c} spots" for c, s in lay["zones"].items()) + f", for {', '.join(args.conditions)}.")


if __name__ == "__main__":
    main()
