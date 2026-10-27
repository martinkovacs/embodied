"""Table frame -> robot base frame (T_robot_table), fitted from table points touched with the fingertips.

Table frame: the one pieces.py reports (chessboard of the table-pose capture: origin at its first inner corner,
x along the rows, z up), mm. Robot frame: the OMX base (URDF link1), mm, the frame /arm/cmd x, y are in.

Close the gripper and touch the fingertips to 4-6 known table points spread over the work area, e.g. inner
chessboard corners (multiples of square_mm) with the board still where it was for the table-pose capture. At each
point type its table "x y"; the script reads /joint_states over rosbridge and runs arm_node's forward kinematics.
The Kabsch fit goes to calib_data/robot_frame.json. Then --check on new points: accept if every error is < 5 mm.
Refit whenever the table pose is captured again.

Moving the arm by hand: torque off with
    ros2 service call /dynamixel_hardware_interface/set_dxl_torque std_srvs/srv/SetBool "{data: false}"
and afterwards restart the bringup instead of turning torque back on: the arm controller still holds its last
command and would snap back to it.

Usage:
    venv/bin/python robot_frame.py [--host localhost]     # fit
    venv/bin/python robot_frame.py --check
    venv/bin/python robot_frame.py --self-test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import roslibpy

sys.path.insert(0, str(Path(__file__).resolve().parent / "ros_ws" / "src" / "arm_node"))
from arm_node.kinematics import fk  # noqa: E402  the model arm_node moves with, so the two frames agree

PATH = Path("calib_data") / "robot_frame.json"
JOINTS = ("joint1", "joint2", "joint3", "joint4")
ACCEPT_MM = 5.0


def kabsch(src, dst):
    """Rigid 4x4 T minimising |T src - dst| over paired (N,3) points."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cd - R @ cs
    return T


def to_robot(T, p):
    """Table point(s) (..., 3) mm -> robot base frame mm."""
    return np.asarray(p, float) @ T[:3, :3].T + T[:3, 3]


def load(path=PATH):
    return np.array(json.loads(Path(path).read_text())["T_robot_table"])


def collect(host):
    """[(table xyz, fingertip xyz)] for each point touched, typed in interactively."""
    ros = roslibpy.Ros(host=host, port=9090)
    ros.run()
    q = {}
    roslibpy.Topic(ros, "/joint_states", "sensor_msgs/JointState").subscribe(
        lambda m: q.update(zip(m["name"], m["position"])))
    pairs = []
    while line := input('fingertips on a table point -> its table "x y" in mm (empty line = done): ').strip():
        try:
            x, y = map(float, line.split())
        except ValueError:
            print("  type two numbers, e.g. 100 -50")
            continue
        if not all(j in q for j in JOINTS):
            print("  no /joint_states from the arm yet")
            continue
        pairs.append(((x, y, 0.0), fk([q[j] for j in JOINTS])))
        print("  fingertips at robot %.1f %.1f %.1f mm" % pairs[-1][1])
    ros.terminate()
    return pairs


def self_test():
    rng = np.random.default_rng(0)
    a, t = 2.0, 0.02  # yaw, tilt
    T_true = np.eye(4)
    T_true[:3, :3] = (np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                      @ np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]))
    T_true[:3, 3] = (210, -90, -8)
    table = np.c_[rng.uniform(0, 200, (6, 2)), np.zeros(6)]
    robot = to_robot(T_true, table) + rng.normal(0, 0.5, (6, 3))   # touch + joint read-out noise
    T = kabsch(table, robot)
    new = np.c_[rng.uniform(0, 200, (50, 2)), np.zeros(50)]
    assert np.linalg.norm(to_robot(T, new) - to_robot(T_true, new), axis=1).max() < 2.0
    assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-9
    # coplanar points fit their mirror image too, by turning the table upside down: that's the tell
    assert T[2, 2] > 0 and kabsch(table * [1, -1, 1], robot)[2, 2] < 0
    print("robot_frame self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="measure new points against the saved fit")
    ap.add_argument("--host", default="localhost", help="rosbridge host (port 9090)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    pairs = collect(args.host)
    if len(pairs) < (1 if args.check else 4):
        sys.exit("not enough points (the fit needs at least 4)")
    table, robot = (np.array(v) for v in zip(*pairs))
    if args.check:
        T = load()
    else:
        T = kabsch(table, robot)
        if T[2, 2] < 0:
            sys.exit("the fit turns the table upside down: the typed points are mirrored (wrong sign of y?)")
        PATH.write_text(json.dumps({"T_robot_table": T.tolist(), "table": table.tolist(), "robot": robot.tolist()},
                                   indent=2))
        print(f"saved {PATH}; table plane at robot z = {T[2, 3]:.1f} mm. Fit residuals (then --check on new points):")
    err = np.linalg.norm(to_robot(T, table) - robot, axis=1)
    for (x, y, _), e in zip(table, err):
        print(f"  table {x:7.1f} {y:7.1f}   error {e:4.1f} mm")
    print(f"max {err.max():.1f} mm -> " + ("OK" if err.max() < ACCEPT_MM else f"NOT OK, needs < {ACCEPT_MM:.0f} mm"))


if __name__ == "__main__":
    main()
