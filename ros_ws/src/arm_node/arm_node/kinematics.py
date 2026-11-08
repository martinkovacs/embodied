"""OpenMANIPULATOR-X kinematics for a top-down grasp. mm and rad; base frame = URDF link1 (bottom of the base).

All joints at 0: upper arm up, forearm and gripper level, pointing +x. joint2..4 turn about +y, so positive
angles tip the arm forward and down. Lengths from open_manipulator_x.urdf.xacro, except TOOL.
No ROS imports: robot_frame.py uses fk() on the host.

Self-test:  python3 kinematics.py
"""
from math import acos, atan2, cos, hypot, pi, sin

J1_X, J2_Z = 12.0, 76.5         # joint1 axis x; joint2 height (17 + 59.5)
UP_X, UP_Z = 24.0, 128.0        # joint2 -> joint3 at q2 = 0
FOREARM = 124.0                 # joint3 -> joint4
TOOL = 146.7                    # joint4 -> fingertips (finger mesh). The URDF end_effector_link (126) sits ~21 mm
                                # behind the tips. Measure on the real arm.
LIMITS = [(-pi, pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
R_MIN = 80.0                    # from the joint1 axis; closer, the hanging fingers hit the base
# Tune in the lab. Heights are fingertip z in the robot frame: the table is at z = 0 if the arm stands on it,
# -8 if it stands on the ROBOTIS base plate (robot_frame.py prints where the table is). Here, not in arm_node.py,
# so runner.py checks the layout at the same heights the arm moves to.
GRASP_Z = 5.0                   # fingertips just above the table, fingers around most of the 30 mm piece
PLACE_Z = GRASP_Z + 3.0         # release a little above where it was picked up
LIFT = 60.0                     # approach and retreat this much above GRASP_Z, clear of the 30 mm pieces

UPPER = hypot(UP_X, UP_Z)
LEAN = atan2(UP_X, UP_Z)        # upper arm's forward lean at q2 = 0


def fk(q):
    """Joint angles -> fingertip (x, y, z)."""
    q1, q2, q3, q4 = q
    r = UP_X * cos(q2) + UP_Z * sin(q2) + FOREARM * cos(q2 + q3) + TOOL * cos(q2 + q3 + q4)
    z = J2_Z - UP_X * sin(q2) + UP_Z * cos(q2) - FOREARM * sin(q2 + q3) - TOOL * sin(q2 + q3 + q4)
    return J1_X + r * cos(q1), r * sin(q1), z


def ik(x, y, z):
    """Fingertip (x, y, z), gripper pointing straight down -> joint angles (elbow up). ValueError if unreachable."""
    r = hypot(x - J1_X, y)
    dz = z + TOOL - J2_Z                # the wrist is TOOL straight above the tips; its height over joint2
    d = hypot(r, dz)                    # joint2 -> wrist
    if not (r >= R_MIN and abs(UPPER - FOREARM) < d < UPPER + FOREARM):
        raise ValueError(f"out of reach: ({x:.0f}, {y:.0f}, {z:.0f}) mm, {r:.0f} mm from the base axis")
    shoulder = acos((UPPER ** 2 + d ** 2 - FOREARM ** 2) / (2 * UPPER * d))      # upper arm vs joint2->wrist
    elbow = acos((UPPER ** 2 + FOREARM ** 2 - d ** 2) / (2 * UPPER * FOREARM))   # inner angle at joint3
    up = atan2(dz, r) + shoulder        # upper arm elevation
    q = (atan2(y, x - J1_X), pi / 2 - LEAN - up, pi / 2 + LEAN - elbow, up + elbow - pi / 2)
    for i, (v, (lo, hi)) in enumerate(zip(q, LIMITS)):
        if not lo <= v <= hi:
            raise ValueError(f"out of reach: ({x:.0f}, {y:.0f}, {z:.0f}) mm needs joint{i + 1} = {v:.2f} rad")
    return q


if __name__ == "__main__":
    assert max(abs(a - b) for a, b in zip(fk((0, 0, 0, 0)), (J1_X + UP_X + FOREARM + TOOL, 0, J2_Z + UP_Z))) < 1e-9
    for r in range(100, 211, 10):               # the pick/place workspace must be reachable...
        for yaw in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for z in (5.0, 65.0):               # grasp and approach heights
                p = (J1_X + r * cos(yaw), r * sin(yaw), z)
                q = ik(*p)
                assert max(abs(a - b) for a, b in zip(fk(q), p)) < 1e-9, p   # ...and land exactly there
                assert abs(q[1] + q[2] + q[3] - pi / 2) < 1e-12, p          # gripper straight down
    for p in ((J1_X + 260, 0, 5), (J1_X + 50, 0, 5), (J1_X + 215, 0, 65), (float("nan"), 0, 5)):
        try:
            ik(*p)
        except ValueError:
            continue
        raise AssertionError(f"{p} should be out of reach")
    print("kinematics ok")
