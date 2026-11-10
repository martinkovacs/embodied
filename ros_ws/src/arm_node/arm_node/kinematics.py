"""OpenMANIPULATOR-X kinematics for a top-down grasp. mm and rad; base frame = URDF link1 (bottom of the base).

All joints at 0: upper arm up, forearm and gripper level, pointing +x. joint2..4 turn about +y, so positive
angles tip the arm forward and down. Lengths from open_manipulator_x.urdf.xacro, except TOOL.
No ROS imports: robot_frame.py uses fk() on the host.

Self-test:  python3 kinematics.py
"""
from math import acos, atan2, ceil, cos, hypot, pi, sin

J1_X, J2_Z = 12.0, 76.5         # joint1 axis x; joint2 height (17 + 59.5)
UP_X, UP_Z = 24.0, 128.0        # joint2 -> joint3 at q2 = 0
FOREARM = 124.0                 # joint3 -> joint4
TOOL = 146.7                    # joint4 -> fingertips (finger mesh). The URDF end_effector_link (126) sits ~21 mm
                                # behind the tips. Measure on the real arm.
LIMITS = [(-pi, pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]
R_MIN = 80.0                    # from the joint1 axis; closer, the hanging fingers hit the base
# Tune in the lab. Heights are above the table surface: pick/place commands carry the robot-frame z of the
# table at (x, y) (robot_frame.py prints where the table is). Here, not in arm_node.py, so runner.py checks
# the layout at the same heights the arm moves to.
GRASP_Z = 5.0                   # fingertips just above the table, fingers around most of the 30 mm piece
PLACE_Z = GRASP_Z + 3.0         # release a little above where it was picked up
LIFT = 60.0                     # approach and retreat this much above GRASP_Z, clear of the 30 mm pieces
CLEAR = 45.0                    # fingertips keep this much above the table while travelling (30 mm pieces + margin)

# Tune in the lab.
HOME = [0.0, -1.0, 0.7, 0.3]    # ROBOTIS "home" (SRDF); check that it hides no piece from either camera
SHOW = [0.0, -0.3, -0.3, 0.6]   # gripper level, held out along +x at ~28 cm; turn joint1 toward the participant

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


def path_min_z(q_from, q_to, n=50):
    """Lowest fingertip z along the straight joint-space line from q_from to q_to, the path a single-point
    trajectory with equal joint scaling and zero end velocities follows."""
    return min(fk(tuple(a + (b - a) * i / n for a, b in zip(q_from, q_to)))[2] for i in range(n + 1))


def check_travel(q_from, q_to, table_z=0.0):
    """Reject a travel path whose fingertips dip below CLEAR above the table surface."""
    low = path_min_z(q_from, q_to) - table_z
    if not (low >= CLEAR):              # fail closed: nan can't pass
        raise ValueError(f"lowest fingertip clearance {low:.0f} mm over the table, needs {CLEAR:.0f} mm; "
                         "travel from a higher pose (lift straight up first)")


def vertical(x, y, z_from, z_to, step=5.0):
    """Waypoints [q] for a strictly vertical descent/ascent over (x, y): ik at evenly spaced heights from z_from
    to z_to (both included, spacing <= step). ValueError naming the height if any is unreachable."""
    n = ceil(abs(z_to - z_from) / step)
    heights = [z_from + (z_to - z_from) * i / n for i in range(n + 1)] if n else [z_from]
    out = []
    for z in heights:
        try:
            out.append(ik(x, y, z))
        except ValueError:
            raise ValueError(f"out of reach: ({x:.0f}, {y:.0f}) mm at height {z:.1f} mm") from None
    return out


LIFT_RECOVERY = ("turn the torque off, move the arm up by hand, then restart the bringup (robot_frame.py's "
                 "docstring)")


def lift_path(q_now, table_z, op_xy=None):
    """Waypoints [q] lifting a low arm (fingertip below table_z + CLEAR) up to table_z + GRASP_Z + LIFT, from
    the interrupted pick/place spot (op_xy, within 10 mm of the fingertip) or its own fingertip column. The
    joint-space move to the first waypoint must not dip more than 2 mm below the current fingertip: ValueError
    naming the pose and the hand-lift procedure when no command can do the lift (a home does the same lift)."""
    x, y, z = fk(q_now)
    msg = (f"arm at ({x:.0f}, {y:.0f}, {z:.0f}) mm over the table cannot be lifted by a command (a home does "
           f"the same lift): {LIFT_RECOVERY}")
    try:
        px, py = (op_xy if op_xy is not None and hypot(x - op_xy[0], y - op_xy[1]) <= 10.0 else (x, y))
        z0 = max(z, table_z + GRASP_Z)
        first = ik(px, py, z0)
        pts = [first, *vertical(px, py, z0, table_z + GRASP_Z + LIFT)]
    except ValueError:
        raise ValueError(msg) from None
    if not (path_min_z(q_now, first) > z - 2.0):        # fail closed: the straight line may dig into the table
        raise ValueError(msg)
    return pts


def heights(q):
    """z of the elbow (joint3), the wrist (joint4) and the fingertips."""
    _, q2, q3, q4 = q
    elbow = J2_Z - UP_X * sin(q2) + UP_Z * cos(q2)
    wrist = elbow - FOREARM * sin(q2 + q3)
    return elbow, wrist, wrist - TOOL * sin(q2 + q3 + q4)


def _seg_gap(a, b, c, d):
    """Distance between the 2D segments a-b and c-d (0 when they cross)."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def to_seg(p, q, r):    # point p to segment q-r
        vx, vy = r[0] - q[0], r[1] - q[1]
        t = max(0.0, min(1.0, ((p[0] - q[0]) * vx + (p[1] - q[1]) * vy) / (vx * vx + vy * vy)))
        return hypot(p[0] - q[0] - t * vx, p[1] - q[1] - t * vy)

    if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
        return 0.0
    return min(to_seg(a, c, d), to_seg(b, c, d), to_seg(c, a, b), to_seg(d, a, b))


def self_gap(q):
    """Smallest centre-line distance between arm parts that can meet, in the arm's own vertical plane: the
    gripper against the upper arm and the base column, and the forearm against the base column."""
    _, q2, q3, q4 = q
    base, j2 = (0.0, 0.0), (0.0, J2_Z)
    j3 = (UP_X * cos(q2) + UP_Z * sin(q2), J2_Z - UP_X * sin(q2) + UP_Z * cos(q2))
    j4 = (j3[0] + FOREARM * cos(q2 + q3), j3[1] - FOREARM * sin(q2 + q3))
    tip = (j4[0] + TOOL * cos(q2 + q3 + q4), j4[1] - TOOL * sin(q2 + q3 + q4))
    return min(_seg_gap(j4, tip, j2, j3), _seg_gap(j4, tip, base, j2), _seg_gap(j3, j4, base, j2))


SELF_CLEAR = 45.0               # centre lines this far apart = parts not touching (links ~35-40 mm thick). Measure.


def wake_path(q_now, q_to=HOME, step=0.1):
    """Waypoints [q] from wherever the arm lies (resting on the table, folded, moved by hand) up to q_to, along
    which the elbow, wrist and fingertips never go more than 2 mm lower than the lowest of them is now (or, for
    an arm already above q_to, than the lowest of them at q_to): whatever the arm rests on, it only moves away
    from it. The arm doesn't pass through itself either: self_gap() stays above SELF_CLEAR, or above what it is
    now for an arm that starts folded tighter. Breadth-first over joint2..4 in `step` rad moves until the
    straight joint-space line to q_to is clear; joint1 only turns on that last line. ValueError if there is none.
    shortcut: knows nothing about pieces or anything else standing around the arm, so it is for starting up
    with a clear table, not for use between trials."""
    floor = min(*heights(q_now), *heights(q_to)) - 2.0
    gap = min(SELF_CLEAR, self_gap(q_now)) - 1.0

    def clear(a, b, n):
        qs = (tuple(m + (w - m) * i / n for m, w in zip(a, b)) for i in range(1, n + 1))
        return all(min(heights(q)) >= floor and self_gap(q) >= gap for q in qs)

    start, q_to = tuple(q_now), tuple(q_to)
    if not all(lo <= v <= hi for v, (lo, hi) in zip(start, LIMITS)):    # also catches nan
        raise ValueError(f"joints {[round(v, 2) for v in start]} are outside their limits")
    came, queue = {(0, 0, 0): None}, [(0, 0, 0)]
    at = lambda k: (start[0], *(start[j + 1] + k[j] * step for j in range(3)))
    for k in queue:                             # appended to while iterating: breadth-first
        if clear(at(k), q_to, 40):
            pts = [q_to]
            while came[k] is not None:
                pts.insert(0, at(k))
                k = came[k]
            return pts
        for j in range(3):
            for d in (-1, 1):
                n = tuple(v + d * (i == j) for i, v in enumerate(k))
                lo, hi = LIMITS[j + 1]
                if n not in came and lo <= at(n)[j + 1] <= hi and clear(at(k), at(n), 2):
                    came[n] = k
                    queue.append(n)
    raise ValueError(f"no way up from here without going lower first: {LIFT_RECOVERY}")


if __name__ == "__main__":
    assert max(abs(a - b) for a, b in zip(fk((0, 0, 0, 0)), (J1_X + UP_X + FOREARM + TOOL, 0, J2_Z + UP_Z))) < 1e-9
    assert CLEAR <= LIFT
    approaches = []
    for r in range(100, 211, 10):               # the pick/place workspace must be reachable...
        for yaw in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for z in (5.0, GRASP_Z + LIFT):     # grasp and approach heights
                p = (J1_X + r * cos(yaw), r * sin(yaw), z)
                q = ik(*p)
                assert max(abs(a - b) for a, b in zip(fk(q), p)) < 1e-9, p   # ...and land exactly there
                assert abs(q[1] + q[2] + q[3] - pi / 2) < 1e-12, p          # gripper straight down
                if z == GRASP_Z + LIFT:
                    approaches.append(q)
    home, show = tuple(HOME), tuple(SHOW)
    for a in approaches:                        # travel passes from home/show...
        check_travel(home, a)
        check_travel(show, a)
    for i, a in enumerate(approaches):          # ...and from every approach pose to any other
        for b in approaches[i + 1:]:
            check_travel(a, b)
    for a in approaches:                        # the approach -> grasp descent is flagged, not travelled
        x, y, _ = fk(a)
        try:
            check_travel(a, ik(x, y, GRASP_Z))
        except ValueError:
            continue
        raise AssertionError("descent to GRASP_Z should be rejected by check_travel")
    for p in ((J1_X + 260, 0, 5), (J1_X + 50, 0, 5), (J1_X + 215, 0, 65), (float("nan"), 0, 5)):
        try:
            ik(*p)
        except ValueError:
            continue
        raise AssertionError(f"{p} should be out of reach")

    def drift(q_from, q_to, x, y, n=20):
        """Widest horizontal excursion of the fingertip from (x, y) along the straight joint-space line."""
        qs = (tuple(m + (w - m) * i / n for m, w in zip(q_from, q_to)) for i in range(n + 1))
        return max(hypot(p[0] - x, p[1] - y) for p in map(fk, qs))

    for a in approaches:                        # vertical(): interpolating between its waypoints keeps the
        x, y, _ = fk(a)                         # fingertip over (x, y), so the descent touches nothing sideways
        for z_to in (GRASP_Z, PLACE_Z):
            pts = vertical(x, y, GRASP_Z + LIFT, z_to)
            assert max(abs(m - w) for m, w in zip(fk(pts[0]), (x, y, GRASP_Z + LIFT))) < 1e-9, pts[0]
            assert max(drift(p, q, x, y) for p, q in zip(pts, pts[1:])) < 0.5, (x, y, z_to)
    try:                                        # an unreachable column names the height
        vertical(J1_X + 350.0, 0.0, GRASP_Z + LIFT, GRASP_Z)
    except ValueError as e:
        assert f"{GRASP_Z + LIFT:.1f}" in str(e), str(e)
    else:
        raise AssertionError("vertical: out-of-reach height should raise")

    a, g = (ik(J1_X + 210.0, 0.0, GRASP_Z + LIFT), ik(J1_X + 210.0, 0.0, GRASP_Z))
    mid = min((tuple(m + (w - m) * i / 100 for m, w in zip(a, g)) for i in range(101)),   # a stop mid-descent:
              key=lambda q: abs(fk(q)[2] - 20.0))                                        # drifted joint line, ~20 mm
    pts = lift_path(mid, 0.0, (J1_X + 210.0, 0.0))
    assert all(hypot(fk(p)[0] - (J1_X + 210.0), fk(p)[1]) < 1e-9 for p in pts), pts      # climbs over the op spot
    assert abs(fk(pts[0])[2] - fk(mid)[2]) < 0.01 and abs(fk(pts[-1])[2] - (GRASP_Z + LIFT)) < 1e-9
    pts = lift_path(ik(J1_X + 150.0, 40.0, GRASP_Z + 10.0), 0.0)                         # a plain low pose lifts
    assert abs(fk(pts[-1])[2] - (GRASP_Z + LIFT)) < 1e-9
    for q in ((0.17, 1.47, 1.37, 0.21), (-0.61, 0.53, 1.26, -1.45)):                     # tilted/hand-moved: no
        try:                                                                            # command can lift, the
            lift_path(q, 0.0)                                                           # straight line digs in
        except ValueError as e:
            assert "hand" in str(e) and "torque" in str(e) and "home" in str(e), str(e)
        else:
            raise AssertionError(f"{q} should be unrecoverable by lift_path")
    assert self_gap(home) > SELF_CLEAR and self_gap(show) > SELF_CLEAR
    assert self_gap((0.0, 1.47, 1.37, 0.21)) == 0.0             # gripper folded back through the base column
    for q in ((0.0, 1.5, -1.2, -0.2), (-0.61, 0.53, 1.26, -1.45), (1.0, 1.0, 1.0, 1.0), (0.0, 1.2, 0.9, 0.3),
              (0.0, -1.5, 1.4, 0.3)):                           # lying stretched out, tilted, folded on the base
        pts, floor = wake_path(q), min(heights(q)) - 2.0        # poses no home could lift: wake_path does, nothing
        gap = min(SELF_CLEAR, self_gap(q)) - 2.0                # dips on the way and the arm stays clear of itself
        assert pts[-1] == home, pts
        for a, b in zip([q, *pts], pts):
            for c in (tuple(m + (w - m) * i / 50 for m, w in zip(a, b)) for i in range(51)):
                assert min(heights(c)) > floor - 1.0 and self_gap(c) > gap, (q, c)
    assert wake_path(home) == [home]
    assert wake_path((0.0, 0.0, 0.0, 0.0)) == [home]            # from above home (fake hardware starts here): down to it
    try:
        wake_path((0.0, 3.0, 0.0, 0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("joints outside their limits should be refused")
    print("kinematics ok")
