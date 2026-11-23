"""Calibration, detection and robot control (NiceGUI): calib_app.py plus a Robot tab and the arm in the 3D view.

The Setup, Calibration, Detection and 3D tabs are calib_app.py's own (imported, not copied), so both apps read and
write the same calib_data/ files and behave the same. The robot parts:

  * Robot tab, Arm: wake / home / show / open gripper, STOP, and the arm's state (rosbridge, joints, fingertips).
    "Air" (on at every start) makes every move the slow dry run: picks and places stop AIR_MM above the table and
    nothing is gripped. Switch it off on purpose for real picks.
  * Robot tab, Move a piece: click a piece in the picture to hover over it or pick it up, click a free spot to put
    it there. The positions go through calib_data/robot_frame.json, exactly as runner.py sends them.
  * Robot tab, Table -> robot fit: robot_frame.py's touch-point step in the browser. Torque off, touch the
    fingertips to a board corner, click that corner in the picture (or type it), Add point. From 4 points the fit
    and its errors show; Save writes robot_frame.json. Touching new corners against a saved fit is the --check.
  * 3D tab: the arm's live pose from /joint_states, placed with the fit.

The arm is driven only through arm_node's existing /arm/cmd commands (arm_node.py's docstring) over rosbridge,
like runner.py does; torque off is the same service call as ./lab.sh torque-off. Nothing changes on the ROS side.

Run:  ./lab.sh arm                                   (first terminal: the ROS side)
      ./lab.sh python robot_app.py                   (then open http://localhost:8082)
      ./lab.sh python robot_app.py --fake-arm --files 'synthetic/table_left.png,synthetic/scene_left.png' \\
                                                      'synthetic/table_right.png,synthetic/scene_right.png'
                                                     (no ROS, no cameras: a simulated arm and image files)
calib_app.py and this app can't use the cameras at the same time: close one before starting the other.
"""
import argparse
import asyncio
import json
import os
import threading
import time
from concurrent.futures import Future, InvalidStateError
from math import cos, sin

import cv2
import numpy as np
from fastapi.responses import Response
from nicegui import app, run, ui

import calib_app as ca
import pieces
import robot_frame      # first: puts ros_ws/src/arm_node on sys.path
from arm_node.kinematics import (FOREARM, GRASP_Z, HOME, J1_X, J2_Z, LIFT, PLACE_Z, SHOW, TOOL, UP_X, UP_Z,  # noqa: E402
                                 fk, ik, jog, travel_moves, vertical, wake_moves)
from calib_app import SIDES, Engine
from cameras import ROOT

PORT = 8082             # calib_app.py is 8080, the runner's panel 8081
# as in arm_node.py (which imports rclpy, so it can't be imported here)
OPEN, CLOSED = 0.019, -0.01
MAX_SPEED, AIR_SPEED, AIR_MM = 0.6, 0.25, 40.0
TORQUE_SERVICE = "/dynamixel_hardware_interface/set_dxl_torque"
TIMEOUT_S = {"wake": 150.0}             # a joint-by-joint wake is slow; every other command
CMD_TIMEOUT_S = 60.0                    # (runner.py's ARM_TIMEOUT_S)
STOP_CONFIRM_S = 8.0                    # arm_node's worst case to confirm a stop (runner.py's STOP_CONFIRM_S)
JOINT_STALE_S = 1.0
SNAP_MM = 25.0                          # a click this close to a detected piece means that piece
SPACING_MM = 40.0                       # runner.py's: room for the open fingers between pieces
MOVES = ("home", "show", "pick", "place", "wake")   # what "air" slows down (hover and jog are always slow)
JOG_STEP = 0.1                          # rad per key press (arm_node allows up to kinematics.JOG_MAX)
JOG_KEYS = {"1": (1, 1), "q": (1, -1), "2": (2, 1), "w": (2, -1), "3": (3, 1), "e": (3, -1), "4": (4, 1), "r": (4, -1)}
LOG = ROOT / "logs" / "robot_app.log"


def log(msg):
    """To the terminal and logs/robot_app.log: every click, command, answer, and why a move was refused."""
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d')} {line}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- the arm's two stand-ins

class RosLink:
    """arm_node over rosbridge: /arm/cmd out, /arm/status and /joint_states in. Keeps reconnecting by itself (start
    ./lab.sh arm before or after the app); every (re)connection bumps `epoch`, so a restarted ROS side is noticed."""

    real = True

    def __init__(self, host):
        self.host = host
        self.label = f"rosbridge {host}:9090"
        self.lock = threading.Lock()
        self.ros = None
        self.q, self.grip, self.q_at = None, None, 0.0
        self.waiting: dict[str, Future] = {}
        self.n, self.nonce = 0, os.urandom(3).hex()     # an answer meant for another program never matches ours
        self.epoch = 0

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def connected(self):
        return bool(self.ros and self.ros.is_connected)

    def _run(self):
        import roslibpy
        self.rl = roslibpy
        ros = roslibpy.Ros(host=self.host, port=9090)
        ros.factory.set_max_delay(3.0)  # retry every few seconds, not up to an hour
        # subscriptions wait for the connection, and roslibpy renews them after a reconnect
        self.cmd = roslibpy.Topic(ros, "/arm/cmd", "std_msgs/String")
        self.cmd.advertise()            # now: the first message on a fresh topic can go missing
        roslibpy.Topic(ros, "/arm/status", "std_msgs/String").subscribe(self._on_status)
        roslibpy.Topic(ros, "/joint_states", "sensor_msgs/JointState", throttle_rate=50).subscribe(self._on_joints)
        self.ros = ros
        try:
            ros.run(timeout=2)
        except Exception:
            pass                        # not up yet: the factory keeps trying
        was = False
        while True:
            now = ros.is_connected
            if now and not was:
                self.epoch += 1
            if was and not now:
                self._fail_all("the connection to rosbridge dropped")
            was = now
            time.sleep(0.2)

    def _on_joints(self, m):
        pos = dict(zip(m["name"], m["position"]))
        if not any(j in pos for j in robot_frame.JOINTS):
            return
        with self.lock:
            if all(j in pos and np.isfinite(pos[j]) for j in robot_frame.JOINTS):
                self.q, self.q_at = [float(pos[j]) for j in robot_frame.JOINTS], time.monotonic()
            else:
                self.q = None           # fail closed, as arm_node does
            if "gripper_left_joint" in pos:
                self.grip = float(pos["gripper_left_joint"])

    def _on_status(self, m):            # roslibpy's thread
        try:
            s = json.loads(m["data"])
        except (ValueError, TypeError):
            return
        if s.get("state") == "started":
            return
        with self.lock:
            fut = self.waiting.pop(s.get("id"), None)
        if fut is not None:
            try:
                fut.set_result(s)
            except InvalidStateError:   # timed out meanwhile: nobody waits for it any more
                pass

    def _fail_all(self, reason):
        with self.lock:
            futs, self.waiting = list(self.waiting.values()), {}
        for f in futs:
            try:
                f.set_result({"state": "failed", "reason": reason})
            except InvalidStateError:
                pass

    def state(self):
        """(joints or None, gripper position or None, age of the joints in s)."""
        with self.lock:
            return self.q, self.grip, time.monotonic() - self.q_at

    def send(self, cmd):
        """Publish one command; a Future of its final /arm/status answer ({"state": "done" | "failed", ...})."""
        fut = Future()
        if not self.connected:          # roslibpy would queue it and send it on reconnect: a stale command
            fut.set_result({"state": "failed", "reason": "no connection to rosbridge: is ./lab.sh arm running?"})
            return fut
        with self.lock:
            self.n += 1
            cid = f"{self.nonce}-{self.n}"
            self.waiting[cid] = fut
        self.cmd.publish(self.rl.Message({"data": json.dumps({"id": cid} | cmd)}))
        return fut

    def torque_off(self):
        """./lab.sh torque-off. Blocking; raises if the driver refuses or doesn't answer."""
        if not self.connected:
            raise RuntimeError("no connection to rosbridge")
        svc = self.rl.Service(self.ros, TORQUE_SERVICE, "std_srvs/SetBool")
        res = svc.call(self.rl.ServiceRequest({"data": False}), timeout=5)
        if not res.get("success", False):
            raise RuntimeError(res.get("message") or "the arm's driver refused")
        return res.get("message", "")


class SimLink:
    """Stands in for rosbridge, arm_node and the arm (--fake-arm): the same commands and answers, moved along the
    same kinematics paths arm_node plans (kinematics.py), with the joints animated. For trying the app without ROS;
    it knows nothing about pieces, so a pick moves nothing in the pictures."""

    real = False
    label = "simulated arm (--fake-arm)"
    connected = True
    epoch = 1

    def __init__(self):
        self.lock = threading.Lock()
        self.q, self.grip = list(HOME), OPEN
        self.busy, self.stopping = False, threading.Event()
        self.table_z, self.op_xy = 0.0, None

    def start(self):
        pass

    def state(self):
        with self.lock:
            return list(self.q), self.grip, 0.0

    def send(self, cmd):
        fut = Future()
        if cmd["op"] == "stop":
            self.stopping.set()
            threading.Thread(target=self._confirm_stop, args=(fut,), daemon=True).start()
            return fut
        with self.lock:
            if self.busy:
                fut.set_result({"state": "failed", "reason": "busy"})
                return fut
            self.busy = True
            self.stopping.clear()
        threading.Thread(target=self._run, args=(cmd, fut), daemon=True).start()
        return fut

    def torque_off(self):
        return "simulated: nothing to let go"

    def _confirm_stop(self, fut):
        end = time.monotonic() + 2
        while self.busy and time.monotonic() < end:
            time.sleep(0.02)
        fut.set_result({"state": "done"} if not self.busy else {"state": "failed", "reason": "stop: still moving"})

    def _run(self, cmd, fut):
        try:
            self._do(cmd)
            res = {"state": "done"}
        except Exception as e:
            res = {"state": "failed", "reason": str(e)}
        with self.lock:
            self.busy = False
        fut.set_result(res)

    def _do(self, cmd):
        op, air = cmd["op"], bool(cmd.get("air"))
        speed = AIR_SPEED if air or op in ("hover", "wake", "jog") else MAX_SPEED
        q = list(self.q)
        if op == "open":
            return self._grip(OPEN)
        if op == "close":
            return self._grip(CLOSED)
        if op == "jog":
            return self._moves([[jog(q, int(cmd["joint"]), float(cmd["delta"]), self.table_z)]], speed)
        if op in ("home", "show"):
            return self._moves(travel_moves(q, HOME if op == "home" else SHOW, self.table_z, self.op_xy), speed)
        if op == "wake":
            return self._moves(wake_moves(q, self.table_z, self.op_xy), speed)
        x, y, z = float(cmd["x"]), float(cmd["y"]), float(cmd["z"])
        if op == "hover":
            target = vertical(x, y, z + GRASP_Z + LIFT, z + GRASP_Z + LIFT)[0]
            visit, self.table_z = max(self.table_z, z), z
            return self._moves(travel_moves(q, target, visit, self.op_xy), speed)
        if op not in ("pick", "place"):
            raise ValueError(f"unknown op {op!r}")
        up = z + GRASP_Z + LIFT
        down = z + (GRASP_Z if op == "pick" else PLACE_Z) + (AIR_MM if air else 0)
        descent, ascent = vertical(x, y, up, down), vertical(x, y, down, up)
        visit = max(self.table_z, z)
        self.op_xy, self.table_z = (x, y), z
        self._moves(travel_moves(q, descent[0], visit, self.op_xy), speed)
        if op == "pick":
            self._grip(OPEN)
        self._moves([descent], speed)
        self._grip(CLOSED if op == "pick" else OPEN)
        self._moves([ascent], speed)
        self.op_xy = None

    def _moves(self, moves, speed):
        for points in moves:
            for target in points:
                start = list(self.q)
                dur = max(max(abs(a - b) for a, b in zip(start, target)) / speed, 0.05)
                t0 = time.monotonic()
                while True:
                    if self.stopping.is_set():
                        raise RuntimeError("stopped")
                    f = min(1.0, (time.monotonic() - t0) / dur)
                    with self.lock:
                        self.q = [a + (b - a) * f for a, b in zip(start, target)]
                    if f >= 1:
                        break
                    time.sleep(0.03)

    def _grip(self, pos):
        g0, t0 = self.grip, time.monotonic()
        while (f := min(1.0, (time.monotonic() - t0) / 0.6)) < 1:
            if self.stopping.is_set():
                raise RuntimeError("stopped")
            with self.lock:
                self.grip = g0 + (pos - g0) * f
            time.sleep(0.03)
        with self.lock:
            self.grip = pos


# ---------------------------------------------------------------- robot state and commands

def arm_points(q):
    """The arm's centre line in the robot frame, mm: base, joint2, joint3, joint4, fingertips (as kinematics.fk)."""
    q1, q2, q3, q4 = q
    r2, z2 = UP_X * cos(q2) + UP_Z * sin(q2), J2_Z - UP_X * sin(q2) + UP_Z * cos(q2)
    a = q2 + q3
    r3, z3 = r2 + FOREARM * cos(a), z2 - FOREARM * sin(a)
    b = a + q4
    r4, z4 = r3 + TOOL * cos(b), z3 - TOOL * sin(b)
    return np.array([(J1_X + r * cos(q1), r * sin(q1), z) for r, z in ((0, 0), (0, J2_Z), (r2, z2), (r3, z3), (r4, z4))])


class Robot:
    """What the Robot tab shows and does: the table -> robot fit, the commands (one at a time), the target, the
    fit points. Lives on the UI's event loop; the link does the talking."""

    def __init__(self, E: Engine, link):
        self.E, self.link = E, link
        self.path = E.data / "robot_frame.json"     # --files: development data, never the lab's fit
        self.T = None
        self.load_frame()
        self.air = True
        self.busy = None                # the running command, described
        self.last = "no command sent yet"
        self.torque_off = False         # set by Torque off, cleared when the ROS side restarts
        self.epoch = link.epoch
        self.target = None              # {"kind": "piece" | "spot", "x", "y", "color"}
        self.holding = None
        self.points = []                # fit: [(table xyz, fingertip robot xyz)]

    def load_frame(self):
        path = self.path
        if not path.exists() and self.E.files:
            path = ROOT / "synthetic" / "robot_frame.json"  # the synthetic scene's fit
        self.T = robot_frame.load(path) if path.exists() else None

    def poll(self):
        if self.link.epoch != self.epoch:   # the ROS side (re)started: torque is on again
            self.epoch, self.torque_off = self.link.epoch, False

    # -- positions
    def joints(self):
        q, grip, age = self.link.state()
        return (q, grip) if q is not None and age < JOINT_STALE_S else (None, grip)

    def tip(self):
        """Fingertips in the robot frame (mm), or None without fresh joints."""
        q, _ = self.joints()
        return None if q is None else np.array(fk(q))

    def to_robot(self, x, y):
        return robot_frame.to_robot(self.T, (x, y, 0.0))    # z: the table surface there, as runner.py

    def to_table(self, p):
        Ti = np.linalg.inv(self.T)
        return np.asarray(p, float) @ Ti[:3, :3].T + Ti[:3, 3]

    def reach(self, x, y):
        """None if the arm can grasp at table (x, y) and stand above it, else why not."""
        if self.T is None:
            return "no table → robot fit yet (Table → robot fit below)"
        rx, ry, rz = self.to_robot(x, y)
        try:
            ik(rx, ry, rz + GRASP_Z)
            ik(rx, ry, rz + GRASP_Z + LIFT)
        except ValueError as e:
            return str(e)
        return None

    # -- commands
    @property
    def can_move(self):
        return self.link.connected and not self.busy and not self.torque_off

    async def do(self, op, xy=None, **fields):
        """Send one command (fields: extra ones, e.g. a jog's joint and delta) and wait for its answer; RuntimeError
        with the reason if it failed."""
        if self.busy:
            raise RuntimeError(f"still running: {self.busy}")
        if self.torque_off:
            raise RuntimeError("torque is off: restart ./lab.sh arm first")
        cmd, what = {"op": op, **fields}, " ".join([op, *(f"{k} {v}" for k, v in fields.items())])
        if xy is not None:
            if self.T is None:
                raise RuntimeError("no table → robot fit yet")
            rx, ry, rz = self.to_robot(*xy)
            cmd |= {"x": round(float(rx), 1), "y": round(float(ry), 1), "z": round(float(rz), 1)}
            what = f"{op} at table ({xy[0]:.0f}, {xy[1]:.0f})"
        if self.air and op in MOVES:
            cmd["air"] = True
            what += " (air)"
        self.busy, self.last = what, f"{what}: running"
        log(f"send {what}: {json.dumps(cmd)}")
        timeout = TIMEOUT_S.get(op, CMD_TIMEOUT_S)
        try:
            s = await asyncio.wait_for(asyncio.wrap_future(self.link.send(cmd)), timeout)
        except TimeoutError:
            ok, why = await self.stop()     # a hung command: the arm may still be moving
            self.last = (f"{what}: no answer in {timeout:.0f} s. Stop " +
                         ("confirmed" if ok else f"NOT confirmed ({why}): use the power switch"))
            raise RuntimeError(self.last) from None
        finally:
            self.busy = None
        if s.get("state") != "done":
            self.last = f"{what}: failed: {s.get('reason', s.get('state'))}"
            raise RuntimeError(s.get("reason", s.get("state")))
        self.last = f"{what}: done"

    async def stop(self):
        """(confirmed, reason). Not confirmed: assume the arm may still be moving."""
        try:
            s = await asyncio.wait_for(asyncio.wrap_future(self.link.send({"op": "stop"})), STOP_CONFIRM_S)
        except TimeoutError:
            return False, f"no answer from arm_node in {STOP_CONFIRM_S:.0f} s"
        return (True, None) if s.get("state") == "done" else (False, s.get("reason", s.get("state")))

    # -- table -> robot fit
    def add_point(self, x, y):
        tip = self.tip()
        if tip is None:
            raise ValueError("no fresh /joint_states from the arm")
        self.points.append(((float(x), float(y), 0.0), tuple(float(v) for v in tip)))

    def fit(self):
        """(errors under the saved fit or None, new fit or None, errors under it or None, problem or None)."""
        if not self.points:
            return None, None, None, None
        table, robot = (np.array(v) for v in zip(*self.points))
        saved = (np.linalg.norm(robot_frame.to_robot(self.T, table) - robot, axis=1)
                 if self.T is not None else None)
        if len(self.points) < 4:
            return saved, None, None, f"{4 - len(self.points)} more point(s) for a new fit"
        sv = np.linalg.svd(table[:, :2] - table[:, :2].mean(0), compute_uv=False) / np.sqrt(len(table))
        if sv[1] < 15 or sv[1] < 0.15 * sv[0]:
            return saved, None, None, "the points lie almost on a line: add some spread over the work area"
        T = robot_frame.kabsch(table, robot)
        err = np.linalg.norm(robot_frame.to_robot(T, table) - robot, axis=1)
        if T[2, 2] < 0:
            return saved, None, None, "the fit turns the table upside down: the typed points are mirrored (wrong sign of y?)"
        return saved, T, err, None

    def save_fit(self):
        _, T, _, problem = self.fit()
        if T is None:
            raise ValueError(problem or "no fit")
        table, robot = (np.array(v) for v in zip(*self.points))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"T_robot_table": T.tolist(), "table": table.tolist(), "robot": robot.tolist()},
                                        indent=2))
        self.T = T
        return T


# ---------------------------------------------------------------- picture geometry

def ray_to_plane(view, u, v, z):
    """Camera image pixel -> table (x, y) where its ray meets the plane at height z (table frame, mm)."""
    ray = cv2.undistortPoints(np.array([[[u, v]]], np.float32), view.K, view.dist).reshape(2)
    d = view.T[:3, :3].T @ np.array([ray[0], ray[1], 1.0])
    c = view.cam_c
    return tuple((c + (z - c[2]) / d[2] * d)[:2])


def click_to_table(view, kind, u, v):
    """A click on a picture -> (x, y) on the pieces' top plane, (x, y) on the table plane (z = 0)."""
    if kind == "top":
        top = view.view_to_table(u, v)
        u, v = view.project([[top[0], top[1], view.z]])[0]
    else:
        top = view.view_to_table(*view.image_to_view(u, v))
    return top, ray_to_plane(view, u, v, 0.0)


def to_px(view, kind, xyz):
    """Table point (mm) -> pixel in the picture (camera image or top view)."""
    u, v = view.project([xyz])[0]
    return view.image_to_view(u, v) if kind == "top" else (u, v)


# ---------------------------------------------------------------- UI

def build_ui(E: Engine, R: Robot):
    @app.get("/v2img/{name}.jpg")
    def image(name: str, mask: str = "", misses: int = 0, lone: int = 0):    # as calib_app.build_ui's
        try:
            img = ca.picture(E, name, mask or None, bool(misses), bool(lone))
        except Exception as e:
            img = ca.placeholder(f"error: {e}"[:60])
        if img is None:
            img = ca.placeholder("no picture yet")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @ui.page("/")
    def page():
        ui.page_title("Calibration + robot")
        ui.query("body").style("background: #121212")

        with ui.header().classes("items-center bg-slate-800 gap-3"):
            ui.label("Calibration + robot").classes("text-lg font-medium")
            if E.files:
                ui.button(icon="chevron_left", on_click=lambda: E.step(-1)).props("flat dense color=white")
                ui.button(icon="chevron_right", on_click=lambda: E.step(1)).props("flat dense color=white")
            src = ui.label().classes("text-sm opacity-80")
            ui.space()
            chip_cam = ui.chip(icon="videocam").props("dense")
            chip_cal = ui.chip(icon="straighten").props("dense")
            chip_det = ui.chip(icon="category").props("dense")
            chip_arm = ui.chip(icon="precision_manufacturing").props("dense")

        with ui.tabs().classes("w-full") as tabs:
            t_setup = ui.tab("Setup", icon="settings")
            t_cal = ui.tab("Calibration", icon="grid_on")
            t_det = ui.tab("Detection", icon="center_focus_weak")
            t_3d = ui.tab("3D", icon="view_in_ar")
            t_rob = ui.tab("Robot", icon="precision_manufacturing")

        with ui.tab_panels(tabs, value=t_setup if E.cams is None else t_cal).classes("w-full bg-transparent"):
            with ui.tab_panel(t_setup):
                ca.setup_panel(E, tabs, t_cal)
            with ui.tab_panel(t_cal).classes("p-2"):
                cal = ca.calibration_panel(E)
            with ui.tab_panel(t_det).classes("p-2"):
                det = ca.detection_panel(E)
            with ui.tab_panel(t_3d).classes("p-2"):
                with ui.column().classes("w-full gap-0") as box:
                    d3 = ca.scene_panel(E)
                arm3d = Arm3D(next(e for e in box.descendants() if isinstance(e, ui.scene)), R)
            with ui.tab_panel(t_rob).classes("p-2"):
                rob = robot_panel(E, R)

        def tick():
            tab = tabs.value if isinstance(tabs.value, str) else tabs.value.props["name"]
            R.poll()
            src.text = E.source_label()
            if E.cam_error:
                chip_cam.text, color = E.cam_error, "negative"
            elif E.cams:
                chip_cam.text, color = "cameras on", "positive"
            else:
                chip_cam.text, color = "no cameras", "grey"
            chip_cam.props(f"color={color} text-color=white")
            ok = E.calib is not None and E.calib.T_left_table is not None
            chip_cal.text = "loading captures…" if E.loading else "calibrated" if ok else "not calibrated"
            chip_cal.props(f"color={'positive' if ok else 'grey'} text-color=white")
            d = E.det
            if E.det_error:
                chip_det.text, color = E.det_error[:50], "negative"
            elif d:
                n = sum(st is not None for _, st in d["matched"])
                chip_det.text, color = f"{n} piece{'s' if n != 1 else ''} · {d['seconds']:.1f} s", "positive"
            else:
                chip_det.text, color = "detection off", "grey"
            chip_det.props(f"color={color} text-color=white")
            chip_arm.text, color = arm_chip(R)
            chip_arm.props(f"color={color} text-color=white")
            E.want_detect = tab in (t_det.props["name"], t_3d.props["name"], t_rob.props["name"]) and det.live.value
            E.want_corners = tab == t_cal.props["name"]
            if tab == t_cal.props["name"]:
                cal.tick()
            elif tab == t_det.props["name"]:
                det.tick()
            elif tab == t_3d.props["name"]:
                d3.tick()
                arm3d.tick()
            elif tab == t_rob.props["name"]:
                rob.tick()
            if tab != t_rob.props["name"]:
                rob.kb.value = False    # the arm's keys only work where its state and STOP are on screen
        ui.timer(0.2, tick)


def arm_chip(R: Robot):
    if not R.link.connected:
        return "arm: not connected", "grey"
    if R.torque_off:
        return "arm: torque off", "warning"
    if R.busy:
        return f"arm: {R.busy}"[:50], "primary"
    if R.joints()[0] is None:
        return "arm: no joint states", "negative"
    return "arm: ready" + (" · air" if R.air else " · REAL moves"), "positive" if R.air else "deep-orange"


def robot_panel(E: Engine, R: Robot):
    with ui.row().classes("w-full no-wrap gap-3 items-start"):
        with ui.column().classes("flex-1 min-w-0 gap-2"):
            with ui.row().classes("w-full items-center gap-2"):
                kind = ui.toggle({"top": "Top view", "cam": "Camera"}, value="top").props("dense")
                side = ui.toggle({"left": "Left", "right": "Right"}, value="left").props("dense")
                ui.space()
                hint = ui.label().classes("text-sm text-amber-300")
                ui.label("A click:").classes("text-sm")
                mode = ui.toggle({"move": "chooses a target", "fit": "names a board corner"}, value="move") \
                    .props("dense").tooltip("Move a piece, or Table → robot fit")
            img = ui.interactive_image(events=["click"], sanitize=False).classes("w-full cursor-crosshair")
            ui.label("Coloured circles: pieces both cameras see. Yellow: the chosen piece, cyan: the chosen spot, orange "
                     "ring: the fingertips (with their height over the table), white dots: the fit points. Needs Live "
                     "detection on (Detection tab).").classes("text-xs opacity-60")
        with ui.column().classes("w-[440px] shrink-0 gap-3"):
            with ui.card().classes("w-full gap-2"):
                ui.label("Arm").classes("text-base font-medium")
                status = ui.label().classes("text-sm whitespace-pre-line font-mono")
                torque_banner = ui.label("Torque is off: the arm can be moved by hand. When done, hold the arm, Ctrl-C "
                                         "./lab.sh arm and start it again; this app reconnects by itself and the "
                                         "buttons come back.").classes("text-sm text-amber-400")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.switch("Air (dry run)").bind_value(R, "air").tooltip(
                        f"Every move slow; picks and places stop {AIR_MM:.0f} mm above the table, nothing is gripped")
                    air_note = ui.label().classes("text-sm")
                with ui.row().classes("gap-2"):
                    b_wake = ui.button("Wake", icon="north", on_click=lambda: cmd("wake")).props("dense").tooltip(
                        "Slowly up to home from wherever the arm lies. Can sweep sideways: clear table")
                    b_home = ui.button("Home", icon="home", on_click=lambda: cmd("home")).props("dense")
                    b_show = ui.button("Show", icon="front_hand", on_click=lambda: cmd("show")).props("dense") \
                        .tooltip("Hold the gripper out to the participant, as for a question")
                    b_open = ui.button("Open gripper", icon="open_in_full", on_click=lambda: cmd("open")).props("dense")
                    b_close = ui.button("Close gripper", icon="close_fullscreen", on_click=lambda: cmd("close")) \
                        .props("dense")
                b_stop = ui.button("STOP", icon="pan_tool", on_click=lambda: stop()) \
                    .props("color=negative size=lg").classes("w-full")
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    kb = ui.switch("Keyboard").tooltip("Off by itself when you leave this tab")
                    ui.label(f"1/q 2/w 3/e 4/r: joint 1-4 ± {JOG_STEP:.2f} rad · o/p: open/close gripper · h: home · "
                             "Esc: STOP").classes("text-xs opacity-70")

            with ui.card().classes("w-full gap-2"):
                ui.label("Move a piece").classes("text-base font-medium")
                ui.label("Click a piece in the picture to hover over it or pick it up, then click a free spot to put it "
                         "there. Home afterwards.").classes("text-xs opacity-60")
                fit_mode_note = ui.label("A click now names a board corner for the fit: switch it to 'chooses a target' "
                                         "above the picture.").classes("text-sm text-amber-400")
                target = ui.label().classes("text-sm whitespace-pre-line")
                holding = ui.label().classes("text-sm")
                with ui.row().classes("gap-2"):
                    b_hover = ui.button("Hover", icon="vertical_align_bottom", on_click=lambda: move("hover")) \
                        .props("dense").tooltip("Fingertips stop above it (approach height), slowly")
                    b_pick = ui.button("Pick", icon="pan_tool_alt", on_click=lambda: move("pick")).props("dense")
                    b_place = ui.button("Place", icon="place_item", on_click=lambda: move("place")).props("dense")
                    ui.button(icon="close", on_click=lambda: setattr(R, "target", None)).props("flat dense") \
                        .tooltip("Forget the target")

            # opening the fit switches the clicks to corners, closing it back to targets; the toggle above the picture
            # shows which, and overrides it
            with ui.expansion("Table → robot fit", icon="architecture",
                              on_value_change=lambda e: mode.set_value("fit" if e.value else "move")) \
                    .classes("w-full bg-zinc-900 rounded"):
                fit = fit_section(E, R)

    async def cmd(op, xy=None):
        try:
            await R.do(op, xy)
            log(R.last)
            ui.notify(R.last, type="positive")
            return True
        except Exception as e:
            log(f"{op} FAILED: {e}")
            ui.notify(f"{op} failed: {e}", type="negative", multi_line=True)
            return False

    async def key(e):
        """Keyboard control (the Keyboard switch): a held key repeats, but a step only starts once the last one ended."""
        if not kb.value or not e.action.keydown:
            return
        if e.key.escape:               # not space: it would also flip a focused switch
            return await stop()
        k = e.key.name.lower()
        if R.busy or not R.can_move:
            return
        if k in JOG_KEYS:
            joint, sign = JOG_KEYS[k]
            try:
                await R.do("jog", joint=joint, delta=sign * JOG_STEP)
            except Exception as err:
                log(f"jog FAILED: {err}")
                ui.notify(f"jog failed: {err}", type="negative", multi_line=True)
        elif k in ("o", "p", "h") and not e.action.repeat:
            await cmd({"o": "open", "p": "close", "h": "home"}[k])
    ui.keyboard(key, ignore=["input", "select", "textarea"])

    def why_not(op):
        """None if the op's button works now, else why it is greyed out."""
        t = R.target
        if not R.link.connected:
            return "no rosbridge connection: is ./lab.sh arm running?"
        if R.torque_off:
            return "torque is off: hold the arm, restart ./lab.sh arm"
        if R.busy:
            return f"still running: {R.busy}"
        if t is None:
            return "no target: click a piece in the picture"
        if why := R.reach(t["x"], t["y"]):
            return why
        if op == "pick" and t["kind"] != "piece":
            return (f"the click is not within {SNAP_MM:.0f} mm of a piece BOTH cameras see (Live detection on? a "
                    "piece seen by one camera only can't be picked)")
        if op == "place" and t["kind"] != "spot":
            return "a piece is there: click a free spot"
        return None

    async def move(op):
        t = R.target
        if t is None:
            return
        if await cmd(op, (t["x"], t["y"])):
            if op == "pick":
                R.holding = t.get("color") or "?"
                R.target = None
                ui.notify("Now click a free spot to put it", type="info")
            elif op == "place":
                R.holding = None

    async def stop():
        ok, why = await R.stop()
        R.last = "stop: confirmed" if ok else f"stop: NOT confirmed ({why})"
        log(R.last)
        if ok:
            ui.notify("Stopped", type="warning")
        else:
            ui.notify(f"The stop wasn't confirmed ({why}): the arm may still be moving. Power switch!",
                      type="negative", multi_line=True, timeout=0, close_button=True)

    def click(e):
        views = E.views
        if views is None:
            ui.notify("Calibrate first", type="warning")
            return
        view = views[SIDES.index(side.value)]
        top, flat = click_to_table(view, kind.value, e.image_x, e.image_y)
        if mode.value == "fit":         # a click names the board corner the fingertips touch
            fit.pick_corner(*flat)
            return
        d = E.det
        near = []
        for p, st in (d["matched"] if d else []):
            if st is not None and (dist := np.hypot(st[0] - top[0], st[1] - top[1])) < SNAP_MM:
                near.append((dist, p.color, float(st[0]), float(st[1])))
        if near:
            _, color, x, y = min(near)
            R.target = {"kind": "piece", "color": color, "x": x, "y": y}
        else:
            R.target = {"kind": "spot", "x": float(flat[0]), "y": float(flat[1])}
        t = R.target
        both = sum(st is not None for _, st in (d["matched"] if d else []))
        log(f"click -> {t['kind']} {t.get('color', '')} at table ({t['x']:.0f}, {t['y']:.0f}) mm; pieces both cameras "
            f"see: {both}" + "".join(f"; {op}: {why_not(op) or 'ok'}" for op in ("hover", "pick", "place")))
    img.on_mouse(click)

    def target_text():
        t = R.target
        if t is None:
            return "Target: none (click the picture)"
        what = f"{t['color']} piece" if t["kind"] == "piece" else "spot"
        lines = [f"Target: {what} at table ({t['x']:.0f}, {t['y']:.0f}) mm"]
        if R.T is not None:
            rx, ry, rz = R.to_robot(t["x"], t["y"])
            lines.append(f"robot ({rx:.0f}, {ry:.0f}, {rz:.0f}) mm")
        why = R.reach(t["x"], t["y"])
        lines.append("✓ in reach" if why is None else f"✗ {why}")
        if t["kind"] == "spot" and E.det:
            others = [(np.hypot(st[0] - t["x"], st[1] - t["y"]), p.color) for p, st in E.det["matched"] if st is not None]
            if others and (m := min(others))[0] < SPACING_MM:
                lines.append(f"⚠ the {m[1]} piece is {m[0]:.0f} mm away: the open fingers need {SPACING_MM:.0f}")
        op = "pick" if t["kind"] == "piece" else "place"
        if why := why_not(op):
            lines.append(f"{op.capitalize()} is off: {why}")
        if R.air:
            lines.append(f"Air is on: a pick stops {AIR_MM:.0f} mm above the piece and grips nothing")
        return "\n".join(lines)

    def status_text():
        q, grip = R.joints()
        lines = [f"{R.link.label}: " + ("connected" if R.link.connected else
                                         "not connected (start ./lab.sh arm)" if R.link.real else "")]
        if q is None:
            lines.append("joints: no fresh /joint_states")
        else:
            tip = np.array(fk(q))
            lines.append("joints: " + " ".join(f"{v:+.2f}" for v in q) + " rad")
            lines.append(f"fingertips: robot ({tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f}) mm")
            if R.T is not None:
                t = R.to_table(tip)
                lines.append(f"            table ({t[0]:.0f}, {t[1]:.0f}), {t[2]:.0f} mm over it")
        if grip is not None:
            lines.append(f"gripper: {'open' if grip > 0.01 else 'closed' if grip < 0 else 'part open'} ({grip * 1000:.0f} mm)")
        lines.append(f"running: {R.busy}" if R.busy else "idle")
        lines.append(f"last: {R.last}")
        return "\n".join(lines)

    def overlay(view, k):
        """SVG on top of the picture: target, fingertips, fit points."""
        w = view.image_size[0] if k == "cam" else view.maps.shape[1]
        s = max(1.0, w / 700)
        out = []

        def circle(xyz, r_mm, color, width, label=None, dash=False):
            u, v = to_px(view, k, xyz)
            u2, v2 = to_px(view, k, (xyz[0] + r_mm, xyz[1], xyz[2]))
            r = max(4 * s, float(np.hypot(u2 - u, v2 - v)))
            if not (np.isfinite(u) and np.isfinite(v)):
                return
            out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="{r:.1f}" fill="none" stroke="{color}" '
                       f'stroke-width="{width * s:.1f}"' + (f' stroke-dasharray="{4 * s:.0f}"' if dash else "") + "/>")
            if label:
                out.append(f'<text x="{u + r + 3 * s:.1f}" y="{v - r:.1f}" fill="{color}" font-size="{13 * s:.0f}" '
                           f'font-family="sans-serif" paint-order="stroke" stroke="#000" stroke-width="{3 * s:.1f}">'
                           f'{label}</text>')
        h = view.z
        t = R.target
        if t is not None:
            if t["kind"] == "piece":
                circle((t["x"], t["y"], h), pieces.PIECE_D_MM / 2 + 6, "#facc15", 3, "target")
            else:
                circle((t["x"], t["y"], 0.0), pieces.PIECE_D_MM / 2, "#22d3ee", 3, "put here", dash=True)
        if mode.value == "fit":
            for i, (tab, _) in enumerate(R.points):
                circle(tab, 3, "#ffffff", 2, str(i + 1))
            if fit.corner is not None:
                circle((*fit.corner, 0.0), 6, "#f472b6", 2, "corner")
        tip = R.tip()
        if tip is not None and R.T is not None:
            tt = R.to_table(tip)
            circle(tt, 8, "#fb923c", 3, f"fingertips {tt[2]:.0f} mm")
        return "".join(out)

    shown = {"key": None, "t": 0.0, "svg": None}

    class Panel:
        def tick(self):
            target.text = target_text()
            holding.text = (f"Holding: {R.holding}" + (" (air: nothing gripped)" if R.air else "")) if R.holding else ""
            status.text = status_text()
            torque_banner.set_visibility(R.torque_off)
            air_note.text = "" if R.air else "REAL moves: picks grip, places let go"
            air_note.classes(replace="text-sm text-red-400 font-medium")
            for b in (b_wake, b_home, b_show, b_open, b_close):
                b.set_enabled(R.can_move)
            b_stop.set_enabled(R.link.connected)
            for b, op in ((b_hover, "hover"), (b_pick, "pick"), (b_place, "place")):
                b.set_enabled(why_not(op) is None)
            fit_mode_note.set_visibility(mode.value == "fit")
            hint.text = ("Click the board corner the fingertips touch" if mode.value == "fit" else
                         "Click a piece, or a free spot" if E.views is not None else "Calibrate first")
            fit.tick()
            d = E.det
            dkey = d["key"] if d else None
            key = (E.frame_id, dkey, kind.value, side.value)
            if key != shown["key"] and time.monotonic() - shown["t"] >= 0.15:
                shown["key"], shown["t"] = key, time.monotonic()
                k = f"{E.frame_id}-{dkey[0] if dkey else 0}-{dkey[1] if dkey else 0}-{E.params_id}"
                img.set_source(f"/v2img/{kind.value}_{side.value}.jpg?k={k}&lone=0&misses=0")
            views = E.views
            svg = overlay(views[SIDES.index(side.value)], kind.value) if views else ""
            if svg != shown["svg"]:
                shown["svg"] = svg
                img.content = svg
    panel = Panel()
    panel.kb = kb
    return panel


def fit_section(E: Engine, R: Robot):
    ui.label("1. The board lies where it was for the calibration capture. Hold the arm, press Torque off, close the "
             "fingers by hand so the fingertips meet.\n"
             "2. Touch the fingertips to a board corner, click that corner in the picture (or type it, mm from the "
             "(0, 0) corner the axes mark), Add point.\n"
             "3. 4-6 corners spread over the board, then Save. Accept if every error is under "
             f"{robot_frame.ACCEPT_MM:.0f} mm.\n"
             "Check a saved fit: touch new corners; the 'saved' column is the --check.\n"
             "4. Restart ./lab.sh arm: lay the arm down, hands off, Ctrl-C, start it again.") \
        .classes("text-xs opacity-70 whitespace-pre-line")
    with ui.row().classes("items-center gap-2"):
        b_torque = ui.button("Torque off", icon="power_off").props("dense color=warning")
        live = ui.label().classes("text-sm font-mono")
    with ui.row().classes("items-end gap-2 no-wrap"):
        fx = ui.number("table x mm", format="%.1f").classes("w-28")
        fy = ui.number("table y mm", format="%.1f").classes("w-28")
        b_add = ui.button("Add point", icon="add_location_alt").props("dense")
    table = ui.table(columns=[{"name": "i", "label": "#", "field": "i"},
                              {"name": "table", "label": "table x, y", "field": "table"},
                              {"name": "robot", "label": "fingertips (robot)", "field": "robot"},
                              {"name": "saved", "label": "saved fit", "field": "saved"},
                              {"name": "new", "label": "new fit", "field": "new"}],
                     rows=[]).props("dense flat").classes("w-full")
    result = ui.label().classes("text-sm whitespace-pre-line")
    with ui.row().classes("gap-2"):
        ui.button("Remove last", icon="undo", on_click=lambda: (R.points.pop() if R.points else None, refresh())) \
            .props("dense flat")
        ui.button("Clear", icon="delete_sweep", on_click=lambda: (R.points.clear(), refresh())).props("dense flat")
        b_save = ui.button("Save fit", icon="save").props("dense")

    with ui.dialog() as confirm, ui.card():
        ui.label("Hold the arm first: with torque off its motors let go and it falls.").classes("font-medium")
        ui.label("Afterwards restart ./lab.sh arm (don't turn torque back on: the arm would snap back to its last "
                 "command).").classes("text-sm")
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=confirm.close).props("flat")
            ui.button("I'm holding it: torque off", on_click=lambda: torque_off()).props("color=warning")
    b_torque.on_click(confirm.open)

    async def torque_off():
        confirm.close()
        try:
            msg = await run.io_bound(R.link.torque_off)
        except Exception as e:
            ui.notify(f"Torque off failed: {e}", type="negative", multi_line=True)
            return
        if R.link.real:
            R.torque_off = True
        ui.notify(f"Torque off{': ' + msg if msg else ''}. Move the arm by hand.", type="warning")

    def add():
        if fx.value is None or fy.value is None:
            ui.notify("Click the corner in the picture, or type its table x and y", type="warning")
            return
        try:
            R.add_point(fx.value, fy.value)
        except ValueError as e:
            ui.notify(str(e), type="negative")
            return
        st.corner = None
        refresh()
    b_add.on_click(add)

    def save():
        try:
            T = R.save_fit()
        except ValueError as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        refresh()
        ui.notify(f"Saved {R.path.name}: table plane at robot z = {T[2, 3]:.1f} mm. The runner uses it now.",
                  type="positive", multi_line=True)
    b_save.on_click(save)

    def refresh():
        saved, T, err, problem = R.fit()
        rows = []
        for i, (tab, rob) in enumerate(R.points):
            rows.append({"i": i + 1, "table": f"{tab[0]:.1f}, {tab[1]:.1f}",
                         "robot": f"{rob[0]:.0f}, {rob[1]:.0f}, {rob[2]:.0f}",
                         "saved": "–" if saved is None else f"{saved[i]:.1f} mm",
                         "new": "–" if err is None else f"{err[i]:.1f} mm"})
        table.rows = rows
        table.update()
        lines = []
        if saved is not None:
            ok = saved.max() < robot_frame.ACCEPT_MM
            lines.append(f"Saved fit: max {saved.max():.1f} mm → " + ("OK" if ok else
                         f"NOT OK, needs < {robot_frame.ACCEPT_MM:.0f} mm"))
        elif R.T is None:
            lines.append("No saved fit yet.")
        if err is not None:
            ok = err.max() < robot_frame.ACCEPT_MM
            lines.append(f"New fit: max {err.max():.1f} mm → " + ("OK" if ok else
                         f"NOT OK, needs < {robot_frame.ACCEPT_MM:.0f} mm") + f"; table plane at robot z = {T[2, 3]:.1f} mm")
        if problem:
            lines.append(problem)
        result.text = "\n".join(lines)
        b_save.set_enabled(T is not None)

    class Fit:
        corner = None

        def pick_corner(self, x, y):
            """A click on the table plane -> the nearest board corner (multiples of the square size)."""
            sq = E.square
            self.corner = (round(x / sq) * sq, round(y / sq) * sq)
            fx.value, fy.value = self.corner

        def tick(self):
            tip = R.tip()
            if tip is None:
                live.text = "no fresh joints"
            elif R.T is not None:
                t = R.to_table(tip)
                live.text = f"fingertips at table ({t[0]:.1f}, {t[1]:.1f}), {t[2]:.1f} mm up"
            else:
                live.text = f"fingertips at robot ({tip[0]:.0f}, {tip[1]:.0f}, {tip[2]:.0f})"
            b_torque.set_enabled(R.link.connected and not R.busy and not R.torque_off)
            b_add.set_enabled(tip is not None)

    st = Fit()
    refresh()
    return st


class Arm3D:
    """The arm's live pose in calib_app's 3D view: its centre line as cylinders, in the table frame via the fit."""

    RADII = (22, 16, 14, 8)     # mm: base column, upper arm, forearm, gripper

    def __init__(self, scene, R: Robot):
        self.scene, self.R = scene, R
        self.objs = None
        self.shown = None

    def _create(self, n):
        with self.scene:
            links = [self.scene.cylinder(r * 1e-3, r * 1e-3, 1.0, 16).material("#a1a1aa" if i else "#52525b")
                     for i, r in enumerate(self.RADII[:n])]
            joints = [self.scene.sphere(0.018).material("#71717a") for _ in range(n - 1)]
            tip = self.scene.sphere(0.007).material("#fb923c")
            label = self.scene.text("arm", "background: rgba(0,0,0,.6); color: #fff; padding: 1px 4px; "
                                           "border-radius: 3px")
        self.objs = links, joints, tip, label

    def tick(self):
        R = self.R
        q, _ = R.joints()
        if R.T is None or q is None:
            if self.objs and self.shown is not None:
                for o in [*self.objs[0], *self.objs[1], self.objs[2], self.objs[3]]:
                    o.visible(False)
                self.shown = None
            return
        key = (tuple(round(v, 3) for v in q), id(R.T))
        if key == self.shown:
            return
        pts = R.to_table(arm_points(q)) * 1e-3
        if self.objs is None:
            self._create(len(pts) - 1)
        links, joints, tip, label = self.objs
        for i, cyl in enumerate(links):
            a, b = pts[i], pts[i + 1]
            d = b - a
            length = float(np.linalg.norm(d))
            y = d / length
            ref = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x = np.cross(y, ref)
            x /= np.linalg.norm(x)
            z = np.cross(x, y)
            # three.js cylinders stand along y: turn y onto the link, stretch to its length
            cyl.move(*map(float, (a + b) / 2)).rotate_R(np.c_[x, y, z].tolist()).scale(1, length, 1).visible(True)
        for sph, p in zip(joints, pts[1:-1]):
            sph.move(*map(float, p)).visible(True)
        tip.move(*map(float, pts[-1])).visible(True)
        label.move(*map(float, pts[1])).visible(True)
        self.shown = key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs=2, metavar=("LEFT", "RIGHT"),
                    help="image files instead of cameras: comma-separated globs per camera (development)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--host", default="localhost", help="rosbridge host (port 9090)")
    ap.add_argument("--fake-arm", action="store_true", help="a simulated arm instead of ROS (development)")
    args = ap.parse_args()
    E = Engine(tuple(args.files) if args.files else None)
    try:
        E.connect()
    except Exception as e:
        print(f"cameras not connected: {e}")
    threading.Thread(target=E.load_captures, daemon=True).start()
    link = SimLink() if args.fake_arm else RosLink(args.host)
    link.start()
    R = Robot(E, link)
    build_ui(E, R)
    app.on_shutdown(E.disconnect)
    ui.run(title="Calibration + robot", port=args.port, reload=False, dark=True, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
