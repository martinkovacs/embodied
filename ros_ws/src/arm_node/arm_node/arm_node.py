"""Scripted OpenMANIPULATOR-X motions behind /arm/cmd (architecture.md section 3).

/arm/cmd     std_msgs/String JSON  {"id": 7, "op": "pick", "x": 120, "y": -40}   x, y: robot base frame (link1), mm
              z: table surface height at (x, y), mm, required for pick/place (GRASP_Z etc. are heights above it)
             op: pick | place | hover (x, y, z) | show | home | open | stop | wake
             air: true (optional) makes a dry run of the command: slow, pick / place stop AIR_MM above the piece
             hover: the fingertips go to the approach height over (x, y), always slowly, and stay there
             wake: slowly up to home from wherever the arm lies, never lower than it is (startup, clear table)
/arm/status  std_msgs/String JSON  {"id": 7, "state": "started" | "done" | "failed", "reason": "..."}

One command at a time: another one arriving meanwhile fails with "busy". stop cancels the running motion and
only answers "done" once the cancellation is confirmed (the controller accepts it, or the goal had already
terminated, or nothing was running); otherwise it answers "failed" and the reason (rejected / no answer):
the operator must then assume the arm may still be moving and use the power switch. The running command fails
with "stopped" either way. Every controller answer is timed out (goal acceptance 5 s; arm result, trajectory
duration + 5 s; gripper result 5 s; cancel 2 s): a controller that never answers (USB dropout) fails the
command instead of leaving the node busy for ever, and no later command moves anything until that goal is
confirmed cancelled.

Run after the ROBOTIS bringup (hardware.launch.py or gazebo.launch.py):  ros2 run arm_node arm_node
"""
import json
import time
from math import isfinite

import rclpy
import rclpy.task
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from arm_node.kinematics import (CLEAR, GRASP_Z, HOME, LIFT, PLACE_Z, SHOW, check_travel, fk, lift_path,
                                 vertical, wake_path)

JOINTS = ["joint1", "joint2", "joint3", "joint4"]
OPEN, CLOSED = 0.019, -0.01     # gripper_left_joint limits, m; closing on a piece just stalls on it
EMPTY_MARGIN = 0.002            # m: tune in the lab: a grasp stalls above CLOSED, a miss closes right to it
MAX_SPEED = 0.6                 # rad/s, for the joint that moves most
AIR_SPEED = 0.25                # rad/s, the same for dry runs and hovers
AIR_MM = 40.0                   # a dry run's pick / place stops this much higher: fingertips ~45 mm over the table
MIN_MOVE_S = 1.0
GOAL_ACCEPT_S = 5.0             # a live controller answers this fast
CANCEL_ACK_S = 2.0              # stop: how long the controller may take to answer its cancel request
RESULT_EXTRA_S = 5.0            # arm goal result: trajectory duration + this
GRIP_RESULT_S = 5.0             # gripper goal result
JOINT_STALE_S = 1.0             # /joint_states: without a reading fresher than this, nothing may move


def trajectory(q_from, points, speed=MAX_SPEED):
    """[JointTrajectoryPoint] for one trajectory from q_from through points, and its total duration s: per
    segment max(largest joint delta / speed, 0.05 s), cumulative, scaled up to MIN_MOVE_S. Zero velocity
    at the ends, central differences in between, so the controller fits smooth splines through all of it."""
    qs = [list(q_from)] + [list(p) for p in points]
    ts = [0.0]
    for a, b in zip(qs, qs[1:]):
        ts.append(ts[-1] + max(max(abs(m - n) for m, n in zip(a, b)) / speed, 0.05))
    if ts[-1] < MIN_MOVE_S:                 # scale up, keeping the pacing
        ts = [t * MIN_MOVE_S / ts[-1] for t in ts]
    out = []
    for i, q in enumerate(qs):
        if 0 < i < len(qs) - 1:
            t0, t2 = ts[i - 1], ts[i + 1]   # central difference over the neighbour segment times
            v = [(n - m) / (t2 - t0) for m, n in zip(qs[i - 1], qs[i + 1])]
        else:
            v = [0.0] * len(JOINTS)
        out.append(JointTrajectoryPoint(positions=q, velocities=v,
                                        time_from_start=Duration(seconds=ts[i]).to_msg()))
    return out, ts[-1]


class ArmNode(Node):
    def __init__(self):
        super().__init__("arm_node")
        self.arm = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.gripper = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd")
        self.status = self.create_publisher(String, "/arm/status", 10)
        self.q = None                   # current arm joints, from /joint_states
        self.q_at = None                # time.monotonic() when the last valid q came in
        self.table_z = 0.0              # z of the table surface at the last pick/place spot
        self.op_xy = None               # (x, y) of the running (or last interrupted) pick/place: the lift home
        self.goal = None                # running controller goal, cancelled by stop
        self.sending = None             # the goal request in flight, while its acceptance is awaited
        self.adrift = None              # a timed-out goal (handle) or request (future) that may yet move the arm
        self.busy = self.stopped = False
        self.speed = MAX_SPEED          # of the running command
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        # reentrant: a stop has to get through while a command is still awaiting the controllers
        self.create_subscription(String, "/arm/cmd", self.on_cmd, 10, callback_group=ReentrantCallbackGroup())

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if not any(j in pos for j in JOINTS):
            return                      # not about the arm (the gripper's broadcaster publishes too)
        # fail closed: a partial or non-finite reading (USB dropout) must not become the node's picture of the arm
        if all(j in pos and isfinite(pos[j]) for j in JOINTS):
            self.q, self.q_at = [pos[j] for j in JOINTS], time.monotonic()
        else:
            self.q = None               # the last full reading may predate the fault: don't move from it

    def joints(self):
        """The current joints, fail closed: RuntimeError without a valid reading younger than JOINT_STALE_S. A
        stale picture (the controllers died mid-motion, a stop left the arm somewhere) must not become the
        start of a trajectory, or the clearance checks would be solved against the wrong arm."""
        if self.q is None:
            raise RuntimeError("no valid /joint_states (all reported arm joints, all finite)")
        if (age := time.monotonic() - self.q_at) > JOINT_STALE_S:
            raise RuntimeError(f"/joint_states is {age:.1f} s old (over {JOINT_STALE_S:.0f} s)")
        return self.q

    async def on_cmd(self, msg):
        cmd = {}
        try:
            cmd = json.loads(msg.data)
            if cmd["op"] == "stop":
                await self.on_stop(cmd)
                return
            if self.busy:
                raise RuntimeError("busy")
            self.busy, self.stopped = True, False
            try:
                self.reply(cmd, "started")
                state, reason = await self.settle_adrift()
                if state != "done":
                    raise RuntimeError(f"an earlier goal may still move the arm (check it, then restart arm_node): "
                                       f"{reason}")
                await self.run(cmd)
            finally:
                self.busy = False
            self.reply(cmd, "done")
        except Exception as e:  # bad command, out of reach, controller error, timeout, stopped
            self.reply(cmd, "failed", str(e))

    async def on_stop(self, cmd):
        """Stop cancels the running goal and answers once the cancellation is verified: "done" only when
        nothing was running or the controller says it cancelled (or had already terminated); "failed" with
        the reason otherwise, so the operator knows the arm may still be moving. Never raises: a stop has to
        get through whatever else the node is doing, and it must be answered."""
        self.stopped = True
        try:
            state, reason = await self._stop_outcome()
        except Exception as e:          # failed timer, rclpy error: still answer, never raise into the spin
            state, reason = "failed", f"stop: {e}"
        self.reply(cmd, state, reason)

    async def settle_adrift(self):
        """A goal that timed out may still run, or a request still be accepted: nothing else may move until the
        controller confirms it is cancelled. ("done", None) when nothing is adrift, else ("failed", reason), and it
        stays adrift. A controller that never answers keeps the node refusing until it is restarted."""
        adrift = self.adrift
        if adrift is None:
            return "done", None
        if isinstance(adrift, rclpy.task.Future):
            if not adrift.done():
                return "failed", "an earlier goal request is still unanswered"
            adrift = adrift.result()
            if not adrift.accepted:
                self.adrift = None
                return "done", None
        state, reason = await self.cancelled(adrift)
        if state == "done":
            self.adrift = None
        return state, reason

    async def _stop_outcome(self):
        if self.adrift is not None:     # never beside a running goal: a command settles this before it moves
            return await self.settle_adrift()
        handle = self.goal
        if handle is None:
            fut = self.sending          # a command is still awaiting acceptance: don't lose the stop here, or
            if fut is None:             # its goal, if it lands at all, can never be cancelled and verified
                return "done", None     # nothing running
            try:                        # bounded like send()'s own acceptance wait, then cancel and verify
                handle = await self.await_ft(fut, GOAL_ACCEPT_S, "goal acceptance")
            except TimeoutError:
                # send(), which owns the request, has failed and late-cancels it if it is accepted after all:
                # nothing was ever confirmed though
                return "failed", f"stop: goal acceptance still pending after {GOAL_ACCEPT_S:.0f} s"
            except Exception as e:      # no answer to go on: don't claim nothing was accepted
                return "failed", f"stop: goal request failed, acceptance unknown: {e}"
            if not handle.accepted:     # rejected: nothing is moving
                return "done", None
        return await self.cancelled(handle)

    async def cancelled(self, handle):
        """Cancel one goal and verify it: ("done", None), or ("failed", reason) when the controller refuses."""
        res = await self.await_ft(handle.cancel_goal_async(), CANCEL_ACK_S, "cancel request")
        code = CancelGoal.Response      # old-style names in Humble: the cancel answer's class
        if res.return_code == code.ERROR_NONE and res.goals_canceling:
            return "done", None         # the controller is cancelling the goal
        if res.return_code == code.ERROR_GOAL_TERMINATED:
            return "done", None         # already finished: the arm holds where the goal left it, not moving
        if res.return_code == code.ERROR_NONE:  # accepted call but nothing to cancel: rejected upstream
            return "failed", "stop: the controller would not cancel the goal (rejected or unknown goal id)"
        why = {code.ERROR_REJECTED: "rejected", code.ERROR_UNKNOWN_GOAL_ID: "unknown goal id"}.get(
            res.return_code, f"return code {res.return_code}")
        return "failed", f"stop: the controller would not cancel the goal ({why})"

    def reply(self, cmd, state, reason=None):
        out = {"id": cmd.get("id") if isinstance(cmd, dict) else None, "state": state}
        if reason is not None:
            out["reason"] = reason
        self.status.publish(String(data=json.dumps(out)))
        self.get_logger().info(json.dumps(out))

    async def run(self, cmd):
        op, air = cmd["op"], bool(cmd.get("air"))
        self.speed = AIR_SPEED if air or op in ("hover", "wake") else MAX_SPEED
        if op == "home":
            await self.travel(HOME, self.table_z)
        elif op == "wake":
            await self.move(wake_path(self.joints()))
        elif op == "show":
            await self.travel(SHOW, self.table_z)
        elif op == "open":
            await self.grip(OPEN)
        elif op == "hover":
            if cmd.get("z") is None:
                raise ValueError("hover needs z (table surface height at (x, y) in the robot frame, mm)")
            table_z = float(cmd["z"])
            q = vertical(float(cmd["x"]), float(cmd["y"]), table_z + GRASP_Z + LIFT, table_z + GRASP_Z + LIFT)[0]
            visit_z, self.table_z = max(self.table_z, table_z), table_z
            await self.travel(q, visit_z)
        elif op in ("pick", "place"):
            if cmd.get("z") is None:        # the table height varies over the layout: assuming 0 would dig in
                raise ValueError(f"{op} needs z (table surface height at (x, y) in the robot frame, mm)")
            x, y = float(cmd["x"]), float(cmd["y"])
            table_z = float(cmd["z"])
            # descent and ascent solved before anything moves, so an unreachable spot fails cleanly
            up_z, down_z = table_z + GRASP_Z + LIFT, table_z + (GRASP_Z if op == "pick" else PLACE_Z) + (AIR_MM if air else 0)
            descent, ascent = vertical(x, y, up_z, down_z), vertical(x, y, down_z, up_z)
            visit_z = max(self.table_z, table_z)    # the approach must clear both tables (the runner checks the max)
            self.op_xy, self.table_z = (x, y), table_z  # remembered for the recovery lift of an interrupted op
            await self.travel(descent[0], visit_z)  # approach is a checked travel move; descent/ascent is not
            if op == "pick":
                await self.grip(OPEN)       # only now: a still-held piece isn't dropped from the approach height
            await self.move(descent)
            if op == "pick":
                res = await self.grip(CLOSED)
                pos = getattr(res.result, "position", None)     # absent (fake hardware): skip the check
                if not air and pos is not None and pos <= CLOSED + EMPTY_MARGIN:   # jaws at the limit: nothing there
                    await self.move(ascent)                     # clear the piece level before failing
                    raise RuntimeError("gripper closed on nothing: missed the piece")
            else:
                await self.grip(OPEN)
            await self.move(ascent)
            self.op_xy = None
        else:
            raise ValueError(f"unknown op {op!r}")

    async def travel(self, q, table_z):
        """Travel move with a clearance check, lifting straight up first when the arm already sits low."""
        q_from = self.joints()
        z = fk(q_from)[2]
        if not (z >= table_z + CLEAR):          # fail closed on nan; a low arm (stop mid-descent or hand move)
            try:                                # first lifts, from the interrupted op's spot or its own column
                lift = lift_path(q_from, table_z, self.op_xy)
            except ValueError as e:
                raise RuntimeError(str(e)) from None
            await self.move(lift)               # one multi-point straight-up trajectory, not path-checked
            q_from = lift[-1]
        try:
            check_travel(q_from, q, table_z)
        except ValueError as e:
            raise RuntimeError(str(e))
        await self.move([q])

    async def move(self, points):
        """One multi-point trajectory from where the arm is (per /joint_states) through points."""
        pts, total = trajectory(self.joints(), points, self.speed)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        goal.trajectory.points = pts
        res = await self.send(self.arm, goal, total + RESULT_EXTRA_S, "arm goal")
        if res.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"arm controller: status {res.status} {res.result.error_string}")

    async def grip(self, position):
        goal = GripperCommand.Goal()
        goal.command.position = position
        return await self.send(self.gripper, goal, GRIP_RESULT_S, "gripper goal")

    async def send(self, client, goal, result_timeout, what):
        """Send one goal, await its acceptance and its result with timeouts: a controller that never answers
        (USB dropout) is cancelled and the command fails instead of leaving the node busy for ever. stop still
        cancels the running goal and fails the command with "stopped"."""
        if self.stopped:
            raise RuntimeError("stopped")
        if not client.server_is_ready():
            raise RuntimeError("controller not running (ros2 control list_controllers)")
        fut = client.send_goal_async(goal)
        self.sending = fut                     # a stop before acceptance waits for this answer too
        try:
            handle = await self.await_ft(fut, GOAL_ACCEPT_S, f"{what} acceptance")
        except TimeoutError as e:
            def cancel_late(f):                # a goal accepted this late must not move the arm for a command
                try:                           # whose caller is already gone; a failed request has no handle
                    if (h := f.result()).accepted:
                        h.cancel_goal_async()
                except Exception:
                    pass
            fut.add_done_callback(cancel_late)
            self.adrift = fut
            raise RuntimeError(str(e)) from None    # no handle yet: the goal request itself is dropped
        finally:
            self.sending = None                # settled either way: a later stop still sees self.goal
        if not handle.accepted:
            raise RuntimeError("goal rejected")
        self.goal = handle
        # no cancel here: stop owns the cancel requests (a stop before or during acceptance has just done
        # one). whatever a stop did, the command fails with "stopped" if the goal so much as reaches the end.
        try:
            res = await self.await_ft(handle.get_result_async(), result_timeout, f"{what} result")
        except TimeoutError as e:
            handle.cancel_goal_async()  # tell the controller to stop on the way out too
            self.adrift = handle        # and the next command or stop verifies that it did
            raise RuntimeError(str(e)) from None
        finally:
            self.goal = None
        if self.stopped:
            raise RuntimeError("stopped")
        return res

    async def await_ft(self, fut, timeout, what):
        """Await an rclpy future with a timeout (rclpy coroutines can't use asyncio): a one-shot timer races
        the future and whichever finishes first settles the shared future. TimeoutError on a silent controller."""
        done = rclpy.task.Future(executor=self.executor)

        def settle(*_):                 # future callbacks are called with the future, timers' with nothing
            if not done.done():
                done.set_result(None)

        timer = self.create_timer(timeout, settle)
        fut.add_done_callback(settle)
        try:
            await done
        finally:
            timer.cancel()
            self.destroy_timer(timer)   # through the node: Timer.destroy() would stay in its timer list
        if not fut.done():
            raise TimeoutError(f"{what}: no answer from the controller in {timeout:.0f} s")
        return fut.result()


def main():
    rclpy.init()
    try:
        rclpy.spin(ArmNode())
    except KeyboardInterrupt:
        pass
