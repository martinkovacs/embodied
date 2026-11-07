"""Scripted OpenMANIPULATOR-X motions behind /arm/cmd (architecture.md section 3).

/arm/cmd     std_msgs/String JSON  {"id": 7, "op": "pick", "x": 120, "y": -40}   x, y: robot base frame (link1), mm
              z: table surface height at (x, y), mm, required for pick/place (GRASP_Z etc. are heights above it)
             op: pick | place (x, y, z) | show | home | open | stop
/arm/status  std_msgs/String JSON  {"id": 7, "state": "started" | "done" | "failed", "reason": "..."}

One command at a time: another one arriving meanwhile fails with "busy". stop answers "done" at once and cancels
the running motion: the arm holds where it is and the running command fails with "stopped". Every controller
answer is timed out (goal 5 s; arm result, trajectory duration + 5 s; gripper result 5 s): a controller that
never answers (USB dropout) fails the command instead of leaving the node busy for ever.

Run after the ROBOTIS bringup (hardware.launch.py or gazebo.launch.py):  ros2 run arm_node arm_node
"""
import json
from math import isfinite

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from arm_node.kinematics import (CLEAR, GRASP_Z, HOME, LIFT, PLACE_Z, SHOW, check_travel, fk, lift_path,
                                 vertical)

JOINTS = ["joint1", "joint2", "joint3", "joint4"]
OPEN, CLOSED = 0.019, -0.01     # gripper_left_joint limits, m; closing on a piece just stalls on it
EMPTY_MARGIN = 0.002            # m: tune in the lab: a grasp stalls above CLOSED, a miss closes right to it
MAX_SPEED = 0.6                 # rad/s, for the joint that moves most
MIN_MOVE_S = 1.0
GOAL_ACCEPT_S = 5.0             # a live controller answers this fast
RESULT_EXTRA_S = 5.0            # arm goal result: trajectory duration + this
GRIP_RESULT_S = 5.0             # gripper goal result


def trajectory(q_from, points):
    """[JointTrajectoryPoint] for one trajectory from q_from through points, and its total duration s: per
    segment max(largest joint delta / MAX_SPEED, 0.05 s), cumulative, scaled up to MIN_MOVE_S. Zero velocity
    at the ends, central differences in between, so the controller fits smooth splines through all of it."""
    qs = [list(q_from)] + [list(p) for p in points]
    ts = [0.0]
    for a, b in zip(qs, qs[1:]):
        ts.append(ts[-1] + max(max(abs(m - n) for m, n in zip(a, b)) / MAX_SPEED, 0.05))
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
        self.table_z = 0.0              # z of the table surface at the last pick/place spot
        self.op_xy = None               # (x, y) of the running (or last interrupted) pick/place: the lift home
        self.goal = None                # running controller goal, cancelled by stop
        self.busy = self.stopped = False
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        # reentrant: a stop has to get through while a command is still awaiting the controllers
        self.create_subscription(String, "/arm/cmd", self.on_cmd, 10, callback_group=ReentrantCallbackGroup())

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        # fail closed: a non-finite reading (USB dropout) must not become the node's picture of the arm
        if all(j in pos and isfinite(pos[j]) for j in JOINTS):
            self.q = [pos[j] for j in JOINTS]

    async def on_cmd(self, msg):
        cmd = {}
        try:
            cmd = json.loads(msg.data)
            if cmd["op"] == "stop":
                self.stopped = True
                if self.goal:
                    self.goal.cancel_goal_async()
            elif self.busy:
                raise RuntimeError("busy")
            else:
                self.busy, self.stopped = True, False
                try:
                    self.reply(cmd, "started")
                    await self.run(cmd)
                finally:
                    self.busy = False
            self.reply(cmd, "done")
        except Exception as e:  # bad command, out of reach, controller error, timeout, stopped
            self.reply(cmd, "failed", str(e))

    def reply(self, cmd, state, reason=None):
        out = {"id": cmd.get("id") if isinstance(cmd, dict) else None, "state": state}
        if reason is not None:
            out["reason"] = reason
        self.status.publish(String(data=json.dumps(out)))
        self.get_logger().info(json.dumps(out))

    async def run(self, cmd):
        op = cmd["op"]
        if op == "home":
            await self.travel(HOME, self.table_z)
        elif op == "show":
            await self.travel(SHOW, self.table_z)
        elif op == "open":
            await self.grip(OPEN)
        elif op in ("pick", "place"):
            if cmd.get("z") is None:        # the table height varies over the layout: assuming 0 would dig in
                raise ValueError(f"{op} needs z (table surface height at (x, y) in the robot frame, mm)")
            x, y = float(cmd["x"]), float(cmd["y"])
            table_z = float(cmd["z"])
            # descent and ascent solved before anything moves, so an unreachable spot fails cleanly
            up_z, down_z = table_z + GRASP_Z + LIFT, table_z + (GRASP_Z if op == "pick" else PLACE_Z)
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
                if pos is not None and pos <= CLOSED + EMPTY_MARGIN:    # jaws at the limit: nothing between them
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
        if self.q is None:
            raise RuntimeError("no /joint_states yet")
        q_from = self.q
        z = fk(self.q)[2]
        if not (z >= table_z + CLEAR):          # fail closed on nan; a low arm (stop mid-descent or hand move)
            try:                                # first lifts, from the interrupted op's spot or its own column
                lift = lift_path(self.q, table_z, self.op_xy)
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
        if self.q is None:
            raise RuntimeError("no /joint_states yet")
        pts, total = trajectory(self.q, points)
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
        try:
            handle = await self.await_ft(client.send_goal_async(goal), GOAL_ACCEPT_S, f"{what} acceptance")
        except TimeoutError as e:
            raise RuntimeError(str(e)) from None    # no handle yet: the goal request itself is dropped
        if not handle.accepted:
            raise RuntimeError("goal rejected")
        self.goal = handle
        if self.stopped:  # stop came in while the goal was being sent
            handle.cancel_goal_async()
        try:
            res = await self.await_ft(handle.get_result_async(), result_timeout, f"{what} result")
        except TimeoutError as e:
            handle.cancel_goal_async()  # tell the controller to stop on the way out too
            raise RuntimeError(str(e)) from None
        finally:
            self.goal = None
        if self.stopped:
            raise RuntimeError("stopped")
        return res

    async def await_ft(self, fut, timeout, what):
        """Await an rclpy future with a timeout (rclpy coroutines can't use asyncio): a one-shot timer races
        the future and whichever finishes first settles the shared future. TimeoutError on a silent controller."""
        done = self.create_future()

        def settle():
            if not done.done():
                done.set_result(None)

        timer = self.create_timer(timeout, settle)
        fut.add_done_callback(settle)
        try:
            await done
        finally:
            timer.cancel()
            timer.destroy()
        if not fut.done():
            raise TimeoutError(f"{what}: no answer from the controller in {timeout:.0f} s")
        return fut.result()


def main():
    rclpy.init()
    try:
        rclpy.spin(ArmNode())
    except KeyboardInterrupt:
        pass
