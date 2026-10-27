"""Scripted OpenMANIPULATOR-X motions behind /arm/cmd (architecture.md section 3).

/arm/cmd     std_msgs/String JSON  {"id": 7, "op": "pick", "x": 120, "y": -40}   x, y: robot base frame (link1), mm
             op: pick | place (x, y) | show | home | open | stop
/arm/status  std_msgs/String JSON  {"id": 7, "state": "started" | "done" | "failed", "reason": "..."}

One command at a time: another one arriving meanwhile fails with "busy". stop answers "done" at once and cancels
the running motion: the arm holds where it is and the running command fails with "stopped".

Run after the ROBOTIS bringup (hardware.launch.py or gazebo.launch.py):  ros2 run arm_node arm_node
"""
import json

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

from arm_node.kinematics import ik

JOINTS = ["joint1", "joint2", "joint3", "joint4"]
# Tune in the lab. Heights are fingertip z in the robot frame: the table is at z = 0 if the arm stands on it,
# -8 if it stands on the ROBOTIS base plate (robot_frame.py prints where the table is).
GRASP_Z = 5.0                   # fingertips just above the table, fingers around most of the 30 mm piece
PLACE_Z = GRASP_Z + 3.0         # release a little above where it was picked up
LIFT = 60.0                     # approach and retreat this much above GRASP_Z, clear of the 30 mm pieces
HOME = [0.0, -1.0, 0.7, 0.3]    # ROBOTIS "home" (SRDF); check that it hides no piece from either camera
SHOW = [0.0, -0.3, -0.3, 0.6]   # gripper level, held out along +x at ~28 cm; turn joint1 toward the participant
OPEN, CLOSED = 0.019, -0.01     # gripper_left_joint limits, m; closing on a piece just stalls on it
MAX_SPEED = 0.6                 # rad/s, for the joint that moves most
MIN_MOVE_S = 1.0


class ArmNode(Node):
    def __init__(self):
        super().__init__("arm_node")
        self.arm = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self.gripper = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd")
        self.status = self.create_publisher(String, "/arm/status", 10)
        self.q = None                   # current arm joints, from /joint_states
        self.goal = None                # running controller goal, cancelled by stop
        self.busy = self.stopped = False
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        # reentrant: a stop has to get through while a command is still awaiting the controllers
        self.create_subscription(String, "/arm/cmd", self.on_cmd, 10, callback_group=ReentrantCallbackGroup())

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in JOINTS):
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
        except Exception as e:  # bad command, out of reach, controller error, stopped
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
            await self.move(HOME)
        elif op == "show":
            await self.move(SHOW)
        elif op == "open":
            await self.grip(OPEN)
        elif op in ("pick", "place"):
            x, y = float(cmd["x"]), float(cmd["y"])
            # both poses solved before anything moves, so an unreachable spot fails cleanly
            above, down = ik(x, y, GRASP_Z + LIFT), ik(x, y, GRASP_Z if op == "pick" else PLACE_Z)
            if op == "pick":
                await self.grip(OPEN)
            await self.move(above)
            await self.move(down)
            await self.grip(CLOSED if op == "pick" else OPEN)
            await self.move(above)
        else:
            raise ValueError(f"unknown op {op!r}")

    async def move(self, q):
        if self.q is None:
            raise RuntimeError("no /joint_states yet")
        t = max(MIN_MOVE_S, max(abs(a - b) for a, b in zip(q, self.q)) / MAX_SPEED)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        # zero end velocity makes the controller fit a cubic: smooth start and stop
        goal.trajectory.points = [JointTrajectoryPoint(positions=list(q), velocities=[0.0] * len(JOINTS),
                                                       time_from_start=Duration(seconds=t).to_msg())]
        res = await self.send(self.arm, goal)
        if res.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"arm controller: status {res.status} {res.result.error_string}")

    async def grip(self, position):
        goal = GripperCommand.Goal()
        goal.command.position = position
        await self.send(self.gripper, goal)  # status ignored: closing on a piece ends in a stall abort

    async def send(self, client, goal):
        if self.stopped:
            raise RuntimeError("stopped")
        if not client.server_is_ready():
            raise RuntimeError("controller not running (ros2 control list_controllers)")
        handle = await client.send_goal_async(goal)
        if not handle.accepted:
            raise RuntimeError("goal rejected")
        self.goal = handle
        if self.stopped:  # stop came in while the goal was being sent
            handle.cancel_goal_async()
        res = await handle.get_result_async()
        self.goal = None
        if self.stopped:
            raise RuntimeError("stopped")
        return res


def main():
    rclpy.init()
    try:
        rclpy.spin(ArmNode())
    except KeyboardInterrupt:
        pass
