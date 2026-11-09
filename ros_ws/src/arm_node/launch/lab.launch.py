"""The whole ROS side in one launch: the ROBOTIS arm driver, rosbridge (for runner.py) and arm_node.

    ros2 launch arm_node lab.launch.py              # the real arm on /dev/ttyUSB0
    ros2 launch arm_node lab.launch.py fake:=true   # simulated joints, no arm

./lab.sh arm wraps this (and checks the USB port first).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fake = LaunchConfiguration("fake")
    bringup = FindPackageShare("open_manipulator_x_bringup")
    return LaunchDescription([
        DeclareLaunchArgument("fake", default_value="false", description="simulated joints instead of the arm"),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(PathJoinSubstitution([bringup, "launch", "hardware.launch.py"])),
            launch_arguments={"start_rviz": "false"}.items(),
            condition=UnlessCondition(fake)),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(PathJoinSubstitution([bringup, "launch", "fake.launch.py"])),
            # without use_sim:=false the fake bringup waits for Gazebo and starts no controller manager
            launch_arguments={"start_rviz": "false", "use_sim": "false"}.items(),
            condition=IfCondition(fake)),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("rosbridge_server"), "launch", "rosbridge_websocket_launch.xml"])),
            # this machine only: rosbridge has no authentication, anyone who reaches it drives the arm
            launch_arguments={"address": "127.0.0.1"}.items()),
        Node(package="arm_node", executable="arm_node", output="screen"),
    ])
