# ROS2 Humble + Gazebo Classic + ROBOTIS OpenMANIPULATOR-X, desktop in the browser (noVNC).
# amd64 on purpose: Gazebo Classic (used by ROBOTIS' humble sim) has no arm64 packages, so on Apple Silicon
# this runs under Rosetta. shortcut: emulated = slower; if too slow, port the URDF to Ignition Fortress (arm64).
FROM tiryoh/ros2-desktop-vnc:humble

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
      libboost-all-dev \
      ros-humble-gazebo-ros-pkgs \
      ros-humble-gazebo-ros2-control \
      ros-humble-ros2-control \
      ros-humble-ros2-controllers \
      ros-humble-gripper-controllers \
      ros-humble-moveit \
      ros-humble-xacro \
      ros-humble-rosbridge-suite \
    && rm -rf /var/lib/apt/lists/*

# ROBOTIS packages, same repos/branches as their own humble Dockerfile
WORKDIR /opt/omx_ws/src
RUN git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git && \
    git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/dynamixel_interfaces.git && \
    git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git && \
    git clone --depth 1 -b humble https://github.com/ROBOTIS-GIT/open_manipulator.git
WORKDIR /opt/omx_ws
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# /etc/bash.bashrc because the base image creates the user's home at container start
RUN echo "source /opt/ros/humble/setup.bash && source /opt/omx_ws/install/setup.bash" >> /etc/bash.bashrc && \
    echo "[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash" >> /etc/bash.bashrc
