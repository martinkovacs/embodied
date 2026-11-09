#!/usr/bin/env bash
# Everything on the lab PC goes through this script (LAB_GUIDE.md is the walkthrough).
#   ./lab.sh setup             check ROS, install what's ours to install, build ros_ws, run the self-tests
#   ./lab.sh arm [--fake]      start the ROS side and leave it running (--fake: simulated joints, no arm)
#   ./lab.sh torque-off        let the arm be moved by hand (afterwards restart ./lab.sh arm)
#   ./lab.sh python X.py ...   run one of our Python scripts
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

export ROS_LOCALHOST_ONLY=1     # everything runs on this PC: never see, or drive, another group's arm on the network
PYTHON=3.13                     # Humble's system Python (3.10) is too old for requirements.txt
PORT=/dev/ttyUSB0
ROBOTIS=(DynamixelSDK dynamixel_interfaces dynamixel_hardware_interface open_manipulator)   # as in docker/Dockerfile
APT_ALWAYS=(curl git ros-humble-rosbridge-suite)
APT_ROBOTIS=(libboost-all-dev ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-ros2-control
             ros-humble-ros2-controllers ros-humble-gripper-controllers ros-humble-moveit ros-humble-xacro)

die() { echo "ERROR: $*" >&2; exit 1; }

omx_ws() {  # the workspace holding the ROBOTIS packages, if there is one
    local w
    for w in "${OMX_WS:-}" "$HOME/omx_ws" "$HOME/ros2_ws" "$HOME/colcon_ws" "$HOME/robotis_ws" /opt/omx_ws; do
        if [ -n "$w" ] && [ -d "$w/install/open_manipulator_x_bringup" ]; then echo "$w"; return; fi
    done
}

source_ros() {
    [ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble isn't installed in /opt/ros/humble. Ask the lab staff."
    set +u      # ROS's setup files read unset variables
    source /opt/ros/humble/setup.bash
    local w; w=$(omx_ws)
    if [ -n "$w" ]; then source "$w/install/setup.bash"; fi
    if [ -f ros_ws/install/setup.bash ]; then source ros_ws/install/setup.bash; fi
    set -u
}

missing_apt() { local p; for p in "$@"; do dpkg -s "$p" >/dev/null 2>&1 || echo -n "$p "; done; }

cmd_setup() {
    source_ros
    local need w
    w=$(omx_ws)
    need=$(missing_apt "${APT_ALWAYS[@]}")
    if [ -z "$w" ]; then need+=$(missing_apt "${APT_ROBOTIS[@]}"); fi
    [ -z "$need" ] || die "these packages are missing. Run this (or ask the lab staff), then run setup again:
    sudo apt-get update && sudo apt-get install -y $need"

    if [ -z "$w" ]; then
        echo "== Building the ROBOTIS packages in $HOME/omx_ws (takes a few minutes)"
        mkdir -p "$HOME/omx_ws/src"
        local r
        for r in "${ROBOTIS[@]}"; do
            [ -d "$HOME/omx_ws/src/$r" ] || git clone --depth 1 -b humble "https://github.com/ROBOTIS-GIT/$r.git" "$HOME/omx_ws/src/$r"
        done
        (cd "$HOME/omx_ws" && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release)
        source_ros
    else
        echo "== ROBOTIS packages: found in $w"
    fi

    echo "== Python environment (venv/)"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -x venv/bin/python ] || env -u PYTHONPATH uv venv --python "$PYTHON" --seed venv
    env -u PYTHONPATH uv pip install --python venv/bin/python -r requirements.txt

    echo "== Building ros_ws"
    (cd ros_ws && colcon build --symlink-install)

    echo "== Self-tests (about a minute)"
    "$0" python runner.py --self-test | tail -1
    "$0" python robot_frame.py --self-test
    "$0" python speech.py --self-test
    "$0" python ros_ws/src/arm_node/arm_node/kinematics.py

    id -nG | grep -qwE 'dialout|uucp' || echo "NOTE: you aren't in the dialout group. If ./lab.sh arm can't open $PORT:
    sudo chmod 666 $PORT"
    echo "Setup finished."
}

cmd_arm() {
    source_ros
    [ -f ros_ws/install/setup.bash ] || die "run ./lab.sh setup first"
    local launch=(hardware.launch.py)
    if [ "${1:-}" = "--fake" ]; then
        launch=(fake.launch.py start_rviz:=false use_sim:=false)
    else
        [ -e "$PORT" ] || die "$PORT doesn't exist: is the arm's USB adapter plugged in and powered?"
        [ -r "$PORT" ] && [ -w "$PORT" ] || die "no permission on $PORT. Run: sudo chmod 666 $PORT"
    fi
    trap 'trap - EXIT INT TERM; echo "Stopping the ROS side."; kill 0 2>/dev/null || true' EXIT INT TERM
    ros2 launch open_manipulator_x_bringup "${launch[@]}" &
    local bringup=$!
    ros2 launch rosbridge_server rosbridge_websocket_launch.xml address:=127.0.0.1 &
    local bridge=$!
    ros2 run arm_node arm_node &
    local node=$!
    (
        for _ in $(seq 40); do
            if [ "$(ros2 control list_controllers 2>/dev/null | grep -v inactive | grep -c active)" -ge 3 ]; then
                echo "=========================================================="
                echo " ARM SIDE READY. Leave this running, open a second terminal."
                echo "=========================================================="
                exit 0
            fi
            sleep 2
        done
        echo "WARNING: the controllers aren't active after 80 s: check the errors above."
    ) &
    wait -n "$bringup" "$bridge" "$node" || true
    echo "ERROR: one of the ROS processes stopped: see the messages above." >&2
    exit 1
}

cmd_torque_off() {
    source_ros
    ros2 service call /dynamixel_hardware_interface/set_dxl_torque std_srvs/srv/SetBool "{data: false}"
    echo "Torque off: HOLD THE ARM, it can now fall. Afterwards restart ./lab.sh arm (don't turn torque back on)."
}

cmd_python() {
    [ -x venv/bin/python ] || die "run ./lab.sh setup first"
    # a terminal that sourced ROS carries Python 3.10 packages our Python must not import
    exec env -u PYTHONPATH -u LD_LIBRARY_PATH venv/bin/python "$@"
}

cmd="${1:-}"
[ $# -eq 0 ] || shift
case "$cmd" in
    setup) cmd_setup ;;
    arm) cmd_arm "$@" ;;
    torque-off) cmd_torque_off ;;
    python) cmd_python "$@" ;;
    *) sed -n '2,6p' "$0"; exit 2 ;;
esac
