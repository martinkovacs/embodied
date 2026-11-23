#!/usr/bin/env bash
# Everything on the lab PC goes through this script (LAB_GUIDE.md is the walkthrough).
#   ./lab.sh setup             once: check ROS, install what's ours to install, build ros_ws, run the self-tests
#   ./lab.sh run P07 [...]     a whole session: start the arm, wait until it has woken up, run the experiment
#                              (--fake first: simulated joints and synthetic images; then runner.py's options)
#   ./lab.sh arm [--fake]      only the ROS side, left running (--fake: simulated joints, no arm)
#   ./lab.sh cmd '{"op": "home"}'   send the arm one command and print its answer (ops: arm_node.py's docstring)
#   ./lab.sh torque-off        let the arm be moved by hand (afterwards restart ./lab.sh arm)
#   ./lab.sh usb               install the arm's USB rule (setup does it; needs sudo)
#   ./lab.sh python X.py ...   run one of our Python scripts
set -euo pipefail
SELF=$(readlink -f "$0")        # absolute: the script calls itself after the cd
cd "$(dirname "$SELF")"

export ROS_LOCALHOST_ONLY=1     # everything runs on this PC: never see, or drive, another group's arm on the network
PYTHON=3.13                     # Humble's system Python (3.10) is too old for requirements.txt
PORT=/dev/ttyUSB0
ROBOTIS=(DynamixelSDK dynamixel_interfaces dynamixel_hardware_interface open_manipulator)   # as in docker/Dockerfile
APT_ALWAYS=(curl git python3-colcon-common-extensions ros-humble-rosbridge-suite)
USB_RULE=/etc/udev/rules.d/99-open-manipulator-cdc.rules
APT_ROBOTIS=(ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-gripper-controllers ros-humble-xacro
             ros-humble-robot-state-publisher ros-humble-rosidl-default-generators)
# of the 12 ROBOTIS packages, the 5 the arm's driver needs: the rest (MoveIt config, GUI, teleop) wants MoveIt and Gazebo
OMX_BUILD=(open_manipulator_x_bringup dynamixel_hardware_interface)

die() { echo "ERROR: $*" >&2; exit 1; }

omx_ws() {  # the workspace holding the ROBOTIS packages the arm's driver needs, if there is one
    local w
    for w in "${OMX_WS:-}" "$HOME/omx_ws" "$HOME/manipulator_ws" "$HOME/ros2_ws" "$HOME/colcon_ws" "$HOME/robotis_ws" \
             /opt/omx_ws; do
        if [ -n "$w" ] && [ -f "$w/install/open_manipulator_x_bringup/share/open_manipulator_x_bringup/launch/hardware.launch.py" ] \
            && [ -d "$w/install/dynamixel_hardware_interface" ]; then echo "$w"; return; fi
    done
}

no_venv() {  # run a ROS build with the system Python (3.10, with ROS's own modules), never a virtualenv's
    local p=":$PATH:" v
    for v in "${VIRTUAL_ENV:-}" "$PWD/venv"; do
        if [ -n "$v" ]; then p=${p//:$v\/bin:/:}; fi
    done
    p=${p#:}
    env -u VIRTUAL_ENV -u PYTHONHOME PATH="${p%:}" "$@"
}
# CMake keeps the Python it found in its cache: name it, so an earlier run from inside a venv doesn't stick
SYSTEM_PY=(-DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3)

source_ros() {
    [ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble isn't installed in /opt/ros/humble. Ask the lab staff."
    set +u      # ROS's setup files read unset variables
    source /opt/ros/humble/setup.bash
    local w; w=$(omx_ws)
    if [ -n "$w" ]; then source "$w/install/setup.bash"; fi
    if [ -f ros_ws/install/setup.bash ]; then source ros_ws/install/setup.bash; fi
    set -u
}

cmd_usb() {  # the ROBOTIS quick start's rule: USB latency 16 ms -> 1 ms (the default is too slow), port open to all users
    echo 'KERNEL=="ttyUSB*", DRIVERS=="ftdi_sio", MODE="0666", ATTR{device/latency_timer}="1"' | sudo tee "$USB_RULE" >/dev/null
    sudo udevadm control --reload-rules && sudo udevadm trigger
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
    # first, so the sudo password is asked while someone is still watching
    [ -e "$USB_RULE" ] || cmd_usb || echo "NOTE: the arm's USB rule isn't installed. Ask the lab staff to run: ./lab.sh usb"

    if [ -z "$w" ]; then
        echo "== Building the ROBOTIS packages in $HOME/omx_ws (takes a few minutes)"
        mkdir -p "$HOME/omx_ws/src"
        local r
        for r in "${ROBOTIS[@]}"; do
            [ -d "$HOME/omx_ws/src/$r" ] || git clone --depth 1 -b humble "https://github.com/ROBOTIS-GIT/$r.git" "$HOME/omx_ws/src/$r"
        done
        (cd "$HOME/omx_ws" && no_venv colcon build --packages-up-to "${OMX_BUILD[@]}" \
            --cmake-args -DCMAKE_BUILD_TYPE=Release "${SYSTEM_PY[@]}")
        source_ros
    else
        echo "== ROBOTIS packages: found in $w"
    fi

    echo "== Python environment (venv/)"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # a venv made with the system's python3 (3.10) is too old for our code: make it again
    if [ -x venv/bin/python ] && ! env -u PYTHONPATH venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        echo "venv/ is $(env -u PYTHONPATH venv/bin/python -V 2>&1), too old: making it again with Python $PYTHON"
        rm -rf venv
    fi
    [ -x venv/bin/python ] || env -u PYTHONPATH uv venv --python "$PYTHON" --seed venv
    env -u PYTHONPATH uv pip install --python venv/bin/python -r requirements.txt

    echo "== Building ros_ws"
    (cd ros_ws && no_venv colcon build --symlink-install --cmake-args "${SYSTEM_PY[@]}")

    echo "== Self-tests (about a minute)"
    "$SELF" python runner.py --self-test | tail -1
    "$SELF" python robot_frame.py --self-test
    "$SELF" python speech.py --self-test
    "$SELF" python ros_ws/src/arm_node/arm_node/kinematics.py

    echo "Setup finished."
}

cmd_arm() {
    source_ros
    [ -f ros_ws/install/setup.bash ] || die "run ./lab.sh setup first"
    # two drivers on one serial port, and two arm_nodes running every command
    ! pgrep -f ros2_control_node >/dev/null || die "the ROS side is already running (or still stopping). Stop it with
    Ctrl-C in its terminal, wait a few seconds, then start again."
    local fake=false
    if [ "${1:-}" = "--fake" ]; then
        fake=true
    else
        [ -e "$PORT" ] || die "$PORT doesn't exist: is the arm's USB adapter plugged in and powered?"
        [ -r "$PORT" ] && [ -w "$PORT" ] || die "no permission on $PORT. Run: ./lab.sh usb"
    fi
    ros2 daemon stop >/dev/null 2>&1 || true    # a daemon left over from a killed session answers nothing, and
                                                # the readiness check below would never see the controllers
    trap 'trap - EXIT INT TERM; echo "Stopping the ROS side."; kill 0 2>/dev/null || true' EXIT INT TERM
    echo "The arm will rise by itself and can sweep sideways: NO PIECES near it until it is up."
    ros2 launch arm_node lab.launch.py fake:=$fake &
    local launch=$!
    (
        for _ in $(seq 40); do
            # timeout: without a controller manager the call itself waits 10 s
            if [ "$(timeout 5 ros2 control list_controllers 2>/dev/null | grep -v inactive | grep -c active)" -ge 3 ]; then
                echo "=========================================================="
                echo " ARM SIDE READY. Leave this running, open a second terminal."
                echo " The arm now rises slowly to its home pose: keep clear."
                echo " Before Ctrl-C here: HOLD THE ARM, its motors let go."
                echo "=========================================================="
                if "$SELF" cmd '{"op": "wake"}'; then echo "ARM AWAKE"; else echo "ARM NOT READY: it didn't wake up, see the reason above."; fi
                exit 0
            fi
            # the driver died (motors not answering, torque left on): the spawner would wait for it for ever
            if ! pgrep -f ros2_control_node >/dev/null; then
                echo "ARM NOT READY: the arm's driver (ros2_control_node) stopped. The reason is above 'process has died':
    'Torque is enabled, cannot proceed': hold the arm, ./lab.sh torque-off (or switch its power off and on), start again.
    'There is no status packet' / 'Cannot connect': the arm's power, and the cable from the USB adapter to the motors."
                exit 0
            fi
            sleep 2
        done
        echo "ARM NOT READY: the controllers aren't active, check the errors above.
    'Torque is enabled, cannot proceed': hold the arm, ./lab.sh torque-off (or switch its power off and on), start again.
    'Cannot connect' / no motor found: the arm's power, and the cable from the USB adapter to the motors."
    ) &
    wait "$launch" || true
    echo "ERROR: the ROS side stopped: see the messages above." >&2
    exit 1
}

cmd_run() {
    [ -n "${1:-}" ] || die "usage: ./lab.sh run PARTICIPANT [--fake] [runner.py options]"
    local participant=$1 arm_args=() vision=cameras log
    why() { grep -iE 'error|torque is enabled|ARM NOT READY|"failed"' -A2 "$log" | tail -25 >&2 || tail -20 "$log" >&2; }
    log=logs/arm_side_$(date +%Y%m%d_%H%M%S).log    # one per run: the arm's errors are what a failed session leaves
    shift
    if [ "${1:-}" = "--fake" ]; then arm_args=(--fake); vision=synthetic; shift; fi
    [ -x venv/bin/python ] || die "run ./lab.sh setup first"
    [ -t 0 ] || die "run needs a terminal: before stopping the arm it waits for someone to hold it"
    mkdir -p logs
    # its own process group: when the ROS side dies it kills its group, and the runner must stay up to show why
    setsid "$SELF" arm "${arm_args[@]}" >"$log" 2>&1 &
    ARM_PID=$! ARM_LOG=$log ARM_UP=     # global: the EXIT trap runs after cmd_run has returned
    # stopping the ROS side drops the arm, also when the runner ends by itself (a typo, a crash): wait for a hand on it
    stop_arm() {
        trap - EXIT TERM
        trap '' INT HUP             # a second Ctrl-C or a closed window must not cut this short
        kill -0 "$ARM_PID" 2>/dev/null || return 0
        # the runner, and with it the panel's STOP, is gone: halt whatever the arm still does, its torque stays on
        if [ -n "$ARM_UP" ] && ! timeout 20 "$SELF" cmd '{"op": "stop"}' >/dev/null 2>&1; then
            echo "WARNING: the arm didn't confirm the stop: it may still be moving" >&2
        fi
        if read -rp "HOLD THE ARM, then press Enter: stopping lets its motors go. "; then
            kill -INT -- -"$ARM_PID" 2>/dev/null || true
            wait "$ARM_PID" 2>/dev/null || true
        else                        # no answer (the window was closed): better left held up than dropped
            local msg="The ROS side still runs and holds the arm up. To stop it: hold the arm, then kill -INT -- -$ARM_PID"
            echo "$msg" >>"$ARM_LOG"
            echo "$msg" >&2 || true
        fi
    }
    trap stop_arm EXIT INT TERM HUP
    echo "Starting the arm. It rises to its home pose by itself and can sweep sideways: NO PIECES near it yet."
    echo "(Its messages go to $log)"
    until grep -qE '^ARM (AWAKE|NOT READY)' "$log"; do
        kill -0 "$ARM_PID" 2>/dev/null || { why; die "the arm side stopped: see above, all of it is in $log"; }
        sleep 1
    done
    grep -q '^ARM AWAKE' "$log" || { why; die "the arm isn't ready: see above, all of it is in $log"; }
    ARM_UP=1
    echo "Arm is up: now lay out the pieces. Open http://localhost:8081 and press Start."
    echo "Ctrl-C here stops everything: it asks you to hold the arm first, its motors let go."
    "$SELF" python runner.py --participant "$participant" --vision "$vision" "$@"
}

cmd_cmd() {  # one /arm/cmd, then wait for its done / failed on /arm/status
    [ -n "${1:-}" ] || die "usage: ./lab.sh cmd '{\"op\": \"home\"}'"
    source_ros
    local out; out=$(mktemp)
    PYTHONUNBUFFERED=1 ros2 topic echo /arm/status --field data >"$out" 2>/dev/null &
    local echo_pid=$!
    trap "kill $echo_pid 2>/dev/null || true; rm -f '$out'" EXIT   # expanded now: the locals are gone at exit
    sleep 2     # the echo has to be listening before the command's first answer
    timeout 15 ros2 topic pub --once /arm/cmd std_msgs/msg/String "{data: '$1'}" >/dev/null \
        || die "nothing listens on /arm/cmd: is ./lab.sh arm running?"
    for _ in $(seq 240); do     # 2 minutes: the slowest move is a long wake
        # "id": null = the answer to this command, which carries no id (the runner's commands all do)
        if grep -E '"id": null, "state": "(done|failed)"' "$out"; then
            ! grep -q '"id": null, "state": "failed"' "$out"; return
        fi
        sleep 0.5
    done
    die "no answer on /arm/status in 2 minutes"
}

cmd_torque_off() {
    if [ -t 0 ]; then read -rp "HOLD THE ARM, then press Enter: its motors let go and it falls. "; fi
    if pgrep -f ros2_control_node >/dev/null; then
        source_ros
        timeout 15 ros2 service call /dynamixel_hardware_interface/set_dxl_torque std_srvs/srv/SetBool "{data: false}" \
            || die "the arm's driver didn't answer: switch the arm's power off (still holding it)"
    else
        # no driver (it refuses to start while a motor's torque is on, and a crashed one never turned it off):
        # tell the motors directly, on the port the driver would use
        [ -x venv/bin/python ] || die "run ./lab.sh setup first"
        env -u PYTHONPATH venv/bin/python -c 'import dynamixel_sdk' 2>/dev/null \
            || env -u PYTHONPATH venv/bin/python -m pip install -q dynamixel-sdk==4.1.0
        env -u PYTHONPATH -u LD_LIBRARY_PATH venv/bin/python - "$PORT" <<'EOF' || die "torque is still on: switch the arm's power off (still holding it)"
import sys
from dynamixel_sdk import PacketHandler, PortHandler
port, packets = PortHandler(sys.argv[1]), PacketHandler(2.0)
try:
    if not (port.openPort() and port.setBaudRate(1000000)):
        raise OSError("no answer")
except Exception as e:
    sys.exit(f"can't open {sys.argv[1]} ({e}): is the arm's USB adapter plugged in, and ./lab.sh arm stopped?")
bad = []
for i in (11, 12, 13, 14, 15):      # joints 1-4 and the gripper; X series: Torque Enable is address 64
    res, err = packets.write1ByteTxRx(port, i, 64, 0)
    if res != 0 or err != 0:
        bad.append(f"ID {i}: {packets.getTxRxResult(res) if res else packets.getRxPacketError(err)}")
port.closePort()
if bad:
    sys.exit("these motors didn't take it: " + "; ".join(bad))
EOF
    fi
    echo "Torque off: the arm can be moved by hand. Afterwards restart ./lab.sh arm (don't turn torque back on)."
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
    run) cmd_run "$@" ;;
    arm) cmd_arm "$@" ;;
    cmd) cmd_cmd "$@" ;;
    torque-off) cmd_torque_off ;;
    usb) cmd_usb ;;
    python) cmd_python "$@" ;;
    *) sed -n '2,10p' "$SELF"; exit 2 ;;
esac
