"""Is the lab PC ready? Each check prints OK, WARN or FAIL with what to do. Moves nothing.

--hover then tests the whole chain from camera to arm: the arm goes home, the cameras find the pieces, and the
fingertips stop above each one (approach height, slowly). Only your eyes can tell if that's the right place. It's
the one check that catches a table origin on the wrong end of the board, because such a fit is self-consistent.

Usage (from the repo root):
    ./lab.sh python lab_check.py [--fake]     # --fake: no USB adapter expected (./lab.sh arm --fake)
    ./lab.sh python lab_check.py --sound      # also plays one of the robot's lines
    ./lab.sh python lab_check.py --hover
"""
import argparse
import asyncio
import csv
import os
import shutil
import subprocess
import time
from math import isfinite

import numpy as np

import cameras
import robot_frame
import runner
from arm_node.kinematics import CLEAR, fk  # noqa: E402  via robot_frame
from stereo_calib import StereoCalib

PORT = "/dev/ttyUSB0"
CONTROLLERS = ("arm_controller", "gripper_controller", "joint_state_broadcaster")
FRAME_PATH = runner.ROOT / "calib_data" / "robot_frame.json"


class Fail(Exception):
    pass


class Warn(Exception):
    pass


def step(name, results, fn):
    """Run one check: fn returns the OK detail, or raises Fail / Warn with what's wrong and what to do."""
    try:
        results.append(("OK", name, fn() or ""))
    except Warn as e:
        results.append(("WARN", name, str(e)))
    except Fail as e:
        results.append(("FAIL", name, str(e)))
    except Exception as e:
        results.append(("FAIL", name, f"{type(e).__name__}: {e}"))
    status, _, detail = results[-1]
    print(f"[{status:4s}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check_port():
    if not os.path.exists(PORT):
        raise Fail(f"{PORT} doesn't exist: plug in and power the arm's USB adapter")
    if not os.access(PORT, os.R_OK | os.W_OK):
        raise Fail(f"no permission: sudo chmod 666 {PORT}")


def connect(host):
    try:
        return runner.RosArm(host)
    except Exception:
        raise Fail("nothing answers on port 9090: start ./lab.sh arm in another terminal") from None


def check_controllers(arm):
    import roslibpy
    svc = roslibpy.Service(arm.ros, "/controller_manager/list_controllers", "controller_manager_msgs/ListControllers")
    res = svc.call(roslibpy.ServiceRequest({}), timeout=8)
    state = {c["name"]: c["state"] for c in res["controller"]}
    bad = [c for c in CONTROLLERS if state.get(c) != "active"]
    if bad:
        raise Fail(f"not active: {', '.join(bad)}. Look at the errors in the ./lab.sh arm terminal")
    return ", ".join(CONTROLLERS)


def read_joints(arm):
    """The arm's four joints from one fresh /joint_states."""
    import roslibpy
    got = {}
    topic = roslibpy.Topic(arm.ros, "/joint_states", "sensor_msgs/JointState")
    topic.subscribe(lambda m: got.update(zip(m["name"], m["position"])))
    end = time.monotonic() + 4
    while time.monotonic() < end and not all(j in got for j in robot_frame.JOINTS):
        time.sleep(0.05)
    topic.unsubscribe()
    if not all(j in got and isfinite(got[j]) for j in robot_frame.JOINTS):
        raise Fail("no valid /joint_states: is the arm powered on?")
    return [got[j] for j in robot_frame.JOINTS]


def check_pose(arm):
    q = read_joints(arm)
    z = fk(q)[2]
    table_z = float(robot_frame.load(FRAME_PATH)[2, 3]) if FRAME_PATH.exists() else 0.0
    if not z >= table_z + CLEAR:
        raise Warn(f"fingertips are {z - table_z:.0f} mm over the table: `home` may refuse. Stop ./lab.sh arm, lift the "
                   "arm by hand, start it again")
    return f"fingertips {z - table_z:.0f} mm over the table"


def check_node(arm):
    """A stop with nothing running moves nothing and proves commands get to arm_node and answers come back."""
    async def ask():
        for _ in range(2):      # the very first message on a new topic can go missing
            try:
                return await asyncio.wait_for(arm.do("stop"), 4)
            except TimeoutError:
                pass
        raise Fail("no answer from arm_node: look for errors in the ./lab.sh arm terminal")
    time.sleep(0.5)             # let the /arm/status subscription settle
    asyncio.run(ask())


def check_calibration():
    if not cameras.CALIB_PATH.exists():
        raise Fail("no real calibration yet: do the calib_gui.py step")
    c = StereoCalib.load(cameras.CALIB_PATH)
    if c.T_left_table is None:
        raise Fail("no table pose: capture it in calib_gui.py")
    if c.rms >= 1.0:
        raise Fail(f"RMS {c.rms:.2f} px is too high: delete the worst pairs in calib_gui.py and recalibrate")
    if c.rms >= 0.5:
        raise Warn(f"RMS {c.rms:.2f} px, aim for under 0.5")
    return f"RMS {c.rms:.2f} px"


def check_cameras():
    st = cameras.load_settings()
    if st["test_mode"]:
        raise Fail("calib_gui.py is in test mode: switch it off in Setup and press Connect")
    if st["one_camera"]:
        raise Fail("calib_gui.py is in one-camera mode: switch it off in Setup and press Connect")
    have = cameras.list_cameras()
    for side in ("left", "right"):
        if int(st[side]) not in [i for i, _ in have]:
            raise Fail(f"{side} camera is {st[side]}, but the cameras found are {have}: fix it in calib_gui.py Setup")
    if st["left"] == st["right"]:
        raise Fail("left and right are the same camera: fix it in calib_gui.py Setup")
    try:
        vision = runner.Vision()
    except RuntimeError as e:
        raise Fail(f"{e} (close calib_gui.py: only one program can use a camera)") from None
    try:
        left, right = cameras.fresh_pair(*vision.sources)
        if left is None or right is None:
            raise Fail("a camera gives no image: both on the same USB port or hub? Move one, or close calib_gui.py")
        for side, img in (("left", left), ("right", right)):
            if img.mean() < 8:
                raise Fail(f"the {side} image is black: lens cap, or no light?")
        found, _ = vision.detect()
    finally:
        for s in vision.sources:
            s.release()
    colors = {c: sum(p[0] == c for p in found) for c in runner.COLORS}
    return f"{left.shape[1]}x{left.shape[0]}, pieces seen by both: " + " ".join(f"{c} {n}" for c, n in colors.items())


def check_frame():
    if not FRAME_PATH.exists():
        raise Fail("not fitted yet: do the robot_frame.py step")
    T = robot_frame.load(FRAME_PATH)
    if T[2, 2] < 0.95:
        raise Warn(f"the table looks tilted by {np.degrees(np.arccos(T[2, 2])):.0f} degrees against the arm: refit?")
    return f"table at robot z = {T[2, 3]:.1f} mm"


def check_layout():
    if not (runner.ROOT / runner.LAB_LAYOUT).exists():
        raise Fail(f"no {runner.LAB_LAYOUT}: do the lab_layout.py step")
    with open(runner.ROOT / "participants.csv") as f:
        conditions = {c for r in csv.DictReader(f) for c in r["conditions"].split()}
    for c in sorted(conditions):
        try:
            runner.load_scenario(c, runner.LAB_LAYOUT)
        except AssertionError as e:
            raise Fail(f"{c}: {e}") from None
    return f"fits {', '.join(sorted(conditions))}"


def check_voice(play):
    lines = [("ask", c) for c in runner.COLORS] + [("which", None), ("done", None)]
    voice = runner.Voice(lines)
    if problems := voice.problems():
        raise Fail(f"{problems[0]}. Install alsa-utils or pipewire")
    if play:
        argv, _, _ = voice.how("ask", "red")
        if subprocess.run(argv, capture_output=True, timeout=10).returncode:
            raise Fail(f"{argv[0]} failed: check the sound output")
    return "you should have heard \"Is this red?\"" if play else f"plays through {shutil.which(voice.how('done')[0][0])}"


def check_speech():
    import speech
    missing = [p.name for p in (speech.MODEL, speech.VAD_MODEL) if not p.exists()]
    if missing:
        raise Warn(f"{', '.join(missing)} not downloaded (README section 1): use --no-speech for now")
    try:
        import sounddevice as sd
        return f"microphone: {sd.query_devices(kind='input')['name']}"
    except Exception as e:      # no libportaudio2, or no input device
        raise Warn(f"no microphone ({e}): use --no-speech for now") from None


async def hover(host):
    vision = runner.Vision()
    T = robot_frame.load(FRAME_PATH)
    arm = runner.RosArm(host, air=True)
    await asyncio.sleep(0.5)
    wrong = right = 0
    try:
        print("Going home. Keep a hand on the power switch; Ctrl-C stops the arm.", flush=True)
        await arm.do("home")
        found, _ = await asyncio.to_thread(vision.detect)
        if not found:
            print("The cameras see no pieces: put some on the table first.")
            return 1
        for color, x, y in found:
            print(f"\n{color} piece, seen at table ({x:.0f}, {y:.0f}) mm", flush=True)
            try:
                await arm.do("hover", *robot_frame.to_robot(T, (x, y, 0.0)))
            except RuntimeError as e:
                print(f"  skipped: {e}")
                continue
            answer = await asyncio.to_thread(input, "  Are the fingertips above it? [Enter = yes, n = no] ")
            wrong += answer.strip().lower().startswith("n")
            right += not answer.strip().lower().startswith("n")
        await arm.do("home")
    except asyncio.CancelledError:
        arm.stop()
        await asyncio.shield(asyncio.sleep(0.5))
        raise
    finally:
        for s in vision.sources:
            s.release()
    print(f"\n{right} right, {wrong} wrong.")
    if wrong:
        print("Far off (the arm goes somewhere else entirely): the table origin was typed at the wrong end of the board.\n"
              "Redo robot_frame.py with x and y measured from the corner the axes in calib_gui.py mark.\n"
              "A few mm off, the same way everywhere: redo robot_frame.py with more care, then --check.")
    return 1 if wrong else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost", help="rosbridge host (port 9090)")
    ap.add_argument("--fake", action="store_true", help="no USB adapter expected (./lab.sh arm --fake)")
    ap.add_argument("--sound", action="store_true", help="play one of the robot's lines")
    ap.add_argument("--hover", action="store_true", help="the arm hovers over each piece the cameras see")
    args = ap.parse_args()
    if args.hover:
        try:
            raise SystemExit(asyncio.run(hover(args.host)))
        except KeyboardInterrupt:
            raise SystemExit("Stopped.") from None

    results, arm = [], []
    if not args.fake:
        step("Arm USB adapter", results, check_port)

    def rosbridge():
        arm.append(connect(args.host))
    step("ROS connection", results, rosbridge)
    if arm:
        step("Arm controllers", results, lambda: check_controllers(arm[0]))
        step("Arm pose", results, lambda: check_pose(arm[0]))
        step("arm_node", results, lambda: check_node(arm[0]))
    step("Camera calibration", results, check_calibration)
    step("Cameras", results, check_cameras)
    step("Table-to-robot fit", results, check_frame)
    step("Layout", results, check_layout)
    step("Robot voice", results, lambda: check_voice(args.sound))
    step("Spoken answers", results, check_speech)
    fails, warns = (sum(r[0] == s for r in results) for s in ("FAIL", "WARN"))
    print(f"\n{fails} to fix, {warns} warnings." if fails or warns else "\nAll good.")
    if arm:
        arm[0].ros.terminate()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
