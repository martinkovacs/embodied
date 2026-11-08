# embodied — robot trust experiment

A robot arm (OpenMANIPULATOR-X, ROS 2 Humble) sorts coloured pieces (Ø35 × 30 mm; red, blue, green, white)
into colour zones. Per trial a script makes it do the task correctly, make a mistake, or hold the piece up and
ask the participant (*"Is this red?"*). Everything is logged for the trust analysis.

- Architecture diagram: [`docs/architecture.html`](docs/architecture.html)
- One trial as a flowchart: [`docs/trial-flow.html`](docs/trial-flow.html)

## How it fits together

| Part | Where it runs | Files |
|---|---|---|
| Vision: 2 RGB cameras → piece colour + position in table mm | host (Mac / laptop) | `pieces.py`, `stereo_calib.py`, `calib_gui.py` |
| Table → robot coordinates | host | `robot_frame.py`, `calib_data/robot_frame.json` |
| Experiment: trial state machine, scenarios, log, operator panel | host | `runner.py`, `scenarios/`, `layouts/`, `participants.csv` |
| Spoken answers: microphone → Silero VAD → Canary-1B-v2 (transcribe.cpp) | host | `speech.py`, `models/` |
| Arm motions: `/arm/cmd` JSON → inverse kinematics → ROBOTIS controllers | ROS 2 (Docker or the lab Ubuntu PC) | `ros_ws/src/arm_node/` |

The host talks to ROS only through **rosbridge** (websocket, port 9090). That's why the same host code runs
against the simulator, the ROBOTIS fake hardware, or the real arm: only `--host` changes.

## 1. Setup (once)

Needs: Docker Desktop, Python 3.11+ (3.13 and 3.14 tested), git. Works on Apple Silicon (the container is amd64 and
runs under Rosetta, because Gazebo Classic has no arm64 build) and on Linux.

```bash
git clone https://github.com/martinkovacs/embodied.git && cd embodied
python3 -m venv venv && venv/bin/pip install -r requirements.txt
mkdir -p models && cd models                            # speech models (~1.1 GB, gitignored)
curl -LO https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q8_0.gguf
curl -LO https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf
cd ..
cd docker && docker compose up -d --build && cd ..     # first build ~15-30 min
```

Keep the clone out of iCloud-synced folders (`~/Documents`, `~/Desktop` on a Mac): Docker can't read files
that iCloud has evicted.

Build our ROS package inside the container (again after changing `ros_ws/src`):

```bash
docker exec robot-arm bash -c "source /opt/ros/humble/setup.bash && source /opt/omx_ws/install/setup.bash && cd /ws && colcon build --symlink-install"
```

## 2. Try it without any hardware

Self-tests, no ROS needed:

```bash
venv/bin/python runner.py --self-test
venv/bin/python robot_frame.py --self-test
venv/bin/python speech.py --self-test
python3 ros_ws/src/arm_node/arm_node/kinematics.py
```

Full run on the ROBOTIS **fake hardware** (simulated joints, no physics): open a shell in the container with
`docker exec -it robot-arm bash` and start the three processes (each in its own shell, or with `&`):

```bash
source /opt/ros/humble/setup.bash && source /opt/omx_ws/install/setup.bash && source /ws/install/setup.bash
ros2 launch open_manipulator_x_bringup fake.launch.py start_rviz:=false use_sim:=false
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 run arm_node arm_node
```

Then on the host:

```bash
venv/bin/python runner.py --participant P00 --block 1 --vision synthetic
```

and open the operator panel at <http://localhost:8081>. Press **Start**. The pieces come from the synthetic
images in `synthetic/` and are tracked as the arm moves them, because nothing really moves on fake hardware.
`--arm fake` runs the same without ROS at all. `--vision` is required, so a real-arm run can't silently use the
synthetic images; `--vision cameras` refuses a scenario that still points at the synthetic layout.

**The robot's voice** comes from the WAVs in `audio/` (`ask_red.wav`, `ask_blue.wav`, `ask_green.wav`,
`ask_white.wav`, `which.wav`, `done.wav`), played with `pw-play` / `paplay` / `aplay` / `afplay`. They were generated
with [magpie-tts.cpp](https://github.com/mudler/magpie-tts.cpp) at commit `3008ff7` and NVIDIA's Magpie TTS
Multilingual 357M (`magpie-tts-multilingual-357m-q8_0.gguf`, `--lang en --speaker Aria --seed 42`), then trimmed at
-45 dBFS, faded and peak-normalized to -3 dBFS. If a WAV is missing, macOS `say` stands in; elsewhere the panel shows
in red which lines can't be spoken, and every `robot_say` event logs `spoken: false`. A question the robot couldn't speak stops the trial as unplanned (an
answer to a question nobody heard isn't data); `--no-voice` runs silent on purpose (development, or the
experimenter reads the questions) and is logged. Trim the silence at the end of each WAV: reaction times are
measured from the end of playback. `--lang` only switches speech recognition; the recordings stay as recorded.

**Spoken answers.** The panel listens on the default microphone (`--mic N` picks another, `--lang en|hu`,
`--no-speech` for buttons only). It is muted while the robot speaks, and it records only while a session runs:
every transcript is logged with its audio. An answer that clearly matches what the robot asked ("yes",
"nope", "no, it's blue", "igen", "kék", …) and began after the question ended answers it. Anything unclear goes
to the experimenter's buttons: hedges ("I don't know", "nem tudom", "maybe"), negated words ("not red"),
"yes" with a different colour, speech that began during the question. Switch *auto-answer* off to confirm
every recognised answer by hand.

`rt_s` runs from the end of the question to the **start of the participant's speech** (the first 32 ms VAD
frame judged to be speech, backdated over the quieter frames before it; the saved WAV with `t_audio` lets you
refine it offline). When the experimenter clicks the answer, `rt_s` still comes from the participant's speech
if there was a transcribed utterance after the question; otherwise the log has `rt_s: null` and only
`rt_click_s` (`rt_from` says which). The speech engine runs in its own process: if it crashes or the
microphone goes quiet, the panel says so and the buttons keep working. The first start on a new machine
compiles GPU shaders (a few minutes, cached afterwards). Check the microphone with
`venv/bin/python speech.py --test`.

**Watching the arm** (optional): open the container desktop at <http://localhost:6080>. If it's black, the
desktop's screen lock is on. Turn it off and start RViz as root, so it sees the same ROS data as the nodes:

```bash
docker exec -u ubuntu robot-arm bash -c 'pkill -f "[m]ate-screensaver"; export $(dbus-launch | head -1); gsettings set org.mate.screensaver lock-enabled false; gsettings set org.mate.screensaver idle-activation-enabled false'
docker exec -u ubuntu -e DISPLAY=:1 robot-arm xhost +local:
docker exec -d -e DISPLAY=:1 robot-arm bash -c 'source /opt/ros/humble/setup.bash && source /opt/omx_ws/install/setup.bash && rviz2 -d /opt/omx_ws/src/open_manipulator/open_manipulator_x_bringup/rviz/open_manipulator_x.rviz'
```

## 3. The experiment files

- `participants.csv`: each participant's conditions in running order (`--block 2` = their second one).
- `scenarios/<condition>.yaml`: the slot order (the same in every condition) and which trials are `mistake`
  (with `put_to`) or `ask`. Unlisted trials are `correct`. **The trial numbers in there are placeholders.**
- `layouts/*.yaml`: the printed table template: piece slots and zone spots in table mm, plus the robot frame
  file. On start the runner checks that the arm reaches every slot and spot.
- `layouts/*.yaml`: on start the runner also checks that the arm travels between every slot, spot, home and
  show clear of the pieces, that zone spots are ≥ 40 mm apart, and that each zone has room for the worst
  case (an asked participant may name any colour).
- `logs/<participant>_<condition>.jsonl`: one event per line, each with the run id and attempt
  (`session_start` with the run settings, `trial_start`, `vision`, `arm_cmd`, `spot` (zone spot reserved /
  released), `robot_say` with `spoken`, `participant_answer` with `rt_s` / `rt_from` / `source`, `speech`
  (every transcript), `unplanned`, `operator_override`, `stop` / `stop_confirm` / `resume`, `trial_end`, …). Camera images and
  speech audio go to `logs/<participant>_<condition>/<run>/`, so a rerun never overwrites them. `logs/` is
  gitignored: it's participant data.
- If the runner quits or crashes mid-block, starting it again with the same participant and block finds the
  unfinished session and offers **Resume** at the interrupted trial, with the zone spots rebuilt from the log.

If something unplanned happens (piece not in its slot, arm error, failed check after placing), the trial stops
and the panel asks the experimenter to **retry / skip / abort**, or **accept** when the arm did place the piece
but the camera missed it. A retry after a confirmed place only goes home and looks again; while a piece is in
the gripper the panel offers **open gripper** (hold the piece). A skip logs how far the trial got (`step`,
`moving`), so you can tell what the participant saw. The robot never asks an unscripted question,
because that would contaminate the conditions. **STOP** halts the arm (and the robot's voice) immediately;
**Resume** asks the same retry / skip / abort about the interrupted trial and carries on with the session.

## 4. In the lab (real arm, real cameras)

The arm is plugged into the lab's Ubuntu PC (U2D2 adapter, `/dev/ttyUSB0`), which runs ROS. The laptop with
the cameras runs everything else and reaches the PC over the network.

On the lab PC (with ROS 2 Humble + the ROBOTIS packages + this repo's `ros_ws` built):

```bash
ros2 launch open_manipulator_x_bringup hardware.launch.py      # port_name:=/dev/ttyUSB0 is the default
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 run arm_node arm_node
```

On the laptop: `venv/bin/python runner.py --participant P07 --block 1 --vision cameras --host <lab-pc-ip>`.

If the link to rosbridge drops, STOP can't reach the arm: the panel says so in red, and the stop is logged with
`sent: false`. Keep the torque-off / power switch in reach. rosbridge has no authentication: anyone who reaches
port 9090 can move the arm. Connect the laptop and the
lab PC directly (cable) or firewall 9090 to the laptop's address. The operator panel listens on 127.0.0.1
only; `--panel-host 0.0.0.0` exposes it to the network.

### Calibrate before any participant, in this order

1. **Arm dimensions.** `ros_ws/src/arm_node/arm_node/kinematics.py` holds the link lengths of the stock
   OpenMANIPULATOR-X. A differently sized (e.g. 3D-printed) arm needs its own lengths there, measured
   joint axis to joint axis, and `TOOL` measured from joint 4 to the fingertips. Rerun the kinematics
   self-test afterwards.
2. **Cameras.** `venv/bin/python calib_gui.py`: turn off test mode, capture 15-25 checkerboard pairs
   (9 × 6 inner corners, 25 mm squares) at different angles. Accept if the RMS error is < 0.5 px.
3. **Table pose.** In the same GUI: lay the checkerboard flat on the table and capture the table pose. This
   defines the table frame every position is measured in. Once that's done, don't move the cameras or the table.
4. **Colours.** Check the detection tab under the lab lights. Tune the HSV ranges in `pieces.py` if a colour
   is missed. White pieces need a dark mat under them.
5. **Table → robot.** `venv/bin/python robot_frame.py --host <lab-pc-ip>`: close the gripper and touch the
   fingertips to 4-6 known table points spread over the work area (torque off, move by hand). Then
   `--check` on new points: accept if every error is < 5 mm. Redo after re-capturing the table pose. Each
   pick / place then carries the table's height at that spot, so a slightly tilted table is fine. The arm
   goes down and up in 5 mm IK steps (truly vertical), so 40 mm between pieces leaves room for the fingers;
   measure the open fingers on the real gripper and raise `SPACING_MM` in `runner.py` if they need more.
6. **Layout.** Write `layouts/lab.yaml` from the printed template (slot and zone positions in table mm) and
   point the scenarios at it. The runner refuses a layout with a spot the arm can't reach.
7. **Microphone and voice.** Place the microphone near the participant, away from the arm's motors and the
   speaker. `venv/bin/python speech.py --test --lang <participant's language>`: the level should move when
   someone talks, and yes / no / the colours should come out right. Record the six WAVs (see section 2).
8. **Dry runs.** 50 pick-and-place moves with no unplanned errors (tune `EMPTY_MARGIN` in `arm_node.py`
   if a good grasp is reported as "closed on nothing"), then a pilot of every condition.
   Check the log and timing.

Safety: keep a torque-off / power switch in reach, and keep speeds low (`MAX_SPEED` in `arm_node.py`).
