# A session on the lab PC, step by step

Do these in order. Stop at the first step that fails: it prints what to do.
Keep one hand near the arm's **power switch** whenever it can move.

You need **two terminal windows**:

- **Terminal 1 (T1)** runs the arm side, the whole session. Nothing else goes in it.
- **Terminal 2 (T2)** runs everything else, one program at a time: only one program can use the cameras.

Everything here is for the lab PC's **Ubuntu 22.04**. Open a terminal with **Ctrl+Alt+T**, a second window with
**Ctrl+Shift+N** in it. Paste into a terminal with **Ctrl+Shift+V** (Ctrl+V doesn't paste there). In both, first
`cd ~/embodied`.

## 0. Get the latest code (T2, every session)

```bash
cd ~/embodied
git pull --autostash
```

`--autostash` keeps this PC's own settings (`calib_data/settings.json`) across the pull. If the list of changed
files it prints has anything under `ros_ws/`, also run `./lab.sh setup` once (a few minutes).
If it stops with *divergent branches*, someone committed on the lab PC itself: don't force anything, ask
before going on.

**First time on a new PC only:**

```bash
git clone https://github.com/martinkovacs/embodied.git ~/embodied
cd ~/embodied
./lab.sh setup
```

It asks for the sudo password once, to install the arm's USB rule, and takes a few minutes. If it says packages are
missing, run the command it prints (or ask the lab staff), then run `./lab.sh setup` again. It should end with
`Setup finished.`

## 1. Place the cameras (no software)

- **High:** 50 to 65 cm above the table, 35 to 45 cm (horizontally) from the middle of the work area, looking down
  at it at 50 to 60 degrees. The work area is the ring 100 to 210 mm around the arm's base.
- **Both on the same side of the table, 60 to 90 degrees apart** around the work area (one from the front, the
  other from the side), on the side away from the participant. Not facing each other: low cameras facing each
  other see mostly the pieces' sides, and then the two cameras disagree. On one side, both cameras can see the
  board you hold up in step 4 at the same time, which calibrates them against each other.
- Each camera sees the whole work area, without the arm or the participant in front of it. A piece's top should be
  at least 40 px wide in the camera picture.
- Tighten them. **From now on the cameras must not move.** If one moves, redo from step 4.

## 2. Start the arm (T1)

1. Plug in the arm and turn on its power. Plug in both cameras too.
2. Clear the table around the arm: no pieces, cables or hands within its reach.
3. In **T1**:

```bash
./lab.sh arm
```

Wait for `ARM SIDE READY`. The arm then **rises slowly to its home pose by itself**, from wherever it was lying:
that is the sign that it works. If it stays down, the line above the warning says why.
**Leave T1 running.**

Two rules for every start and stop of the arm side:

- **Starting:** no pieces near the arm. It may sweep sideways while it gets up. Pieces go on the table after it is up.
- **Stopping (Ctrl-C):** hold the arm first, or let it rest on the table. Its motors let go and it falls.
  Never just close the terminal: the motors then stay on, and the next start refuses with
  `Torque is enabled, cannot proceed` until the arm's power is switched off and on.

## 3. Check that it moves (T2)

```bash
./lab.sh python lab_check.py
```

Before calibrating, many lines say FAIL. These four must be OK or WARN:
*Arm USB adapter, ROS connection, Arm controllers, arm_node*. Then:

```bash
./lab.sh cmd '{"op": "open"}'
./lab.sh cmd '{"op": "home"}'
```

`"state": "done"` means it did it; `"failed"` comes with the reason.

## 4. Calibrate the cameras (T2)

```bash
./lab.sh python calib_app.py
```

Open <http://localhost:8080>.

1. **Setup tab:** press **Find cameras**, pick the two Basler cameras as *Left camera* and *Right camera* (the ⇄
   button swaps them). Under *Checkerboard*:
   - *Inner corners*: 9 and 6.
   - *Square size*: **measure it**. Measure 5 squares in a row with a ruler and divide by 5. A few % off here makes
     every distance a few % off.
   - *Board thickness*: **measure it** (a printed board on cardboard can be 3 to 5 mm).

   Press **Apply board settings**, then **Connect**.
2. **Calibration tab:** press **Delete all**. Captures from earlier sessions are from other camera positions.
3. **Lens views:** hold the board still at different angles (20 to 45° tilted, different ways) and in different
   places in front of the cameras, and press **Capture view** each time. One camera seeing it is enough, but hold
   it so **both** see it as often as you can: those views also fix where one camera is from the other. Go on until
   the counter next to the button says **6/6 ✓ for both cameras** (8 to 12 is better). The result under the list
   then shows `RMS` per camera: it should be **about 1 px or less**. Delete a capture shown in red (over 1 px, its
   ✕), or one where the board was blurred. Fewer than 6 views: the header says *rough* and positions can be
   centimetres off.
4. **The table capture:** lay the board **flat** on the table in front of the arm, with its short side that has
   the **two dark corner squares against the arm's base**. Press **Capture table**. It gets the ★ and sets the
   table.
5. Look at the coloured axes: they start at (0, 0), the inner corner next to a dark corner square, and that must
   be **next to the arm**. If they start at the far end, turn the board 180 degrees, delete that capture and
   capture it again. **Write down which way x and y point.**
6. **Detection tab:** press **Defaults** in *Thresholds*, then **Save**. That clears thresholds tuned for an
   earlier setup.

**Leave the board exactly where it is.** Close calib_app with Ctrl-C in T2.

## 5. Show the arm where the table is (T2)

1. Close the gripper, so the two fingertips meet: `./lab.sh cmd '{"op": "close"}'`.
2. Hold the arm, then:

```bash
./lab.sh torque-off
./lab.sh python robot_frame.py
```

3. Move the arm by hand until the fingertips touch a corner of the board squares. Type that corner's position in mm:
   `x y`, counted from the (0, 0) corner of step 4, in squares times the square size (for 70 mm squares, for
   example `70 140`).
4. Do this for 6 corners spread over the part the arm reaches. Press Enter on an empty line to finish. It should
   end with **OK**.
5. Check with new corners. Every error must be **under 5 mm**, otherwise repeat this step:

```bash
./lab.sh python robot_frame.py --check
```

6. **Take the board away**, carefully: the table must not move. Lay the arm down and **take your hands off it**.
   In **T1** press Ctrl-C, wait a few seconds and start `./lab.sh arm` again: the arm rises to its home pose.

## 6. Check the detection (T2)

The arm must be up before the pieces go on the table. Put pieces on the white table in the work area, then:

```bash
./lab.sh python calib_app.py
```

**Detection tab**, **Live detection** on. Every piece should be **coloured**: both cameras agree on it. Grey with `?`
is a piece only one camera finds; the runner ignores those.

- A piece is missing: switch **Near misses** on. The grey circle with scores shows which limit keeps it out. First
  try *rim* 0.5 instead of 0.6. A colour isn't found: click the middle of that piece's top (*Click to tune*).
  Press **Save** after any change.
- Then press the **bug button**, **Save snapshot**, and send the zip from `calib_data/debug/`: with it the
  thresholds can be checked away from the lab.

Close calib_app with Ctrl-C.

## 7. Make the layout and place the pieces (T2)

```bash
./lab.sh python lab_layout.py --conditions smoke correct
./lab.sh python lab_layout.py --view --condition smoke
```

Put one piece on each of the 3 thick circles (r1, g1, b1). The circle gets thicker when the camera sees the right
colour there. Press **q** when all 3 are thick. If some circles are off the table, add
`--bounds XMIN XMAX YMIN YMAX` (table mm) to the first command.

## 8. Check again, then the hover test (T2)

```bash
./lab.sh python lab_check.py
```

Fix anything that says FAIL. *Spoken answers* can stay a warning.

```bash
./lab.sh python lab_check.py --hover
```

The arm stops above each piece. Answer whether the fingertips are really above it. If it goes somewhere completely
different, the (0, 0) corner in step 5 was the wrong one: redo step 5.

**One piece at a time, by hand** (optional):

```bash
./lab.sh python robot_app.py
```

Open <http://localhost:8082>. Detection tab: switch **Live detection** on. Robot tab: click a piece, then **Hover**,
or **Pick** and then a free spot and **Place**. **Air** is on at every start: the arm then stops 40 mm above the
piece and grips nothing. Switch it off for a real pick. A greyed-out button has its reason under *Target*
("Pick is off: …"). Every click, command and answer, with the reason when something is refused, is written to the
terminal and to `logs/robot_app.log`.

**Keyboard:** switch **Keyboard** on in the Robot tab (it switches itself off when you leave the tab). Each press
turns one joint by 0.1 rad, slowly: `1`/`q` joint 1 (base, left/right), `2`/`w` joint 2, `3`/`e` joint 3,
`4`/`r` joint 4 (wrist). `o` opens the gripper, `p` closes it, `h` goes home, **Esc** stops. A step past a joint's
limit, into the table or into the arm itself is refused, with the reason on screen and in the log. **Close
gripper** is also a button. An arm side started before this version answers `unknown op 'jog'`: restart T1.

## 9. Dry run (T2)

```bash
./lab.sh python runner.py --participant TEST --vision cameras --no-speech --air
```

Open <http://localhost:8081> and press **Start**. The arm does everything slowly and high up, without touching a
piece. `--no-speech` means no microphone: if the robot asks something, click the answer on the panel. The red
**STOP** button halts it. Close it with Ctrl-C when it says done.

## 10. The real thing (T2)

The same command without `--air`:

```bash
./lab.sh python runner.py --participant TEST --vision cameras --no-speech
```

Three pieces sorted into their zones means **the robot works**.

## Every later session

Once steps 1 to 10 have worked, a session is step 0 (`git pull --autostash`) and then one command in one terminal.
Stop `./lab.sh arm` first if it is still running. If a camera or the table moved since, redo from step 4 first.

1. Table clear of pieces, arm powered on.
2. Start it. The arm rises by itself; if it doesn't, the reason is printed.

```bash
./lab.sh run P07 --no-speech
```

3. When it says `Arm is up`, lay out the pieces, open <http://localhost:8081> and press **Start**.
4. When the robot asks something, the panel shows the buttons **yes**, **no**, **red**, **blue**, **green**:
   click what the participant answers. The robot still says its question out loud.
5. At the end: hold the arm, then Ctrl-C.

`--no-speech` means the answers are clicked on the panel, and no microphone is needed. Leave it out
(`./lab.sh run P07`) to have the participant's spoken answers recognised; the buttons work then too.
Other options of `runner.py` go in the same place, for example `./lab.sh run TEST --no-speech --air`.

## What a healthy start prints

`./lab.sh arm` prints these lines, in this order (`./lab.sh run` writes them to `logs/arm_side_*.log`).
The first one that is missing shows where the start broke:

1. `Succeeded to open the port!` and `Succeeded to change the [1000000] baudrate!`: the USB adapter answers.
2. `[ID:011] Request ping - Ping succeeded. Dynamixel model number : 1020`, once for each of 011 to 015:
   every motor answers. A missing ID is that motor: its cable, its ID or its baud rate.
3. `[ID:011] Torque ON`, again for 011 to 015, then `Dynamixel Hardware Start!`
4. `Configured and activated` for `joint_state_broadcaster`, `arm_controller` and `gripper_controller`.

`No real-time kernel detected` is a normal warning.

## If something goes wrong

| What you see | What to do |
|---|---|
| A motor's `Ping succeeded` line is missing, or the motors answer but a joint or the gripper doesn't move | The settings stored in the motors: the start doesn't repair them. Stop `./lab.sh arm` and check with ROBOTIS' DYNAMIXEL Wizard 2.0: IDs 11 to 14 (joints) and 15 (gripper), all at 1 Mbps, joints in *Position Control Mode*, gripper in *Current based Position Control Mode*. Old firmware is updated there too |
| `/dev/ttyUSB0 doesn't exist` | Check the arm's USB cable and power |
| `no permission on /dev/ttyUSB0` | `./lab.sh usb` (installs the USB rule, asks for the sudo password) |
| The arm is jerky, or the controllers don't start | `cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` must print `1`. If not: `./lab.sh usb` |
| `ARM NOT READY: the controllers aren't active` or `the arm's driver (ros2_control_node) stopped` | Look above it for `Torque is enabled, cannot proceed` (a crashed or killed start leaves the torque on, and Ctrl-C can't release it then): hold the arm, `./lab.sh torque-off` (or switch its power off and on), start again. `Cannot connect` or no motor found: the arm's power and the cable from the USB adapter to the motors |
| `the ROS side is already running` | Ctrl-C in the terminal that runs `./lab.sh arm` or `./lab.sh run`, wait a few seconds |
| `the arm did not get there` | A motor gave up. If a motor LED blinks: overload, switch the arm's power off and on, then restart. Otherwise something blocks the arm |
| `joints ... are outside their limits` | The arm lies past an end stop: `./lab.sh torque-off`, move it a little by hand, restart `./lab.sh arm` |
| No motor LED blinks once when the power is switched on | Power supply or the cable chain between the motors, not software |
| `home` refuses, "cannot be lifted by a command" | Take the pieces near the arm away, then `./lab.sh cmd '{"op": "wake"}'` |
| The arm doesn't rise after `ARM SIDE READY` | Read the reason printed there. "controller not running": power and USB of the arm, then restart `./lab.sh arm` |
| The arm doesn't go to a piece, or Pick is greyed out (`robot_app.py`) | Read "Pick is off: …" under *Target*, or the last lines of `logs/robot_app.log`. `out of reach … mm from the base axis`: keep pieces about 100 to 210 mm from the arm's base. `not within 25 mm of a piece BOTH cameras see`: only pieces both cameras find can be picked; Live detection on, better light or thresholds. The arm side's own answer (`cmd:` and `"failed"` lines) is in the `./lab.sh arm` terminal |
| A camera won't open | Close `calib_app.py` / `robot_app.py`. Only one program can use a camera at a time |
| "Cameras found: none" for the Basler cameras (USB) | `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2676", MODE="0666"' \| sudo tee /etc/udev/rules.d/69-basler.rules`, then `sudo udevadm control --reload-rules`, and replug them |
| Basler cameras not found (network models) | They need an IP on the PC's network: ask the lab staff, or set it in Basler's pylon Viewer |
| Basler image too dark or colours shifting | Exposure and white balance are automatic: add light, and don't change it during a session |
| Only one camera available | `calib_gui.py --one-camera` tries out the GUI, but can't be used for the real run |
| "Camera gave no image" | Plug the two cameras into different USB ports |
| A colour isn't detected, or a piece is seen twice | Check the lighting, then tune the thresholds under *Thresholds* in calib_app.py's Detection tab (the colour mask shows what passes) and press Save |
| "Closed on nothing" with a piece in the gripper | Lower `EMPTY_MARGIN` in `arm_node.py`, and restart `./lab.sh arm` |
| An empty gripper isn't noticed by the arm | Expected for now: the cameras catch the miss. Not yet tuned on the real arm |
| The fingers hit the table or miss the piece | Press STOP, then adjust `GRASP_Z` in `kinematics.py` and restart `./lab.sh arm` |
| Anything alarming | Power switch |
