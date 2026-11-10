# First session on the lab PC

Do these in order. Stop at the first step that fails: it prints what to do.
Keep one hand near the arm's **power switch** whenever it can move.

## 1. Get the code (once)

Open a terminal:

```bash
git clone https://github.com/martinkovacs/embodied.git
cd embodied
./lab.sh setup
```

It asks for the sudo password once, to install the arm's USB rule, and takes a few minutes. If it says packages are missing, run the command it prints (or ask the lab staff), then
run `./lab.sh setup` again. It should end with `Setup finished.`

## 2. Start the arm

1. Plug in the arm and turn on its power.
2. Clear the table around the arm: no pieces, cables or hands within its reach.
3. In the terminal:

```bash
./lab.sh arm
```

Wait for `ARM SIDE READY`. The arm then **rises slowly to its home pose by itself**, from wherever it was lying:
that is the sign that it works. If it stays down, the line above the warning says why.

**Leave this terminal running.** Everything below goes in a **second terminal**
(`cd embodied` first).

Two rules for every start and stop of the arm side, today and later:

- **Starting:** no pieces near the arm. It may sweep sideways while it gets up. Pieces go on the table after it is up.
- **Stopping (Ctrl-C):** hold the arm first, or let it rest on the table. Its motors let go and it falls.
  Never just close the terminal: the motors then stay on, and the next start refuses with
  `Torque is enabled, cannot proceed` until the arm's power is switched off and on.

## 3. Check that it moves

```bash
./lab.sh python lab_check.py
```

Many lines say FAIL at this point, because nothing is calibrated yet. These four must be OK or WARN:
*Arm USB adapter, ROS connection, Arm controllers, arm_node*.

To move the arm without the cameras or the experiment, send it one command and read the answer:

```bash
./lab.sh cmd '{"op": "open"}'
./lab.sh cmd '{"op": "home"}'
```

`"state": "done"` means it did it; `"failed"` comes with the reason.

Plug in both cameras too, if they aren't already.

## 4. Calibrate the cameras

```bash
./lab.sh python calib_gui.py
```

A window opens.

1. **Setup tab:** press **Use the Basler cameras**. Set the board's real square size and thickness. Press **Connect**.
2. **Stereo tab:** hold the checkerboard in view of both cameras and press capture, 15 to 25 times, at different
   angles and in different corners of the image. Press calibrate. The RMS must be **under 0.5**.
3. **Table pose tab:** lay the board flat on the table in front of the arm, seen by both cameras. Press
   **Capture table pose**.
4. **Write down which corner the coloured axes start from, and which way x and y point.** That corner is (0, 0).

**Don't move the cameras, the table or the arm from now on.** Leave the board where it is and close the window.

## 5. Show the arm where the table is

1. Close the gripper, so the two fingertips meet.
2. Hold the arm, then run:

```bash
./lab.sh torque-off
./lab.sh python robot_frame.py
```

3. Move the arm by hand until the fingertips touch a corner of the board squares. Type that corner's position in mm:
   `x y`, counted from the (0, 0) corner of step 4. Squares are 25 mm, so for example `50 75`.
4. Do this for 6 corners spread over the board. Press Enter on an empty line to finish. It should end with **OK**.
5. Check with new corners. Every error must be **under 5 mm**, otherwise repeat this step:

```bash
./lab.sh python robot_frame.py --check
```

6. Take the board away. Lay the arm down and **take your hands off it**. In the first terminal press Ctrl-C, wait
   a few seconds and start `./lab.sh arm` again: the arm rises to its home pose by itself.

## 6. Make the layout and place the pieces

```bash
./lab.sh python lab_layout.py --conditions smoke correct
./lab.sh python lab_layout.py --view --condition smoke
```

The arm must be up before the pieces go on the table. Put one piece on each of the 4 thick circles (r1, w1, g1, b1). The circle gets thicker when the camera sees the
right colour there. White pieces need a dark mat under them. Press **q** when all 4 are thick.

If some circles are off the table, add `--bounds XMIN XMAX YMIN YMAX` (table mm) to the first command.

## 7. Check again

```bash
./lab.sh python lab_check.py
```

Fix anything that says FAIL. *Spoken answers* can stay a warning today.

## 8. Hover test

```bash
./lab.sh python lab_check.py --hover
```

The arm stops above each piece. Answer whether the fingertips are really above it. If it goes somewhere completely
different, the (0, 0) corner in step 5 was the wrong one: redo step 5.

## 9. Dry run

```bash
./lab.sh python runner.py --participant TEST --vision cameras --no-speech --air
```

Open <http://localhost:8081> and press **Start**. The arm does everything slowly and high up, without touching a
piece. The red **STOP** button halts it. Close it with Ctrl-C when it says done.

## 10. The real thing

The same command without `--air`:

```bash
./lab.sh python runner.py --participant TEST --vision cameras --no-speech
```

Four pieces sorted into their zones means **the robot works**.

## Every later session

Once steps 1 to 10 have worked, a session is one command in one terminal. Stop `./lab.sh arm` first if it is
still running.

1. Table clear of pieces, arm powered on.
2. Start it. The arm rises by itself; if it doesn't, the reason is printed.

```bash
./lab.sh run P07
```

3. When it says `Arm is up`, lay out the pieces, open <http://localhost:8081> and press **Start**.
4. At the end: hold the arm, then Ctrl-C.

Options of `runner.py` go after the participant, for example `./lab.sh run TEST --no-speech --air`.

## If something goes wrong

| What you see | What to do |
|---|---|
| `/dev/ttyUSB0 doesn't exist` | Check the arm's USB cable and power |
| `no permission on /dev/ttyUSB0` | `./lab.sh usb` (installs the USB rule, asks for the sudo password) |
| The arm is jerky, or the controllers don't start | `cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` must print `1`. If not: `./lab.sh usb` |
| `ARM NOT READY: the controllers aren't active` | Look above it for `Torque is enabled, cannot proceed`: hold the arm, switch its power off and on, start again. `Cannot connect` or no motor found: the arm's power and the cable from the USB adapter to the motors |
| `the ROS side is already running` | Ctrl-C in the terminal that runs `./lab.sh arm` or `./lab.sh run`, wait a few seconds |
| `the arm did not get there` | A motor gave up. If a motor LED blinks: overload, switch the arm's power off and on, then restart. Otherwise something blocks the arm |
| `joints ... are outside their limits` | The arm lies past an end stop: `./lab.sh torque-off`, move it a little by hand, restart `./lab.sh arm` |
| No motor LED blinks once when the power is switched on | Power supply or the cable chain between the motors, not software |
| `home` refuses, "cannot be lifted by a command" | Take the pieces near the arm away, then `./lab.sh cmd '{"op": "wake"}'` |
| The arm doesn't rise after `ARM SIDE READY` | Read the reason printed there. "controller not running": power and USB of the arm, then restart `./lab.sh arm` |
| A camera won't open | Close `calib_gui.py`. Only one program can use a camera at a time |
| "Cameras found: none" for the Basler cameras (USB) | `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2676", MODE="0666"' \| sudo tee /etc/udev/rules.d/69-basler.rules`, then `sudo udevadm control --reload-rules`, and replug them |
| Basler cameras not found (network models) | They need an IP on the PC's network: ask the lab staff, or set it in Basler's pylon Viewer |
| Basler image too dark or colours shifting | Exposure and white balance are automatic: add light, and don't change it during a session |
| Only one camera available | `calib_gui.py --one-camera` tries out the GUI, but can't be used for the real run |
| "Camera gave no image" | Plug the two cameras into different USB ports |
| A colour isn't detected | Check the lighting, then tune the HSV ranges in `pieces.py` |
| "Closed on nothing" with a piece in the gripper | Lower `EMPTY_MARGIN` in `arm_node.py`, and restart `./lab.sh arm` |
| An empty gripper isn't noticed by the arm | Expected for now: the cameras catch the miss. Not yet tuned on the real arm |
| The fingers hit the table or miss the piece | Press STOP, then adjust `GRASP_Z` in `kinematics.py` and restart `./lab.sh arm` |
| Anything alarming | Power switch |
