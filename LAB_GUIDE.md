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

It takes a few minutes. If it says packages are missing, run the command it prints (or ask the lab staff), then
run `./lab.sh setup` again. It should end with `Setup finished.`

## 2. Start the arm

1. Plug in the arm and turn on its power.
2. Move the arm by hand so the fingers are well above the table (a hand's width is plenty).
3. In the terminal:

```bash
./lab.sh arm
```

Wait for `ARM SIDE READY`. **Leave this terminal running.** Everything below goes in a **second terminal**
(`cd embodied` first). To stop the arm side, press Ctrl-C here.

## 3. Check that it moves

```bash
./lab.sh python lab_check.py
```

Many lines say FAIL at this point, because nothing is calibrated yet. These four must be OK or WARN:
*Arm USB adapter, ROS connection, Arm controllers, arm_node*.

Plug in both cameras too, if they aren't already.

## 4. Calibrate the cameras

```bash
./lab.sh python calib_gui.py
```

A window opens.

1. **Setup tab:** type the two camera numbers shown after "Cameras found" (left and right). Set the board's real
   square size and thickness. Press **Connect**.
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

6. Take the board away. Support the arm, go to the first terminal, press Ctrl-C, and start `./lab.sh arm` again.

## 6. Make the layout and place the pieces

```bash
./lab.sh python lab_layout.py --conditions smoke correct
./lab.sh python lab_layout.py --view --condition smoke
```

Put one piece on each of the 4 thick circles (r1, w1, g1, b1). The circle gets thicker when the camera sees the
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

## If something goes wrong

| What you see | What to do |
|---|---|
| `/dev/ttyUSB0 doesn't exist` | Check the arm's USB cable and power |
| `no permission on /dev/ttyUSB0` | `sudo chmod 666 /dev/ttyUSB0` |
| `home` refuses, "cannot be lifted by a command" | Stop `./lab.sh arm`, lift the arm by hand, start it again |
| A camera won't open | Close `calib_gui.py`. Only one program can use a camera at a time |
| Only one camera available | `calib_gui.py --one-camera` tries out the GUI, but can't be used for the real run |
| "Camera gave no image" | Plug the two cameras into different USB ports |
| A colour isn't detected | Check the lighting, then tune the HSV ranges in `pieces.py` |
| "Closed on nothing" with a piece in the gripper | Lower `EMPTY_MARGIN` in `arm_node.py`, and restart `./lab.sh arm` |
| The fingers hit the table or miss the piece | Press STOP, then adjust `GRASP_Z` in `kinematics.py` and restart `./lab.sh arm` |
| Anything alarming | Power switch |
