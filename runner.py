"""Experiment runner (architecture.md sections 5-6): the robot sorts the pieces of a fixed layout into colour
zones, one piece = one trial, behaving as the condition's script says (correct / mistake / ask), and logs it all.

Per trial attempt: [home] -> detect -> pick -> [ask: show, "Is this <colour>?", wait yes/no] -> place -> home -> verify.

Files:
  scenarios/<condition>.yaml   slot order + per-trial script; names its layout
  layouts/*.yaml               the printed template: piece slots and zone spots (table mm), robot frame file
  participants.csv             participant -> conditions in the order they run (counterbalancing)
  audio/ask_<colour>.wav, which.wav, done.wav   the robot's recorded lines (SAY); without them macOS `say` stands in
  logs/<participant>_<condition>.jsonl         every event; a block that never finished is resumed on the next start
  logs/<participant>_<condition>/<run>/        trialNN_aA_{before,after}_{left,right}.png, speech/*.wav (A = attempt)

Participant answers come in through Runner.answer(): the operator panel buttons, or speech recognition with the
speech onset time. Anything unplanned (piece not where the layout says, arm failure, failed verify) halts the trial
and asks the operator to retry / skip / abort, or accept when the camera missed a piece the arm did place. The
robot never improvises a question: that would contaminate the conditions. STOP halts the arm; Resume asks the
operator the same about the interrupted trial, then carries on with the rest of the session.

Usage (from the repo root):
    venv/bin/python runner.py --participant P00 --vision synthetic|cameras [--block 1] [--arm ros|fake]
                              [--host localhost]         # rosbridge
                              [--panel-host 127.0.0.1]   # operator panel on http://127.0.0.1:8081
                              [--lang en|hu] [--mic N] [--no-speech]   # spoken answers (speech.py), or buttons only
                              [--no-voice]               # the robot stays silent (logged)
    venv/bin/python runner.py --self-test                # every condition: fake arm, synthetic vision, no ROS
"""
import argparse
import asyncio
import copy
import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

import cameras
import pieces
import robot_frame
from arm_node.kinematics import GRASP_Z, HOME, LIFT, PLACE_Z, SHOW, check_travel, ik  # noqa: E402  via robot_frame
from stereo_calib import StereoCalib

ROOT = Path(__file__).resolve().parent
COLORS = ("red", "blue", "green", "white")
BEHAVIORS = ("correct", "mistake", "ask")
MATCH_MM = 20.0                 # a detection this close to a slot / spot is the piece there (pieces are 35 mm)
SPACING_MM = 40.0               # zone spots at least this far from every other spot and slot (centre to centre)
REACH_Z = (GRASP_Z, PLACE_Z, GRASP_Z + LIFT)   # arm_node's heights: every slot and spot must be reachable at all
ARM_TIMEOUT_S = 60.0
STOP_CONFIRM_S = 8.0            # the node's worst case to confirm a stop: GOAL_ACCEPT_S + CANCEL_ACK_S
PORT = 8081
SAY = {"ask": "Is this {color}?", "which": "Which colour is it?", "done": "Done."}
PLAYERS = ("pw-play", "paplay", "aplay", "afplay")   # first one installed plays the WAVs


class Unplanned(Exception):
    pass


class Aborted(Exception):
    pass


def load_scenario(condition):
    """scenarios/<condition>.yaml with its layout and robot frame, checked (check_scenario)."""
    sc = yaml.safe_load((ROOT / "scenarios" / f"{condition}.yaml").read_text())
    assert sc["condition"] == condition, f"{condition}.yaml says condition {sc['condition']}"
    lay = yaml.safe_load((ROOT / sc["layout"]).read_text())
    return check_scenario(sc, lay, robot_frame.load(ROOT / lay["robot_frame"])) | {"layout": sc["layout"]}


def check_scenario(sc, lay, T):
    """Parsed scenario + layout, checked: ids, script entries, that the arm reaches every slot and spot and travels
    between them clear of the table, spacing, and room in every zone for the worst case. AssertionError names
    the slot / spot / zone at fault."""
    slots = {k: (c, float(x), float(y)) for k, (c, x, y) in lay["slots"].items()}
    zones = {c: [tuple(map(float, p)) for p in lay["zones"][c]] for c in COLORS}
    script = {int(k): v for k, v in (sc.get("script") or {}).items()}
    order = sc["order"]
    assert all(s in slots for s in order) and len(set(order)) == len(order), "order: unknown or repeated slot"
    assert all(c in COLORS for c, _, _ in slots.values()), "slots: unknown colour"
    for i, b in script.items():
        assert 1 <= i <= len(order) and b["behavior"] in BEHAVIORS, f"script {i}: {b}"
        if b["behavior"] == "mistake":
            assert b.get("put_to") in COLORS and b["put_to"] != slots[order[i - 1]][0], f"script {i}: bad put_to"
        else:   # put_to anywhere else would be a mistake the log calls something else
            assert "put_to" not in b, f"script {i}: put_to only goes with behavior: mistake"

    # reach: name -> table (x, y) and robot (x, y, z of the table there)
    slot_xy = {f"slot {k} ({x:.0f}, {y:.0f})": (x, y) for k, (_, x, y) in slots.items()}
    spot_xy = {f"{c} spot {i} ({x:.0f}, {y:.0f})": (x, y) for c in COLORS for i, (x, y) in enumerate(zones[c], 1)}
    robot = {name: robot_frame.to_robot(T, (x, y, 0.0)) for name, (x, y) in (slot_xy | spot_xy).items()}
    above = {}  # the approach pose arm_node travels to and from
    for name, (rx, ry, rz) in robot.items():
        try:
            for z in REACH_Z:
                ik(rx, ry, rz + z)
        except ValueError as e:
            raise AssertionError(f"layout {name}: {e}") from None
        above[name] = ik(rx, ry, rz + GRASP_Z + LIFT)

    # travel: every move a trial makes, and the recovery moves home, at the table height arm_node checks it with
    def travel(a, b, q_from, q_to, table_z):
        try:
            check_travel(q_from, q_to, table_z)
        except ValueError as e:
            raise AssertionError(f"travel {a} -> {b}: {e}") from None
    top = max(z for _, _, z in robot.values())
    travel("show", "home", SHOW, HOME, top)
    for s in slot_xy:
        travel("home", s, HOME, above[s], robot[s][2])
        travel(s, "home", above[s], HOME, robot[s][2])
        travel(s, "show", above[s], SHOW, robot[s][2])
        for p in spot_xy:
            travel(s, p, above[s], above[p], max(robot[s][2], robot[p][2]))
    for p in spot_xy:
        travel("show", p, SHOW, above[p], robot[p][2])
        travel(p, "home", above[p], HOME, robot[p][2])

    # spacing: the open fingers need room around every piece they go down to
    slot_min = float(lay.get("min_slot_spacing_mm", SPACING_MM))
    spots, slot_items = list(spot_xy.items()), list(slot_xy.items())
    for i, (a, (ax, ay)) in enumerate(spots):
        for b, (bx, by) in spots[i + 1:] + slot_items:
            d = np.hypot(ax - bx, ay - by)
            assert d >= SPACING_MM, f"layout {a} is {d:.0f} mm from {b}, under {SPACING_MM:.0f} mm"
    for i, (a, (ax, ay)) in enumerate(slot_items):
        for b, (bx, by) in slot_items[i + 1:]:
            d = np.hypot(ax - bx, ay - by)
            assert d >= slot_min, f"layout {a} is {d:.0f} mm from {b}, under {slot_min:.0f} mm"

    # worst case: an asked participant may say "no" and name any colour
    need = dict.fromkeys(COLORS, 0)
    for i, slot in enumerate(order, 1):
        b = script.get(i, {"behavior": "correct"})
        for c in {"correct": [slots[slot][0]], "mistake": [b.get("put_to")], "ask": COLORS}[b["behavior"]]:
            need[c] += 1
    for c in COLORS:
        assert len(zones[c]) >= need[c], f"layout {c} zone: {len(zones[c])} spots, the script can need {need[c]}"
    return dict(order=order, script=script, slots=slots, zones=zones, T=T)


def nearest(found, x, y):
    """The detection [(color, x, y)] closest to (x, y) within MATCH_MM, or None."""
    d = [np.hypot(px - x, py - y) for _, px, py in found]
    return found[int(np.argmin(d))] if d and min(d) < MATCH_MM else None


class Vision:
    """Stereo detection with calib_gui.py's calibration and camera settings: [(color, x, y)] table mm + the image
    pair. Blocking (~1 s): the runner calls detect() in a worker thread."""

    def __init__(self, synthetic=False):
        if not cameras.CALIB_PATH.exists():
            raise RuntimeError(f"no {cameras.CALIB_PATH}: run calib_gui.py (stereo calibration + table pose)")
        self.calib = StereoCalib.load(cameras.CALIB_PATH)
        if self.calib.T_left_table is None:
            raise RuntimeError(f"{cameras.CALIB_PATH} has no table pose: run calib_gui.py and capture the table pose")
        st = cameras.load_settings()
        b = float(st["board_mm"])
        self.views = pieces.view_from_calib(self.calib, "left", b), pieces.view_from_calib(self.calib, "right", b)
        w, h = int(st["width"]), int(st["height"])
        keys = ("test_scene_left", "test_scene_right") if synthetic else ("left", "right")
        self.sources = [cameras.Source(st[k], w, h) for k in keys]
        self.lock = threading.Lock()    # a detection a STOP abandoned may still be running when the next one starts

    def detect(self):
        with self.lock:
            left, right = cameras.fresh_pair(*self.sources)
            if left is None or right is None:
                raise Unplanned("camera gave no image")
            found = pieces.matched_pieces(self.calib, pieces.detect_pieces(left, self.views[0]),
                                          pieces.detect_pieces(right, self.views[1]))
        # only pieces both cameras agree on: one camera alone is where false detections come from
        return [(p.color, float(st[0]), float(st[1])) for p, st in found if st is not None], (left, right)

    def after(self, op, x, y):
        pass


class SimWorld(Vision):
    """The synthetic scene detected once, then kept up to date from the arm's moves: stands in for the cameras
    while the arm is ROBOTIS fake hardware or Gazebo without cameras, where no image would change."""

    def __init__(self):
        super().__init__(synthetic=True)
        self.initial, _ = super().detect()
        self.reset()

    def reset(self):
        self.found, self.held = list(self.initial), None

    def detect(self):
        return list(self.found), None

    def after(self, op, x, y):
        if op == "pick" and (p := nearest(self.found, x, y)):
            self.found.remove(p)
            self.held = p[0]
        elif op == "place" and self.held:
            self.found.append((self.held, x, y))
            self.held = None


class RosArm:
    """arm_node's /arm/cmd + /arm/status over rosbridge. x, y, z robot frame mm (z: the table surface there)."""

    def __init__(self, host):
        import roslibpy
        self.roslibpy = roslibpy
        self.ros = roslibpy.Ros(host=host, port=9090)
        self.ros.run(timeout=10)  # raises if rosbridge isn't up
        self.cmd = roslibpy.Topic(self.ros, "/arm/cmd", "std_msgs/String")
        self.waiting, self.n, self.last_stop = {}, 0, None
        self.nonce = os.urandom(3).hex()    # a reply meant for an earlier runner (or replayed) never matches ours
        roslibpy.Topic(self.ros, "/arm/status", "std_msgs/String").subscribe(self.on_status)

    @property
    def connected(self):
        return self.ros.is_connected

    def on_status(self, msg):  # roslibpy's thread
        s = json.loads(msg["data"])
        if s["state"] != "started" and (w := self.waiting.pop(s.get("id"), None)):
            loop, fut = w
            try:
                loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(s))
            except RuntimeError:    # the loop is gone (shutdown): nobody is awaiting this reply any more
                pass

    async def do(self, op, x=None, y=None, z=None):
        self.n += 1
        cid = f"{self.nonce}-{self.n}"
        cmd = ({"id": cid, "op": op} | ({} if x is None else {"x": round(x, 1), "y": round(y, 1)})
               | ({} if z is None else {"z": round(z, 1)}))
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.waiting[cid] = (loop, fut)
        try:
            if not self.connected:
                raise RuntimeError("no connection to rosbridge")
            self.cmd.publish(self.roslibpy.Message({"data": json.dumps(cmd)}))
            s = await asyncio.wait_for(fut, ARM_TIMEOUT_S)
        except TimeoutError:
            self.stop()  # a hung controller keeps arm_node busy until its goal is cancelled (fire-and-forget)
            raise RuntimeError(f"no answer from arm_node in {ARM_TIMEOUT_S:.0f} s (is it running?)") from None
        finally:
            self.waiting.pop(cid, None)
        if s["state"] != "done":
            raise RuntimeError(s.get("reason", s["state"]))

    def stop(self):
        """True if the stop went out. roslibpy would queue it while disconnected and send it on reconnect: worse.
        Each stop gets its own id, so on_status can match the node's answer instead of every stop looking like
        id 0: stop_confirm() awaits that answer."""
        if not self.connected:
            return False
        self.n += 1
        cid = f"{self.nonce}-{self.n}"
        loop = asyncio.get_running_loop()   # a stop is only ever sent from the panel's / do's event loop
        fut = loop.create_future()
        self.waiting[cid] = (loop, fut)     # nothing else to match: other commands' ids are distinct
        self.last_stop = cid, fut
        self.cmd.publish(self.roslibpy.Message({"data": json.dumps({"id": cid, "op": "stop"})}))
        return True

    async def stop_confirm(self):
        """(True, None) once the latest stop was confirmed: the node's "done" (goal cancelled / already terminated /
        nothing ran). (False, reason) otherwise: the node failed to cancel, or sent no answer within STOP_CONFIRM_S
        (its own worst case is GOAL_ACCEPT_S + CANCEL_ACK_S): presume the arm may still move. Only the latest stop
        counts: an earlier one (a timed-out command's, which nobody awaits) says nothing about this one."""
        if not self.last_stop:
            return False, "no stop sent"
        cid, fut = self.last_stop
        try:
            s = await asyncio.wait_for(fut, STOP_CONFIRM_S)
        except TimeoutError:
            return False, f"no answer from arm_node in {STOP_CONFIRM_S:.0f} s (is it running?)"
        finally:
            self.waiting.pop(cid, None)
        return (True, None) if s["state"] == "done" else (False, s.get("reason", s["state"]))


class FakeArm:
    connected = True

    async def do(self, op, x=None, y=None, z=None):
        await asyncio.sleep(0)

    def stop(self):
        return True

    async def stop_confirm(self):
        return True, None       # the fake arm is still at once: nothing to confirm


class Voice:
    """The robot's lines: audio/ask_<colour>.wav, audio/which.wav, audio/done.wav through the first installed player,
    else macOS `say` as a stopgap until the WAVs are recorded (one voice, one tempo for every participant).
    lines: [(key, color or None)] the scenario can say."""

    def __init__(self, lines, audio_dir=ROOT / "audio"):
        self.lines, self.dir = lines, Path(audio_dir)

    def wav(self, key, color=None):
        p = self.dir / (f"{key}_{color}.wav" if color else f"{key}.wav")   # no generic ask.wav: it can't say the colour
        return p if p.exists() else None

    def how(self, key, color=None):
        """(argv, via, asset) that speaks the line, via "wav" | "say"; (None, None, None) if nothing can."""
        wav, player = self.wav(key, color), next((p for p in map(shutil.which, PLAYERS) if p), None)
        if wav and player:
            return [player, str(wav)], "wav", wav.name
        if say := shutil.which("say"):
            return [say, SAY[key].format(color=color)], "say", None
        return None, None, None

    def problems(self):
        out = []
        for key, color in self.lines:
            if self.how(key, color)[0] is None:
                missing = f"no player ({', '.join(PLAYERS)})" if self.wav(key, color) else \
                    f"no {self.dir.name}/{key}{f'_{color}' if color else ''}.wav"
                out.append(f'"{SAY[key].format(color=color)}": {missing} and no `say`')
        return out

    def stand_in(self):
        """The lines that would come from `say` rather than a recording."""
        return [SAY[k].format(color=c) for k, c in self.lines if self.how(k, c)[1] == "say"]


class Log:
    """One JSONL line per event (architecture.md section 6). Appends: a restarted block never loses data. Every
    record carries this run's id and the attempt; camera images and speech audio go to <log>/<run>/, so a rerun of
    the same block never overwrites what an earlier run's records point to."""

    def __init__(self, path, participant, condition):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.earlier = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] \
            if path.exists() else []
        self.path, self.f = path, path.open("a")
        self.participant, self.condition, self.trial, self.attempt, self.tail = participant, condition, 0, 1, []
        self.run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.media = path.with_suffix("") / self.run

    def __call__(self, event, trial=None, **data):
        rec = {"t_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "t_mono": round(time.monotonic(), 3), "participant": self.participant,
               "condition": self.condition, "run": self.run, "trial": self.trial if trial is None else trial,
               "attempt": self.attempt, "event": event, "data": data}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.tail.append(f"{rec['t_wall'][11:23]}  #{rec['trial']:<2} {event} {json.dumps(data) if data else ''}")

    def images(self, pair, tag, trial, attempt):
        if pair is not None:
            self.media.mkdir(parents=True, exist_ok=True)
            for side, img in zip(("left", "right"), pair):
                cv2.imwrite(str(self.media / f"trial{trial:02d}_a{attempt}_{tag}_{side}.png"), img)

    def unfinished(self):
        """The events of the last session in the file if it never logged session_end (crash, kill), else None."""
        starts = [i for i, e in enumerate(self.earlier) if e["event"] == "session_start"]
        if not starts:
            return None
        events = self.earlier[starts[-1]:]
        return None if any(e["event"] == "session_end" for e in events) else events


class Runner:
    def __init__(self, participant, condition, arm, vision, log_dir=ROOT / "logs", auto=None, audio=True, info=None):
        self.sc = load_scenario(condition)
        self.arm, self.vision, self.auto, self.audio = arm, vision, auto, audio
        self.info = info or {}                 # how this run is set up, for session_start: vision, arm, language, ...
        self.log = Log(Path(log_dir) / f"{participant}_{condition}.jsonl", participant, condition)
        asked = sorted({self.sc["slots"][self.sc["order"][i - 1]][0]
                        for i, b in self.sc["script"].items() if b["behavior"] == "ask"})
        self.voice = Voice([("ask", c) for c in asked] + ([("which", None)] if asked else []) + [("done", None)])
        self.voice_problem = ("; ".join(self.voice.problems()) or None) if audio else "audio off"   # None: all fine
        self.voice_stand_in = self.voice.stand_in() if audio else []   # lines macOS `say` speaks, not a recording
        self.used = dict.fromkeys(COLORS, 0)   # spots taken in each zone (reserved before each place)
        self.prompt = None                     # {"kind": "answer" | "operator", "options", "text", "fut", "since", ...}
        self.state = "ready"                   # ready | running | stopped | done | aborted | error
        self.status, self.task, self.said_at = "ready", None, None
        self.at_home = False                   # the arm's last op was a home that finished
        self.held = False                      # a piece is (or may be) in the gripper: after a pick, until a place / open
        self.in_trial = False                  # trial_start logged and its trial_end not: the trial has begun
        self.stop_problem = None               # why the last STOP wasn't confirmed: the arm may still be moving
        self.confirming = None                 # the task awaiting the node's answer to the last STOP
        self.on_say = (lambda: None, lambda: None)   # (a line starts, it ended): the speech bridge mutes around it
        self.new_trial()
        self.interrupted = "STOPPED by the operator."
        if earlier := self.log.unfinished():   # this block ran before and never finished: pick up where it stopped
            self.replay(earlier)
            self.state = "stopped"
            self.status = f"INTERRUPTED run found: Resume continues at trial {self.log.trial}"
            self.interrupted = ("The runner was restarted during this session (crash or quit): check the arm, the "
                                "gripper and the table against the log.")

    def new_trial(self):
        self.step = self.moving = None         # this attempt's last arm op done; the op in flight (stays if it failed)
        self.target = self.spot = None         # the zone and spot index reserved for this trial's place
        self.placed = None                     # the zone a finished place op put the piece in
        self.in_trial = False                  # a crash between trials never opened the next one

    def replay(self, events):
        """Rebuild the zone spots taken and the interrupted trial's state from an unfinished session's events."""
        for e in events:
            d = e["data"]
            if e["event"] == "trial_start":
                self.new_trial()
                self.log.trial, self.log.attempt, self.in_trial = e["trial"], e.get("attempt", 1), True
            elif e["event"] == "trial_end":
                self.log.trial, self.in_trial = e["trial"] + 1, False
                self.new_trial()
            elif e["event"] == "spot":
                self.used[d["zone"]] += 1 if d["action"] == "reserve" else -1
                self.target, self.spot = (d["zone"], d["index"]) if d["action"] == "reserve" else (None, None)
            elif e["event"] == "arm_cmd":
                self.moving = d["op"]
                if d["op"] == "pick":
                    self.held = True           # may be in the gripper from now on (move() does the same when sent)
            elif e["event"] == "arm_done":
                self.step, self.moving = d["op"], None
                self.held = d["op"] in ("pick", "show") or (self.held and d["op"] == "home")
                if d["op"] == "place":
                    self.placed = self.target
        self.log.trial = min(max(self.log.trial, 1), len(self.sc["order"]) + 1)

    # --- input: participant answers (panel or speech) and operator choices ---
    def answer(self, word, source="panel", t=None):
        """t: time.monotonic() of the speech onset, when the answer comes from speech."""
        if t is not None and self.said_at is not None and t < self.said_at:
            return  # speech that began before the robot's question ended: not an answer to it
        self._resolve("answer", word, source, t)

    def choose(self, option):
        self._resolve("operator", option, "operator", None)

    def _resolve(self, kind, value, source, t):
        p = self.prompt
        if p and p["kind"] == kind and value in p["options"] and not p["fut"].done():
            p["fut"].set_result((value, source, t))

    async def _wait(self, kind, options, text, **extra):
        if self.auto:
            return self.auto(kind, options), "auto", None
        self.prompt = {"kind": kind, "options": options, "text": text, "since": time.monotonic(),
                       "fut": asyncio.get_running_loop().create_future(), **extra}
        try:
            return await self.prompt["fut"]
        finally:
            self.prompt = None

    async def ask(self, options, color=None):
        """color: the colour the robot asked about (the speech bridge needs it to read "yes, it's red")."""
        word, source, t = await self._wait("answer", options, "waiting for the participant: " + " / ".join(options),
                                           color=color)
        # rt_s: end of the robot's question -> start of the participant's speech. Without a speech onset there is
        # no rt_s, only the experimenter's click time.
        rt = {"rt_s": None, "rt_from": "click", "rt_click_s": round(time.monotonic() - self.said_at, 3)} \
            if t is None else {"rt_s": round(t - self.said_at, 3), "rt_from": "speech_onset", "t_onset": round(t, 3)}
        self.log("participant_answer", answer=word, source=source, **rt)
        return word

    async def operator(self, slot, reason):
        """The operator's call on an interrupted trial (unplanned, stopped, restarted). Returns "retry" (run it again),
        "verify" (it was placed: go home and look again) or None (skip / accept ended it); abort raises Aborted."""
        while True:
            placing = self.moving == "place"
            options = ("retry", "skip", "abort") + (("accept",) if self.placed or placing else ()) \
                + (("open gripper",) if self.held else ())
            if placing:
                note = (f" The place may have finished: check the gripper and {self.target} spot {self.spot + 1}."
                        " Accept if the piece is there; to retry, put it back in its slot.")
            elif self.moving == "pick":
                note = " The piece may be in the gripper: check. To retry, put it back in its slot."
            elif self.held:
                note = (" The piece is in the gripper: hold it and press open gripper (or take it out by hand)."
                        " To retry, put it back in its slot.")
            else:
                note = ""
            choice, _, _ = await self._wait("operator", options, f"{reason}{note} Fix the setup, then:")
            self.log("operator_override", choice=choice)
            if choice != "open gripper":
                break
            try:
                await self.move("open")
            except Unplanned as e:
                reason = f"UNPLANNED: {e}."
        color = self.sc["slots"][slot][0]
        if choice == "abort":
            self.log("trial_end", aborted=True, step=self.step, moving=self.moving)
            self.in_trial = False
            raise Aborted
        if choice == "accept":  # the operator saw the piece in the zone
            self.log("trial_end", placed=self.placed or self.target, correct=(self.placed or self.target) == color,
                     operator_verified=True)
            self.in_trial = False
            self.held = False
            return None
        # skip / retry: the operator has dealt with the gripper (the prompt said so)
        self.held = False
        if choice == "skip":  # step / moving: how far it got, i.e. what the participant saw
            self.log("trial_end", skipped=True, step=self.step, moving=self.moving)
            self.in_trial = False
            return None
        self.log.attempt += 1
        if self.placed:
            return "verify"
        if self.spot is not None:  # reserved, but the operator put the piece back: the spot is free again
            self.release()
        return "retry"

    def reserve(self, zone):
        self.target, self.spot = zone, self.used[zone]
        self.used[zone] += 1
        self.log("spot", action="reserve", zone=zone, index=self.spot)

    def release(self):
        self.used[self.target] -= 1
        self.log("spot", action="release", zone=self.target, index=self.spot)
        self.target = self.spot = None

    # --- output ---
    async def say(self, key, **kw):
        """Speak a line. A question that couldn't be spoken is unplanned (unless the voice is off on purpose):
        an answer to a question the participant never heard isn't data."""
        text = SAY[key].format(**kw)
        argv, via, asset = self.voice.how(key, kw.get("color")) if self.audio else (None, None, None)
        spoken = False
        self.on_say[0]()
        try:
            if argv:
                proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.DEVNULL,
                                                            stderr=asyncio.subprocess.DEVNULL)
                try:
                    code = await proc.wait()
                except asyncio.CancelledError:  # STOP: the robot falls silent too
                    proc.kill()
                    await proc.wait()
                    self.log("robot_say", key=key, text=text, spoken=False, interrupted=True, via=via, audio=asset)
                    raise
                spoken = code == 0
                if not spoken:
                    self.voice_problem = f"{Path(argv[0]).name} exited with {code} on {text!r}"
        finally:
            self.said_at = time.monotonic()
            self.on_say[1]()
        self.log("robot_say", key=key, text=text, spoken=spoken, via=via, audio=asset)  # t_mono: end of playback
        if self.audio and not spoken and key != "done":
            raise Unplanned(f'the robot\'s question "{text}" was not spoken ({self.voice_problem})')

    async def move(self, op, x=None, y=None):
        """Arm op at table (x, y) mm, sent with the robot-frame height of the table there. Any failure is unplanned."""
        rx = ry = rz = None
        if x is not None:
            rx, ry, rz = robot_frame.to_robot(self.sc["T"], (x, y, 0.0))
        self.log("arm_cmd", op=op, **({} if x is None else {"table": [round(x, 1), round(y, 1)],
                                                             "robot": [round(rx, 1), round(ry, 1), round(rz, 1)]}))
        self.at_home, self.moving = False, op  # until a home finishes: a failed or stopped op may leave the arm anywhere
        if op == "pick":
            self.held = True                   # may be in the gripper from now on: not just after a pick finished
        try:
            await self.arm.do(op, rx, ry, rz)
        except Exception as e:  # CancelledError (STOP) isn't an Exception: it passes
            raise Unplanned(f"arm {op}: {e}") from None
        self.log("arm_done", op=op)
        self.step, self.at_home, self.moving = op, op == "home", None
        self.held = op in ("pick", "show") or (self.held and op == "home")
        self.vision.after(op, x, y)

    async def detect(self, tag):
        trial, attempt = self.log.trial, self.log.attempt
        job = asyncio.ensure_future(asyncio.to_thread(self.vision.detect))
        try:
            found, pair = await asyncio.shield(job)
        except asyncio.CancelledError:  # STOP: the detection still finishes; keep what it saw
            job.add_done_callback(lambda j: j.cancelled() or j.exception() or
                                  self.seen(f"{tag}_stopped", *j.result(), trial, attempt))
            raise
        await asyncio.to_thread(self.log.images, pair, tag, trial, attempt)
        self.seen(tag, found, None, trial, attempt)
        return found

    def seen(self, tag, found, pair, trial, attempt):
        self.log("vision", trial=trial, tag=tag, pieces=[[c, round(x, 1), round(y, 1)] for c, x, y in found])
        if pair is not None:
            asyncio.get_running_loop().run_in_executor(None, self.log.images, pair, tag, trial, attempt)

    # --- the experiment ---
    async def trial(self, slot, behavior, verify_only=False):
        """One attempt. verify_only: the piece was placed but not confirmed, so only home and look again."""
        color, sx, sy = self.sc["slots"][slot]
        self.step = self.moving = None
        self.log("trial_start", slot=slot, color=color, **behavior, **({"verify_only": True} if verify_only else {}))
        self.in_trial = True
        if not self.at_home:  # first trial, or after a stop / failure: the arm may be anywhere, maybe before a camera
            await self.move("home")
            self.step = None  # nothing of the trial itself has happened yet
        if not verify_only:
            p = nearest(await self.detect("before"), sx, sy)
            if p is None or p[0] != color:
                raise Unplanned(f"slot {slot}: expected {color}, sees {p[0] if p else 'nothing'}")
            await self.move("pick", p[1], p[2])
            target = behavior["put_to"] if behavior["behavior"] == "mistake" else color
            if behavior["behavior"] == "ask":
                await self.move("show")
                await self.say("ask", color=color)
                if await self.ask(("yes", "no"), color) == "no":
                    await self.say("which")
                    target = await self.ask(COLORS)
            if self.used[target] >= len(self.sc["zones"][target]):
                raise Unplanned(f"{target} zone is full")
            self.reserve(target)   # before the place: a failed or stopped place may still have put the piece there
            await self.move("place", *self.sc["zones"][target][self.spot])
            self.placed = target
            await self.move("home")
        tx, ty = self.sc["zones"][self.target][self.spot]
        p = nearest(await self.detect("after"), tx, ty)
        if p is None or p[0] != color:
            raise Unplanned(f"verify: no {color} piece at {self.target} spot {self.spot + 1} ({tx:.0f}, {ty:.0f})")
        self.log("trial_end", placed=self.target, correct=self.target == color)
        self.in_trial = False

    async def trials(self, first, interrupted):
        """Trials first.. to the end. interrupted: trial `first` was cut off (STOP, restart); the operator decides."""
        sc = self.sc
        for i in range(first, len(sc["order"]) + 1):
            slot = sc["order"][i - 1]
            behavior = sc["script"].get(i, {"behavior": "correct"})
            self.log.trial, self.status = i, f"trial {i}/{len(sc['order'])}: {slot}"
            # an interrupted but never begun trial (crashed or stopped between trials) just runs it normally
            if i == first and interrupted and self.in_trial:
                go = await self.operator(slot, self.interrupted)
            else:
                self.log.attempt = 1
                self.new_trial()
                go = "retry"
            while go:
                try:
                    await self.trial(slot, behavior, verify_only=go == "verify")
                    break
                except Unplanned as e:
                    self.log("unplanned", reason=str(e), step=self.step, moving=self.moving)
                    go = await self.operator(slot, f"UNPLANNED: {e}.")

    async def run(self, first=1, interrupted=False):
        """The session, or after a resume its rest. session_end once, when it's done, aborted or broke; a stop
        (task cancelled) leaves it to be resumed."""
        self.state, end = "running", "error"
        try:
            if not interrupted:
                self.log("session_start", order=self.sc["order"], layout=self.sc["layout"],
                         voice_problem=self.voice_problem, voice_stand_in=self.voice_stand_in,
                         script={str(k): v for k, v in self.sc["script"].items()}, **self.info)
            await self.trials(first, interrupted)
            self.log.trial = 0
            await self.say("done")
            end = "done"
        except Aborted:
            end = "aborted"
        except asyncio.CancelledError:
            end = None
            raise
        except Exception:   # a bug or a broken link: the session can't go on, and the log says why
            self.log("error", traceback=traceback.format_exc())
        finally:
            if end:
                self.state = self.status = end
                self.log("session_end", status=end)

    def start(self):
        if self.state == "ready":
            self.state = "running"  # now, not when the task first runs: a second click must not start a second one
            self.stop_problem = None    # a STOP pressed before the session (link down): the operator has moved on
            self.task = asyncio.create_task(self.run())

    def stop(self):
        sent = self.arm.stop()
        self.interrupted = "STOPPED by the operator."
        self.log("stop", source="operator", step=self.step, moving=self.moving, sent=sent)
        if self.task and not self.task.done():
            self.task.cancel()
            self.state = "stopped"
            # the stop isn't real until arm_node confirms the cancellation; until it answers, resume waits
            self.status = f"STOPPED at trial {self.log.trial}" + (" (confirming at the arm…)" if sent else "")
        if sent:
            self.confirming = asyncio.create_task(self.confirm_stop())
        else:
            self.stop_not_confirmed("stop was not sent")     # the red link-warning situation already says so

    def stop_not_confirmed(self, reason):
        """The operator must not trust the STOP: the arm may still be moving. Only Resume (once the confirm
        answer came or timed out) carries on, logging the problem."""
        self.stop_problem = reason
        self.status = f"STOP NOT CONFIRMED at trial {self.log.trial}: {reason}"

    async def confirm_stop(self):
        """The node's answer to the last stop: "done" (cancelled, or already finished, or nothing ran) or a
        failure with the reason. Logged either way; a broken confirm path counts as not confirmed."""
        try:
            confirmed, reason = await self.arm.stop_confirm()
        except Exception as e:
            confirmed, reason = False, f"stop confirm failed: {e}"
        self.log("stop_confirm", confirmed=confirmed, reason=reason)
        if confirmed:
            self.stop_problem = None
            if self.state == "stopped":
                self.status = f"STOPPED at trial {self.log.trial}"
        else:
            self.stop_not_confirmed(reason)

    def resume(self):
        if self.state == "stopped":
            if self.confirming and not self.confirming.done():
                return              # the stop is still being confirmed: don't continue on an unverified arm
            self.state = "running"
            # the operator pressed Resume over a failed confirm: the log says so
            self.log("resume", **({"stop_problem": self.stop_problem} if self.stop_problem else {}))
            self.stop_problem = None
            # trial 0: stopped after the last trial, while saying "done"
            first = self.log.trial or len(self.sc["order"]) + 1
            self.task = asyncio.create_task(self.run(first, interrupted=True))


class Speech:
    """Spoken answers: speech.py's listener (in its own process, so an engine crash leaves the panel buttons working)
    -> the runner, on the panel's event loop. The listener is muted while the robot speaks. Only a running session is
    recorded: every utterance is logged with its audio, under the trial and question that were current when it
    began. One that began after the question ended and clearly matches the awaited answers answers it (auto), or
    waits for the operator's confirm. An answer the operator clicks is timed from the participant's speech onset when
    there was one (a transcribed utterance, not a cough-length blip), else it logs the click time only."""

    WATCH_S = 0.5

    def __init__(self, runner, language="en", auto=True, device=None):
        import speech
        self.match, self.runner, self.auto = speech.match, runner, auto
        self.heard = None           # the last utterance, for the panel: {"text", "word", "clipped", "t", "said_at", ...}
        self.last_onset = None      # the latest VAD onset, for the panel's "speaking…"
        self.answer_onsets = []     # onsets of utterances that can be answers: what a click is timed from
        self.questions = []         # (said_at, trial) of every line the robot finished: an utterance belongs to the
        self.warning = None         # last one before its onset; the listener's warning, logged when it changes
        self.n = 0
        self.lp = speech.ListenerProcess(language, device=device, on_utterance=self._from_thread(self.utterance),
                                         on_speech_start=self._from_thread(self.onset),
                                         on_error=self._from_thread(self.error))
        runner.on_say = (self.lp.mute, self.question_ended)

    def _from_thread(self, fn):  # the listener calls back on its reader thread; the runner lives on the loop
        return lambda *a: self.loop.call_soon_threadsafe(fn, *a)

    def start(self):
        self.loop = asyncio.get_running_loop()
        self.lp.start()
        self.loop.create_task(self.watch())

    async def restart(self):
        await asyncio.to_thread(self.lp.stop)   # joins the child: seconds, never on the loop (STOP must get through)
        self.lp.start()

    async def watch(self):
        while True:
            if self.lp.warning != self.warning:
                self.warning = self.lp.warning
                self.runner.log("speech_warning", warning=self.warning)
            await asyncio.sleep(self.WATCH_S)

    def question_ended(self):
        self.lp.unmute()
        self.questions = self.questions[-20:] + [(self.runner.said_at, self.runner.log.trial)]

    def onset(self, t):
        self.last_onset = t

    def error(self, msg):
        self.runner.log("speech_error", error=msg)

    def awaited(self):
        p = self.runner.prompt
        return p["options"] if p and p["kind"] == "answer" else None

    def utterance(self, u):
        r = self.runner
        if r.state != "running":    # setup talk before Start, or after the session: not participant data
            return
        said, trial = next(((s, t) for s, t in reversed(self.questions) if s <= u.t_onset), (None, r.log.trial))
        options = self.awaited()
        current = options is not None and said is not None and said == r.said_at   # an answer to this question
        word = None
        if current and not u.error:
            word = self.match(u.text, options, u.language, r.prompt.get("color"))
            if not u.clipped:
                self.answer_onsets = self.answer_onsets[-20:] + [u.t_onset]
        self.n += 1
        wav = r.log.media / "speech" / f"trial{trial:02d}_u{self.n:03d}.wav"
        r.log("speech", trial=trial, text=u.text, match=word, awaited=list(options) if current else None,
              language=u.language, clipped=u.clipped, clock=u.clock, t_onset=round(u.t_onset, 3),
              t_end=round(u.t_end, 3), t_audio=round(u.t_audio, 3), asr_s=round(u.asr_s, 3),
              audio=str(wav.relative_to(r.log.path.parent)), **({"asr_error": u.error} if u.error else {}))
        self.loop.run_in_executor(None, self.save, wav, u.audio)
        if said is not None and said == r.said_at:
            self.heard = {"text": u.text, "word": None if u.clipped else word, "clipped": u.clipped,
                          "error": u.error, "t": u.t_onset, "said_at": said, "answered": False}
            if word and self.auto and not u.clipped:
                self.confirm()

    def save(self, wav, audio):  # worker thread
        try:
            write_wav(wav, audio)
        except Exception as e:
            msg = f"saving {wav.name}: {e}"   # `e` is gone once the except block ends
            self.loop.call_soon_threadsafe(lambda: self.runner.log("speech_error", error=msg))

    def confirm(self):
        """Answer with the heard word (auto, or the operator's confirm in confirm mode)."""
        h = self.heard
        if h and h["word"] and not h["answered"] and h["said_at"] == self.runner.said_at and self.awaited():
            h["answered"] = True
            self.runner.answer(h["word"], "speech" if self.auto else "speech_confirmed", h["t"])

    def operator_answer(self, word):
        """A panel button: the operator's call, timed from the participant's speech onset when there was one."""
        said = self.runner.said_at
        t = next((t for t in self.answer_onsets if said is not None and t >= said), None)
        self.runner.answer(word, "panel", t)


def write_wav(path, audio):
    """16 kHz float32 mono -> 16-bit WAV. Never overwrites: a file that exists belongs to an earlier record."""
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f, wave.open(f, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


SPEECH_COLORS = {"ready": "positive", "loading": "warning", "error": "negative", "crashed": "negative",
                 "stopped": "grey"}
STATE_COLORS = {"ready": "grey", "running": "primary", "stopped": "negative", "done": "positive",
                "aborted": "negative", "error": "negative"}
OPERATOR_OPTIONS = ("retry", "skip", "abort", "accept", "open gripper")


def panel(runner, title, host, speech=None):
    from nicegui import app, ui

    def startable():
        return runner.state == "ready" and not (speech and speech.lp.state == "loading")

    @ui.page("/")
    def page():
        ui.query("body").style("font-size: 15px")
        # title and controls stick to the top, above everything that appears and disappears: STOP never moves or
        # scrolls away
        with ui.column().classes("sticky top-0 z-10 w-full gap-2 pb-2").style("background: #121212"):
            with ui.row().classes("w-full items-center"):
                ui.label(title).classes("text-2xl font-medium")
                ui.space()
                state = ui.badge().classes("text-base px-3 py-1")
            with ui.row().classes("items-center gap-3"):
                start = ui.button("Start", on_click=runner.start)
                ui.button("Resume", on_click=runner.resume).bind_visibility_from(runner, "state",
                                                                                  lambda s: s == "stopped")
                ui.button("STOP", color="red", on_click=runner.stop).classes("text-lg px-8")
                status = ui.label().classes("text-lg ml-4")
            link = ui.label("ARM LINK DOWN: STOP can't reach the arm. Use the power / torque switch.") \
                .classes("text-lg font-medium text-red-400")
            stoprisk = ui.label("STOP NOT CONFIRMED: the arm may still be moving. Use the power / torque switch.") \
                .classes("text-lg font-medium text-red-400")

        with ui.row().classes("w-full gap-4 items-stretch"):
            with ui.card().classes("p-3 w-96"):
                ui.label("Robot voice").classes("text-sm opacity-70")
                voice = ui.label()
            with ui.card().classes("p-3 grow"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.label("Speech").classes("text-sm opacity-70")
                    s_state = ui.badge()
                    s_detail = ui.label().classes("text-sm")
                    ui.space()
                    if speech:
                        ui.toggle({"en": "EN", "hu": "HU"}, value=speech.lp.language,
                                  on_change=lambda e: setattr(speech.lp, "language", e.value))
                        ui.switch("auto-answer", value=speech.auto,
                                  on_change=lambda e: setattr(speech, "auto", e.value))
                        restart = ui.button("Restart", on_click=speech.restart).props("flat dense")
                with ui.row().classes("w-full items-center gap-3"):
                    meter = ui.linear_progress(value=0, show_value=False, size="10px").classes("grow")
                    speaking = ui.label().classes("text-sm w-24")

        with ui.card().classes("w-full p-4 min-h-40"):
            prompt = ui.label().classes("text-2xl text-orange-400")
            heard = ui.label().classes("text-lg")
            with ui.row().classes("gap-3") as answers:
                answer_buttons = {w: ui.button(w, on_click=lambda w=w: speech.operator_answer(w) if speech
                                               else runner.answer(w)).classes("text-lg px-6")
                                  for w in ("yes", "no", *COLORS)}
                confirm = ui.button(on_click=lambda: speech.confirm(), color="positive").classes("text-lg px-6")
            with ui.row().classes("gap-3") as ops:
                op_buttons = {c: ui.button(c, on_click=lambda c=c: runner.choose(c)).classes("text-lg px-6")
                              for c in OPERATOR_OPTIONS}
        log = ui.log(max_lines=300).classes("w-full h-80")
        shown = {"n": 0}

        def tick():
            p = runner.prompt
            state.text = runner.state
            state.props(f"color={STATE_COLORS.get(runner.state, 'grey')}")
            start.enabled = startable()
            link.visible = not getattr(runner.arm, "connected", True)
            stoprisk.visible = runner.stop_problem is not None
            status.text = runner.status
            prompt.text = f'{p["text"]}  ({time.monotonic() - p["since"]:.0f} s)' if p else ""
            problems = runner.voice_problem.split("; ") if runner.voice_problem else []   # the full list is in the log
            if problems:
                voice.text = problems[0] + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
            elif runner.voice_stand_in:
                voice.text = f"macOS `say` stands in for {len(runner.voice_stand_in)} line(s): record the WAVs"
            else:
                voice.text = "all lines play from recordings"
            voice.classes(replace="text-red-400" if problems else "text-amber-400" if runner.voice_stand_in
                          else "text-green-400")
            asking = bool(p and p["kind"] == "answer")
            answers.visible, ops.visible = asking, bool(p and p["kind"] == "operator")
            for w, b in answer_buttons.items():
                b.visible = asking and w in p["options"]
            for c, b in op_buttons.items():
                b.visible = bool(p and c in p["options"])
            if speech:
                lp, h = speech.lp, speech.heard
                s_state.text = lp.state
                s_state.props(f"color={SPEECH_COLORS.get(lp.state, 'grey')}")
                s_detail.text = lp.warning or lp.detail
                s_detail.classes(replace="text-red-400" if lp.warning else "opacity-80")
                restart.visible = lp.state in ("error", "crashed", "stopped") or bool(lp.warning)
                meter.value = min(max((lp.level_db + 60) / 60, 0), 1) if lp.state == "ready" else 0
                recent = speech.last_onset is not None and time.monotonic() - speech.last_onset < 1.0
                speaking.text = "speaking…" if recent else ""
                current = h and h["said_at"] == runner.said_at
                heard.text = "" if not current else f'heard “{h["text"]}” → ' + (
                    "recognition failed, pick one" if h["error"] else
                    "began during the question, pick one" if h["clipped"] else
                    f'{h["word"]} (answered)' if h["answered"] else
                    f'{h["word"]}: confirm?' if h["word"] else "no clear answer, pick one")
                confirm.visible = bool(asking and current and h["word"] and not h["answered"])
                confirm.text = f'Confirm “{h["word"]}”' if confirm.visible else ""
            else:
                s_state.text, s_detail.text, confirm.visible = "off", "answers through the buttons only", False
            for line in runner.log.tail[shown["n"]:]:
                log.push(line)
            shown["n"] = len(runner.log.tail)

        ui.timer(0.2, tick)

    def shutdown():
        runner.arm.stop()
        runner.log("shutdown", state=runner.state)   # an unfinished session is resumed on the next start
    app.on_shutdown(shutdown)
    if speech:
        app.on_startup(speech.start)
        app.on_shutdown(speech.lp.stop)
    ui.run(title=title, host=host, port=PORT, reload=False, dark=True, show=False)


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        world = SimWorld()

        def run(condition, auto, setup=lambda r: None, arm=None):
            r = Runner(f"T{len(os.listdir(tmp))}", condition, arm or FakeArm(), world, tmp, auto=auto, audio=False)
            world.reset()
            setup(r)
            asyncio.run(r.run())
            return r, events(r)

        def events(r):
            return [json.loads(line) for line in r.log.path.read_text().splitlines()]

        def last(events, name):
            return [e["data"] for e in events if e["event"] == name]

        for cond in ("correct", "mistake", "ask_low", "ask_high"):
            r, ev = run(cond, lambda kind, opts: "yes")
            ends, script = last(ev, "trial_end"), r.sc["script"]
            assert len(ends) == len(r.sc["order"]) and last(ev, "session_end") == [{"status": "done"}], cond
            assert [i for i, e in enumerate(ends, 1) if not e["correct"]] == \
                   [i for i, b in script.items() if b["behavior"] == "mistake"], cond
            asks = [i for i, b in script.items() if b["behavior"] == "ask"]
            assert len(last(ev, "participant_answer")) == len(asks) == sum(e["key"] == "ask" for e in last(ev, "robot_say"))
            assert not last(ev, "unplanned"), last(ev, "unplanned")
            assert all(a["rt_s"] is None and a["rt_from"] == "click" for a in last(ev, "participant_answer"))
            print(f"  {cond:9s} {len(ends)} trials, {len(asks)} asks, "
                  f"{sum(not e['correct'] for e in ends)} mistakes: ok")

        # audio off: nothing spoken, and the panel says so
        assert r.voice_problem == "audio off" and not any(e["spoken"] or e["via"] for e in last(ev, "robot_say"))

        # "no" to a question: the robot asks which colour and puts it there
        answers = iter(["no", "green"])
        _, ev = run("ask_low", lambda kind, opts: next(answers, "yes"))
        assert [e["answer"] for e in last(ev, "participant_answer")] == ["no", "green", "yes"]
        assert sum(not e["correct"] for e in last(ev, "trial_end")) == 1

        # a piece missing from its slot: unplanned, the operator skips it, the rest still runs
        def remove_first(r):
            c, x, y = r.sc["slots"][r.sc["order"][0]]
            world.found.remove(nearest(world.found, x, y))
        r, ev = run("correct", lambda kind, opts: "skip" if kind == "operator" else "yes", remove_first)
        assert len(last(ev, "unplanned")) == 1 and last(ev, "trial_end")[0] == {"skipped": True, "step": None, "moving": None}
        assert len(last(ev, "trial_end")) == len(r.sc["order"])

        # the place fails: it may have put the piece down (accept), the piece may be held (open gripper); skip logs
        # how far it got
        class PlaceFailsOnce(FakeArm):
            failed = False

            async def do(self, op, x=None, y=None, z=None):
                if op == "place" and not self.failed:
                    self.failed = True
                    raise RuntimeError("controller error")
        offered = []
        r, ev = run("correct", lambda kind, opts: offered.append(opts) or "skip", arm=PlaceFailsOnce())
        assert offered == [("retry", "skip", "abort", "accept", "open gripper")], offered
        assert last(ev, "trial_end")[0] == {"skipped": True, "step": "pick", "moving": "place"}

        # placed, but the camera misses it: the operator accepts, the trial counts as done
        def camera_misses_first_place(op, x, y):
            if op == "place" and not missed:
                missed.append(1)
                world.held = None
            else:
                SimWorld.after(world, op, x, y)
        missed, world.after = [], camera_misses_first_place
        try:
            r, ev = run("correct", lambda kind, opts: "accept" if "accept" in opts else "yes")
        finally:
            del world.after
        assert last(ev, "unplanned")[0]["step"] == "home"
        assert last(ev, "trial_end")[0] == {"placed": "red", "correct": True, "operator_verified": True}
        assert len(last(ev, "trial_end")) == len(r.sc["order"]) and not any("skipped" in e for e in last(ev, "trial_end"))

        # every pick / place carries the robot-frame z of the table there: tilted and raised robot frame
        class Recording(FakeArm):
            def __init__(self):
                self.cmds = []

            async def do(self, op, x=None, y=None, z=None):
                self.cmds.append((op, x, y, z))
                await asyncio.sleep(0)
        tilt, T = 0.03, None
        arm = Recording()

        def tilted(r):
            nonlocal T
            T = r.sc["T"].astype(float)
            T[:3, :3] = T[:3, :3] @ np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
            T[2, 3] = 12.0
            r.sc["T"] = T
        _, ev = run("correct", lambda kind, opts: "yes", tilted, arm)
        sent = [c for c in arm.cmds if c[0] in ("pick", "place")]
        cmds = [e for e in last(ev, "arm_cmd") if e["op"] in ("pick", "place")]
        assert len(sent) == len(cmds) == 24 and len({round(c[3]) for c in sent}) > 1, sent
        for (_, rx, ry, rz), e in zip(sent, cmds):
            want = robot_frame.to_robot(T, (*e["table"], 0.0))
            assert np.allclose((rx, ry, rz), want, atol=0.2) and np.allclose(e["robot"], want, atol=0.2), (e, rz)
        assert all(c[1:] == (None, None, None) for c in arm.cmds if c[0] in ("home", "show"))

        # the scenario check names what's wrong: unreachable spot, too small a zone, crowded spots, low travel
        sc = yaml.safe_load((ROOT / "scenarios" / "ask_high.yaml").read_text())
        lay = yaml.safe_load((ROOT / sc["layout"]).read_text())
        T0 = robot_frame.load(ROOT / lay["robot_frame"])
        check_scenario(sc, lay, T0)

        def rejects(change, expect, sc=sc):
            bad = copy.deepcopy(lay)
            change(bad)
            try:
                check_scenario(sc, bad, T0)
            except AssertionError as e:
                assert expect in str(e), (expect, str(e))
                return
            raise AssertionError(f"check_scenario passed, expected: {expect}")
        rejects(lambda b: b["zones"]["red"].append([100, 30]), "red spot 10 (100, 30): out of reach")
        rejects(lambda b: b.pop("min_slot_spacing_mm"), "mm from slot")   # the synthetic slots touch
        bad_sc = copy.deepcopy(sc)
        bad_sc["script"][1] = {"behavior": "correct", "put_to": "green"}
        try:
            check_scenario(bad_sc, lay, T0)
            raise AssertionError("put_to on a correct trial passed")
        except AssertionError as e:
            assert "put_to only goes with behavior: mistake" in str(e), e
        rejects(lambda b: b["zones"].update(blue=b["zones"]["blue"][:4]), "blue zone: 4 spots, the script can need 5")
        x, y = lay["zones"]["green"][0]
        rejects(lambda b: b["zones"]["green"].insert(1, [x, y + 30]), f"green spot 1 ({x:.0f}, {y:.0f}) is 30 mm from green spot 2")
        global HOME
        home, HOME = HOME, list(ik(150.0, 0.0, 10.0))   # a home pose just over the table
        try:
            rejects(lambda b: None, "travel show -> home")
        finally:
            HOME = home

        # STOP during trial 3's place, then Resume: the operator retries it, the rest of the session runs
        class StopsOnThirdPlace(FakeArm):
            places = 0

            async def do(self, op, x=None, y=None, z=None):
                self.places += op == "place"
                if op == "place" and self.places == 3:
                    r.stop()   # the cancellation lands at the next await
                await asyncio.sleep(0)

        def put_back(kind, opts):  # the operator takes the piece out of the gripper and puts it back in its slot
            if kind == "operator" and world.held:
                _, x, y = r.sc["slots"][r.sc["order"][2]]
                world.found.append((world.held, x, y))
                world.held = None
            return "retry" if kind == "operator" else "yes"

        async def stop_and_resume():
            r.start()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
            await r.confirming          # the fake arm confirms at once; explicit for determinism
            assert r.state == "stopped" and r.log.trial == 3, (r.state, r.log.trial)
            r.resume()
            await r.task
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", StopsOnThirdPlace(), world, tmp, auto=put_back, audio=False)
        world.reset()
        asyncio.run(stop_and_resume())
        ev = events(r)
        assert r.state == "done" and last(ev, "session_end") == [{"status": "done"}]
        assert len(last(ev, "stop")) == len(last(ev, "resume")) == 1 and last(ev, "stop")[0]["step"] == "pick"
        assert last(ev, "stop")[0]["sent"] is True
        assert last(ev, "stop")[0]["moving"] == "place"
        assert len(last(ev, "trial_end")) == len(r.sc["order"]) and not last(ev, "unplanned")
        after = [e for e in ev[[e["event"] for e in ev].index("resume"):] if e["event"] in ("arm_cmd", "vision")]
        assert after[0]["event"] == "arm_cmd" and after[0]["data"]["op"] == "home" and after[0]["trial"] == 3, after[:2]
        assert ev[-2]["event"] == "robot_say" and ev[-1]["data"] == {"status": "done"}

        # a double Start runs one session
        async def twice():
            r.start()
            r.start()
            await r.task
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", FakeArm(), world, tmp, auto=lambda k, o: "yes", audio=False)
        world.reset()
        asyncio.run(twice())
        assert len(last(events(r), "session_start")) == 1 and len(last(events(r), "trial_end")) == len(r.sc["order"])

        # abort: the interrupted trial still ends in the log, then the session
        r, ev = run("correct", lambda kind, opts: "abort" if kind == "operator" else "yes", remove_first)
        assert last(ev, "trial_end") == [{"aborted": True, "step": None, "moving": None}]
        assert last(ev, "session_end") == [{"status": "aborted"}]

        # placed but the camera missed it; the operator fixes the view and retries: home and look again, no re-pick
        def fix_view(kind, opts):
            if kind == "operator":
                world.found.append(("red", *r.sc["zones"]["red"][0]))
                return "retry"
            return "yes"
        missed, world.after = [], camera_misses_first_place
        try:
            r, ev = run("correct", fix_view)
        finally:
            del world.after
        first = [e for e in ev if e["trial"] == 1]
        assert [e["data"].get("verify_only") for e in first if e["event"] == "trial_start"] == [None, True]
        assert sum(e["event"] == "arm_cmd" and e["data"]["op"] == "pick" for e in first) == 1
        assert last(ev, "trial_end")[0] == {"placed": "red", "correct": True}
        assert [e["data"]["index"] for e in ev if e["event"] == "spot" and e["data"]["zone"] == "red"] == [0, 1, 2, 3, 4]

        # the runner quits mid-session (STOP, then the process ends): the next start resumes where it stopped
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", StopsOnThirdPlace(), world, tmp, auto=put_back, audio=False)
        world.reset()

        async def quit_at_third():
            r.start()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
        asyncio.run(quit_at_third())
        name, used = r.log.participant, dict(r.used)
        r.log.f.close()
        r = Runner(name, "correct", FakeArm(), world, tmp, auto=put_back, audio=False)
        assert r.state == "stopped" and r.log.trial == 3 and r.used == used and r.moving == "place", (r.log.trial, r.used)

        async def resume():
            r.resume()
            await r.task
        asyncio.run(resume())
        ev = events(r)
        assert sorted(e["trial"] for e in ev if e["event"] == "trial_end") == list(range(1, len(r.sc["order"]) + 1))
        assert last(ev, "session_end") == [{"status": "done"}] and len({e["run"] for e in ev}) == 2

        # a stop whose confirm fails: stop_confirm says so, the status and the panel flag carry the warning, and
        # Resume carries on, logging stop_problem in the resume event and clearing it
        class StopFailsConfirm(FakeArm):
            def __init__(self):
                self.places = 0

            async def do(self, op, x=None, y=None, z=None):
                self.places += op == "place"
                if op == "place" and self.places == 3:
                    r.stop()   # the cancellation lands at the next await
                await asyncio.sleep(0)

            async def stop_confirm(self):
                return False, "the controller would not cancel the goal"

        async def stop_fail_resume():
            r.start()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
            await r.confirming
            assert r.state == "stopped" and r.stop_problem, (r.state, r.stop_problem)
            assert "STOP NOT CONFIRMED" in r.status, r.status
            c = last(events(r), "stop_confirm")
            assert c and c[-1] == {"confirmed": False, "reason": "the controller would not cancel the goal"}, c
            r.resume()
            await r.task
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", StopFailsConfirm(), world, tmp, auto=put_back, audio=False)
        world.reset()
        asyncio.run(stop_fail_resume())
        ev = events(r)
        assert r.state == "done" and r.stop_problem is None and last(ev, "session_end") == [{"status": "done"}]
        assert last(ev, "resume") == [{"stop_problem": "the controller would not cancel the goal"}]
        assert len(last(ev, "trial_end")) == len(r.sc["order"])

        # resume does nothing at all while the stop's confirmation is still pending
        class StopsAndWaitsConfirm(FakeArm):
            def __init__(self):
                self.gate, self.places = None, 0

            async def do(self, op, x=None, y=None, z=None):
                self.places += op == "place"
                if op == "place" and self.places == 3:
                    r.stop()   # the cancellation lands at the next await
                await asyncio.sleep(0)

            async def stop_confirm(self):
                await self.gate.wait()
                self.gate = None
                return True, None

        async def resume_blocked():
            gate = asyncio.Event()
            r.arm.gate = gate
            r.start()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
            assert r.state == "stopped" and r.confirming and not r.confirming.done()
            n = len(r.log.tail)
            r.resume()                     # must be a no-op: the stop isn't verified yet
            assert r.state == "stopped" and len(r.log.tail) == n and r.task.cancelled()
            gate.set()
            await r.confirming             # confirmed now: resume (the same button) may proceed
            assert r.stop_problem is None
            before = len(r.log.tail)
            r.resume()
            await r.task
            assert len(r.log.tail) > before
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", StopsAndWaitsConfirm(), world, tmp, auto=put_back,
                   audio=False)
        world.reset()
        asyncio.run(resume_blocked())
        assert r.state == "done" and last(events(r), "session_end") == [{"status": "done"}]

        # a pick that fails after dispatch: the piece may be in the gripper, so the prompt offers open gripper
        # (live), and the same after a replay of a run stopped mid-pick
        class PickFails(FakeArm):
            async def do(self, op, x=None, y=None, z=None):
                if op == "pick":
                    raise RuntimeError("controller error")
                await asyncio.sleep(0)
        offered = []
        r, ev = run("correct", lambda k, o: offered.append(o) or ("open gripper" if "open gripper" in o else "skip"),
                    arm=PickFails())
        assert offered and offered[0] == ("retry", "skip", "abort", "open gripper"), offered
        # every failed pick offers it once; once it was used, the next prompt has nothing to open
        assert len(offered) == 2 * len(r.sc["order"]) and offered[-1] == ("retry", "skip", "abort"), offered
        assert all(("open gripper" in o) == (i % 2 == 0) for i, o in enumerate(offered)), offered
        assert last(ev, "trial_end")[0] == {"skipped": True, "step": "open", "moving": None}
        assert r.held is False

        class StopsAtPick(FakeArm):
            async def do(self, op, x=None, y=None, z=None):
                if op == "pick":
                    r.stop()   # before arm_done: the log says pick was sent, not that it finished
                await asyncio.sleep(0)

        def retry_answers(kind, opts):
            if kind == "operator":
                offered.append(opts)
                return "retry"
            return "yes"
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", StopsAtPick(), world, tmp, auto=retry_answers,
                   audio=False)
        world.reset()

        async def quit_at_pick():
            r.start()
            try:
                await r.task
            except asyncio.CancelledError:
                pass
        asyncio.run(quit_at_pick())
        name = r.log.participant
        r.log.f.close()
        r = Runner(name, "correct", FakeArm(), world, tmp, auto=retry_answers, audio=False)
        assert r.state == "stopped" and r.held and r.moving == "pick", (r.state, r.held, r.moving)
        asyncio.run(resume())
        ev = events(r)
        # a trial_start was logged but no trial_end: the operator is asked about the interrupted trial
        assert offered and "open gripper" in offered[-1], offered
        assert last(ev, "operator_override")[0]["choice"] == "retry"
        assert r.state == "done" and last(ev, "session_end") == [{"status": "done"}]

        # a crash between trials (trial_end logged, no trial_start after it): resume just runs the next trial,
        # with attempt 1, no operator prompt and no skipped record
        def half_log(participant, recs):
            """An earlier run_id's session: session_start and recs, no session_end (the process died)."""
            run_id = "half"
            path = Path(tmp) / f"{participant}_correct.jsonl"
            base = {"t_wall": "2024-01-01T00:00:00.000+00:00", "t_mono": 1.0, "participant": participant,
                    "condition": "correct", "run": run_id}
            path.write_text("\n".join(json.dumps(base | {"trial": r["t"], "attempt": r.get("a", 1),
                                                        "event": r["event"], "data": r.get("d", {})})
                                      for r in [dict(event="session_start", t=0)] + recs) + "\n")
            return run_id
        name = f"T{len(os.listdir(tmp))}"
        half_log(name, [dict(event="trial_start", t=1, d={"slot": r.sc["order"][0], "color": "red"}),
                        dict(event="arm_cmd", t=1, d={"op": "pick"}),
                        dict(event="arm_done", t=1, d={"op": "pick"}),
                        dict(event="vision", t=1, d={"tag": "before", "pieces": []}),
                        dict(event="trial_end", t=1, d={"placed": "red", "correct": True})])
        world.reset()
        r = Runner(name, "correct", FakeArm(), world, tmp, auto=lambda k, o: "yes", audio=False)
        assert r.state == "stopped" and r.log.trial == 2 and not r.in_trial, (r.state, r.log.trial)
        asyncio.run(resume())
        ev = events(r)
        resumed = [e for e in ev if e["run"] != "half"]
        assert last(resumed, "resume") == [{}]
        assert not last(ev, "operator_override") and not any(t.get("skipped") for t in last(ev, "trial_end"))
        n = len(r.sc["order"])
        starts = [e for e in ev if e["event"] == "trial_start"]
        assert [e["trial"] for e in starts] == list(range(1, n + 1)) and [e["attempt"] for e in starts] == [1] * n
        assert r.state == "done" and last(resumed, "session_end") == [{"status": "done"}]

        # the answer to an earlier stop (a timed-out command's) must not confirm the operator's later STOP
        published = []

        class RosStub:      # RosArm's roslibpy side: connected, and publish just records
            is_connected, Message, publish = True, staticmethod(dict), staticmethod(published.append)

        async def latest_stop_counts():
            arm = RosArm.__new__(RosArm)
            arm.ros = arm.cmd = arm.roslibpy = RosStub
            arm.waiting, arm.n, arm.last_stop, arm.nonce = {}, 0, None, "test"
            assert arm.stop() and arm.stop()
            first, second = (json.loads(m["data"])["id"] for m in published)
            arm.on_status({"data": json.dumps({"id": first, "state": "done"})})
            arm.on_status({"data": json.dumps({"id": second, "state": "failed", "reason": "rejected"})})
            assert await arm.stop_confirm() == (False, "rejected") and not arm.waiting
        asyncio.run(latest_stop_counts())

        # with the voice on, a question nobody could hear is unplanned, not an answer
        which = shutil.which
        shutil.which = lambda name: None
        try:
            r, ev = run("ask_low", lambda kind, opts: "skip" if kind == "operator" else "yes")
        finally:
            shutil.which = which
        assert len(last(ev, "unplanned")) == 0   # audio=False in run(): silent on purpose
        r = Runner(f"T{len(os.listdir(tmp))}", "ask_low", FakeArm(), world, tmp, audio=True,
                   auto=lambda kind, opts: "skip" if kind == "operator" else "yes")
        world.reset()
        shutil.which = lambda name: None
        try:
            asyncio.run(r.run())
        finally:
            shutil.which = which
        ev = events(r)
        assert r.voice_problem and [u["reason"].startswith('the robot\'s question "Is this') for u in
                                    last(ev, "unplanned")] == [True, True] and not last(ev, "participant_answer")

        # voice: a recorded line plays through the player; a line with no WAV and no `say` is a problem
        (Path(tmp) / "audio").mkdir()
        (Path(tmp) / "audio" / "ask_red.wav").touch()
        voice = Voice([("ask", "red"), ("ask", "blue"), ("done", None)], Path(tmp) / "audio")
        which = shutil.which
        shutil.which = lambda name: "/usr/bin/true" if name == "pw-play" else None
        try:
            assert voice.how("ask", "red") == (["/usr/bin/true", str(Path(tmp) / "audio" / "ask_red.wav")], "wav",
                                               "ask_red.wav")
            assert voice.problems() == ['"Is this blue?": no audio/ask_blue.wav and no `say`',
                                        '"Done.": no audio/done.wav and no `say`'], voice.problems()
        finally:
            shutil.which = which

        # speech answers: rt_s from the speech onset; speech that began during the question is not an answer
        async def speech():
            r.said_at = time.monotonic()
            asking = asyncio.create_task(r.ask(("yes", "no")))
            await asyncio.sleep(0)
            r.answer("no", "speech", t=r.said_at - 0.5)
            r.answer("yes", "speech", t=r.said_at + 0.25)
            return await asking
        r = Runner(f"T{len(os.listdir(tmp))}", "ask_low", FakeArm(), world, tmp, audio=False)
        assert asyncio.run(speech()) == "yes"
        a = last(events(r), "participant_answer")
        assert a == [{"answer": "yes", "source": "speech", "rt_s": 0.25, "rt_from": "speech_onset",
                      "t_onset": round(r.said_at + 0.25, 3)}], a
        # the speech bridge: a clear answer after the question answers it (auto) and is logged with its audio;
        # an unclear one waits, and the operator's button gets the speech onset as its reaction time
        from speech import Utterance

        async def spoken():
            r.state, sp = "running", Speech(r)
            r.said_at = time.monotonic() - 2.0   # onsets are always in the past
            sp.loop = asyncio.get_running_loop()
            sp.question_ended()
            u = Utterance("Yes, it is.", r.said_at + 0.4, r.said_at + 1.0, 0.05, np.zeros(16000, np.float32))
            asking = asyncio.create_task(r.ask(("yes", "no"), "red"))
            await asyncio.sleep(0)
            sp.onset(u.t_onset)
            sp.utterance(u)
            first = await asking
            r.said_at = time.monotonic()
            sp.question_ended()
            asking = asyncio.create_task(r.ask(COLORS))
            await asyncio.sleep(0)
            sp.onset(r.said_at + 0.3)
            sp.utterance(Utterance("hmm", r.said_at + 0.3, r.said_at + 0.6, 0.05, np.zeros(8000, np.float32)))
            await asyncio.sleep(0)
            assert not asking.done() and sp.heard["word"] is None
            sp.operator_answer("blue")
            return first, await asking
        r = Runner(f"T{len(os.listdir(tmp))}", "ask_low", FakeArm(), world, tmp, audio=False)
        assert asyncio.run(spoken()) == ("yes", "blue")
        ev = events(r)
        a, s = last(ev, "participant_answer"), last(ev, "speech")
        assert [(x["answer"], x["source"], x["rt_s"]) for x in a] == [("yes", "speech", 0.4), ("blue", "panel", 0.3)], a
        assert [x["match"] for x in s] == ["yes", None] and (r.log.path.parent / s[0]["audio"]).stat().st_size > 32000
    print("runner self-test ok")


def main():
    os.chdir(ROOT)  # robot_frame uses paths relative to the repo
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant")
    ap.add_argument("--block", type=int, default=1, help="which of the participant's conditions (1-based)")
    ap.add_argument("--vision", choices=["synthetic", "cameras"], help="required: synthetic images (tracked from the "
                    "arm's moves) or the real cameras")
    ap.add_argument("--arm", choices=["ros", "fake"], default="ros")
    ap.add_argument("--host", default="localhost", help="rosbridge host (port 9090)")
    ap.add_argument("--panel-host", default="127.0.0.1", help="operator panel address (the panel has no "
                                                             "authentication: only use a loopback host unless the "
                                                             "lab's network is one you trust with the controls)")
    ap.add_argument("--lang", choices=["en", "hu"], default="en", help="language the participant answers in")
    ap.add_argument("--no-speech", action="store_true", help="answers through the panel buttons only")
    ap.add_argument("--mic", type=int, help="input device for spoken answers (speech.py --list-devices)")
    ap.add_argument("--no-voice", action="store_true", help="the robot stays silent (development, or the experimenter "
                                                             "reads the questions); logged")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.participant or not args.vision:
        ap.error("--participant and --vision are required")
    with open(ROOT / "participants.csv") as f:
        rows = {r["participant"]: r["conditions"].split() for r in csv.DictReader(f)}
    if args.participant not in rows:
        sys.exit(f"{args.participant} is not in participants.csv")
    conditions = rows[args.participant]
    if not 1 <= args.block <= len(conditions):
        sys.exit(f"{args.participant} has blocks 1..{len(conditions)}")
    condition = conditions[args.block - 1]
    sc = load_scenario(condition)
    if args.vision == "cameras" and "synthetic" in sc["layout"]:  # slots that aren't on the real table
        sys.exit(f"scenarios/{condition}.yaml uses {sc['layout']}: write the lab layout and point the scenarios at it")
    try:
        vision = SimWorld() if args.vision == "synthetic" else Vision()
    except RuntimeError as e:  # no calibration, or a camera that doesn't open
        sys.exit(str(e))
    arm = RosArm(args.host) if args.arm == "ros" else FakeArm()
    info = dict(vision=args.vision, arm=args.arm, host=args.host, language=args.lang, speech=not args.no_speech,
                voice=not args.no_voice)
    runner = Runner(args.participant, condition, arm, vision, audio=not args.no_voice, info=info)
    speech = None if args.no_speech else Speech(runner, args.lang, device=args.mic)
    if args.panel_host not in ("127.0.0.1", "localhost", "::1"):   # loopback: present but unreachable remotely
        print(f"WARNING: --panel-host {args.panel_host} serves the operator panel to anyone who can reach this "
              "machine: the panel has no authentication and can control the experiment and the robot.",
              file=sys.stderr)
    panel(runner, f"{args.participant} · block {args.block}/{len(conditions)} · {condition}", args.panel_host, speech)


if __name__ == "__main__":
    main()
