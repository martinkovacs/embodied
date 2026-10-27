"""Experiment runner (architecture.md sections 5-6): the robot sorts the pieces of a fixed layout into colour
zones, one piece = one trial, behaving as the condition's script says (correct / mistake / ask), and logs it all.

Per trial, arm at home: detect -> pick -> [ask: show, "Is this <colour>?", wait yes/no] -> place -> home -> verify.

Files:
  scenarios/<condition>.yaml   slot order + per-trial script; names its layout
  layouts/*.yaml               the printed template: piece slots and zone spots (table mm), robot frame file
  participants.csv             participant -> conditions in the order they run (counterbalancing)
  logs/<participant>_<condition>.jsonl, logs/<participant>_<condition>/trialNN_aA_{before,after}_{left,right}.png  (A = attempt)

Participant answers come in through Runner.answer(): the operator panel buttons for now, speech.py later.
Anything unplanned (piece not where the layout says, arm failure, failed verify) halts the trial and asks the
operator to retry / skip / abort, or accept when the camera missed a piece the arm did place. The robot never
improvises a question: that would contaminate the conditions.

Usage (from the repo root):
    venv/bin/python runner.py --participant P00 [--block 1] [--vision synthetic|cameras] [--arm ros|fake]
                              [--host localhost]         # operator panel on http://localhost:8081
    venv/bin/python runner.py --self-test                # every condition: fake arm, synthetic vision, no ROS
"""
import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

import robot_frame
from arm_node.kinematics import GRASP_Z, LIFT, PLACE_Z, ik  # noqa: E402  on sys.path via robot_frame

ROOT = Path(__file__).resolve().parent
COLORS = ("red", "blue", "green", "white")
BEHAVIORS = ("correct", "mistake", "ask")
MATCH_MM = 20.0                 # a detection this close to a slot / spot is the piece there (pieces are 35 mm)
REACH_Z = (GRASP_Z, PLACE_Z, GRASP_Z + LIFT)   # arm_node's heights: every slot and spot must be reachable at all
ARM_TIMEOUT_S = 60.0
PORT = 8081
SAY = {"ask": "Is this {color}?", "which": "Which colour is it?", "done": "Done."}


class Unplanned(Exception):
    pass


def load_scenario(condition):
    """The scenario with its layout, checked: ids, script entries, and that the arm reaches every slot and spot."""
    sc = yaml.safe_load((ROOT / "scenarios" / f"{condition}.yaml").read_text())
    lay = yaml.safe_load((ROOT / sc["layout"]).read_text())
    T = robot_frame.load(ROOT / lay["robot_frame"])
    slots = {k: (c, float(x), float(y)) for k, (c, x, y) in lay["slots"].items()}
    zones = {c: [tuple(map(float, p)) for p in lay["zones"][c]] for c in COLORS}
    script = {int(k): v for k, v in (sc.get("script") or {}).items()}
    order = sc["order"]
    assert sc["condition"] == condition, f"{condition}.yaml says condition {sc['condition']}"
    assert all(s in slots for s in order) and len(set(order)) == len(order), "order: unknown or repeated slot"
    assert all(c in COLORS for c, _, _ in slots.values()), "slots: unknown colour"
    for i, b in script.items():
        assert 1 <= i <= len(order) and b["behavior"] in BEHAVIORS, f"script {i}: {b}"
        if b["behavior"] == "mistake":
            assert b["put_to"] in COLORS and b["put_to"] != slots[order[i - 1]][0], f"script {i}: bad put_to"
    for name, (x, y) in [(k, v[1:]) for k, v in slots.items()] + [(c, p) for c in COLORS for p in zones[c]]:
        rx, ry, _ = robot_frame.to_robot(T, (x, y, 0.0))
        for z in REACH_Z:
            try:
                ik(rx, ry, z)
            except ValueError as e:
                raise AssertionError(f"layout {name} ({x:.0f}, {y:.0f}): {e}") from None
    return dict(order=order, script=script, slots=slots, zones=zones, T=T)


def nearest(found, x, y):
    """The detection [(color, x, y)] closest to (x, y) within MATCH_MM, or None."""
    d = [np.hypot(px - x, py - y) for _, px, py in found]
    return found[int(np.argmin(d))] if d and min(d) < MATCH_MM else None


class Vision:
    """Stereo detection with calib_gui's calibration and sources: [(color, x, y)] table mm + the image pair."""

    def __init__(self, synthetic=False):
        import calib_gui as cg  # its State loads calib_data/: calibration and camera settings
        import pieces
        self.cg, self.pieces = cg, pieces
        st = cg.S.settings
        b = float(st["board_mm"])
        self.views = pieces.view_from_calib(cg.S.calib, "left", b), pieces.view_from_calib(cg.S.calib, "right", b)
        w, h = int(st["width"]), int(st["height"])
        keys = ("test_scene_left", "test_scene_right") if synthetic else ("left", "right")
        self.sources = [cg.Source(st[k], w, h) for k in keys]

    def detect(self):
        for s in self.sources:
            s.grab()
        left, right = (s.retrieve() for s in self.sources)
        if left is None or right is None:
            raise Unplanned("camera gave no image")
        found = self.cg.matched_pieces(self.pieces.detect_pieces(left, self.views[0]),
                                       self.pieces.detect_pieces(right, self.views[1]))
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
    """arm_node's /arm/cmd + /arm/status over rosbridge. x, y robot frame mm."""

    def __init__(self, host):
        import roslibpy
        self.roslibpy = roslibpy
        self.ros = roslibpy.Ros(host=host, port=9090)
        self.ros.run(timeout=10)  # raises if rosbridge isn't up
        self.cmd = roslibpy.Topic(self.ros, "/arm/cmd", "std_msgs/String")
        self.waiting, self.n = {}, 0
        roslibpy.Topic(self.ros, "/arm/status", "std_msgs/String").subscribe(self.on_status)

    def on_status(self, msg):  # roslibpy's thread
        s = json.loads(msg["data"])
        if s["state"] != "started" and (w := self.waiting.pop(s.get("id"), None)):
            loop, fut = w
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(s))

    async def do(self, op, x=None, y=None):
        self.n += 1
        cmd = {"id": self.n, "op": op} | ({} if x is None else {"x": round(x, 1), "y": round(y, 1)})
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.waiting[self.n] = (loop, fut)
        self.cmd.publish(self.roslibpy.Message({"data": json.dumps(cmd)}))
        try:
            s = await asyncio.wait_for(fut, ARM_TIMEOUT_S)
        except TimeoutError:
            self.waiting.pop(cmd["id"], None)
            self.stop()  # a hung controller keeps arm_node busy until its goal is cancelled
            raise RuntimeError(f"no answer from arm_node in {ARM_TIMEOUT_S:.0f} s (is it running?)") from None
        if s["state"] != "done":
            raise RuntimeError(s.get("reason", s["state"]))

    def stop(self):
        self.cmd.publish(self.roslibpy.Message({"data": json.dumps({"id": 0, "op": "stop"})}))


class FakeArm:
    async def do(self, op, x=None, y=None):
        await asyncio.sleep(0)

    def stop(self):
        pass


class Log:
    """One JSONL line per event (architecture.md section 6). Appends: a restarted block never loses data."""

    def __init__(self, path, participant, condition):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.f = path, path.open("a")
        self.participant, self.condition, self.trial, self.tail = participant, condition, 0, []
        self.attempt = 1

    def __call__(self, event, **data):
        rec = {"t_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "t_mono": round(time.monotonic(), 3), "participant": self.participant,
               "condition": self.condition, "trial": self.trial, "event": event, "data": data}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.tail.append(f"{rec['t_wall'][11:23]}  #{self.trial:<2} {event} {json.dumps(data) if data else ''}")

    def images(self, pair, tag):
        if pair is not None:
            d = self.path.with_suffix("")
            d.mkdir(exist_ok=True)
            for side, img in zip(("left", "right"), pair):
                cv2.imwrite(str(d / f"trial{self.trial:02d}_a{self.attempt}_{tag}_{side}.png"), img)


class Runner:
    def __init__(self, participant, condition, arm, vision, log_dir=ROOT / "logs", auto=None, audio=True):
        self.sc = load_scenario(condition)
        self.arm, self.vision, self.auto, self.audio = arm, vision, auto, audio
        self.log = Log(Path(log_dir) / f"{participant}_{condition}.jsonl", participant, condition)
        self.used = dict.fromkeys(COLORS, 0)   # next free spot in each zone
        self.prompt = None                     # {"kind": "answer" | "operator", "options", "text", "fut"}
        self.status, self.task, self.said_at = "ready", None, None
        self.step = self.placed = None         # this trial's last arm op done, and the zone it placed in

    # --- input: participant answers (panel now, speech later) and operator choices ---
    def answer(self, word, source="panel"):
        self._resolve("answer", word, source)

    def choose(self, option):
        self._resolve("operator", option, "operator")

    def _resolve(self, kind, value, source):
        p = self.prompt
        if p and p["kind"] == kind and value in p["options"] and not p["fut"].done():
            p["fut"].set_result((value, source))

    async def _wait(self, kind, options, text):
        if self.auto:
            return self.auto(kind, options), "auto"
        self.prompt = {"kind": kind, "options": options, "text": text,
                       "fut": asyncio.get_running_loop().create_future()}
        try:
            return await self.prompt["fut"]
        finally:
            self.prompt = None

    async def ask(self, options):
        word, source = await self._wait("answer", options, "waiting for the participant: " + " / ".join(options))
        self.log("participant_answer", answer=word, source=source,
                 rt_s=round(time.monotonic() - self.said_at, 3))  # from the end of the robot's question
        return word

    # --- output ---
    async def say(self, key, **kw):
        text = SAY[key].format(**kw)
        self.log("robot_say", key=key, text=text)
        wav = next((p for p in (ROOT / "audio" / f"{key}_{kw.get('color')}.wav", ROOT / "audio" / f"{key}.wav")
                    if p.exists()), None)
        # shortcut: macOS `say` until the WAVs are recorded (one voice, one tempo for every participant)
        argv = ["afplay", str(wav)] if wav else ["say", text]
        if self.audio and shutil.which(argv[0]):
            await (await asyncio.create_subprocess_exec(*argv)).wait()
        self.said_at = time.monotonic()

    async def move(self, op, x=None, y=None):
        """Arm op at table (x, y) mm. Any failure is unplanned."""
        rx, ry = (None, None) if x is None else robot_frame.to_robot(self.sc["T"], (x, y, 0.0))[:2]
        self.log("arm_cmd", op=op, **({} if x is None else {"table": [round(x, 1), round(y, 1)],
                                                             "robot": [round(rx, 1), round(ry, 1)]}))
        try:
            await self.arm.do(op, rx, ry)
        except RuntimeError as e:
            raise Unplanned(f"arm {op}: {e}") from None
        self.log("arm_done", op=op)
        self.step = op
        self.vision.after(op, x, y)

    def detect(self, tag):
        found, pair = self.vision.detect()
        self.log("vision", tag=tag, pieces=[[c, round(x, 1), round(y, 1)] for c, x, y in found])
        self.log.images(pair, tag)
        return found

    # --- the experiment ---
    async def trial(self, slot, behavior):
        color, sx, sy = self.sc["slots"][slot]
        self.step = self.placed = None
        self.log("trial_start", slot=slot, color=color, **behavior)
        p = nearest(self.detect("before"), sx, sy)
        if p is None or p[0] != color:
            raise Unplanned(f"slot {slot}: expected {color}, sees {p[0] if p else 'nothing'}")
        await self.move("pick", p[1], p[2])
        target = behavior.get("put_to", color)
        if behavior["behavior"] == "ask":
            await self.move("show")
            await self.say("ask", color=color)
            if await self.ask(("yes", "no")) == "no":
                await self.say("which")
                target = await self.ask(COLORS)
        if self.used[target] >= len(self.sc["zones"][target]):
            raise Unplanned(f"{target} zone is full")
        tx, ty = self.sc["zones"][target][self.used[target]]
        await self.move("place", tx, ty)
        self.used[target] += 1
        self.placed = target
        await self.move("home")
        p = nearest(self.detect("after"), tx, ty)
        if p is None or p[0] != color:
            raise Unplanned(f"verify: no {color} piece at the {target} zone spot ({tx:.0f}, {ty:.0f})")
        self.log("trial_end", placed=target, correct=target == color)

    async def run(self):
        sc, end = self.sc, "aborted"
        self.log("session_start", order=sc["order"], script={str(k): v for k, v in sc["script"].items()})
        try:
            await self.move("home")
            for i, slot in enumerate(sc["order"], 1):
                self.log.trial = i
                self.status = f"trial {i}/{len(sc['order'])}: {slot}"
                behavior = sc["script"].get(i, {"behavior": "correct"})
                self.log.attempt = 1
                while True:
                    try:
                        await self.trial(slot, behavior)
                        break
                    except Unplanned as e:
                        self.log("unplanned", reason=str(e), step=self.step)
                        # placed but not confirmed (camera miss, home failed): the operator can confirm it by eye
                        options = ("retry", "skip", "abort") + (("accept",) if self.placed else ())
                        held = " The piece is in the gripper: take it out." if self.step in ("pick", "show") else ""
                        choice, _ = await self._wait("operator", options, f"UNPLANNED: {e}.{held} Fix the setup, then:")
                        self.log("operator_override", choice=choice)
                        if choice == "accept":
                            color = sc["slots"][slot][0]
                            self.log("trial_end", placed=self.placed, correct=self.placed == color,
                                     operator_verified=True)
                            break
                        if choice == "skip":  # step: how far it got, i.e. what the participant saw
                            self.log("trial_end", skipped=True, step=self.step)
                            break
                        if choice == "abort":
                            return
                        self.log.attempt += 1
            self.log.trial = 0
            await self.say("done")
            end = "done"
        finally:  # also on stop (task cancelled)
            self.status = end
            self.log("session_end", status=end)

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    def stop(self):
        self.arm.stop()
        self.log("stop", source="operator")
        if self.task:
            self.task.cancel()


def panel(runner, title):
    from nicegui import app, ui

    @ui.page("/")
    def page():
        ui.label(title).classes("text-2xl")
        status = ui.label()
        prompt = ui.label().classes("text-xl text-orange-400 min-h-8")  # fixed height: STOP must not jump
        with ui.row():
            ui.button("Start", on_click=runner.start).bind_enabled_from(runner, "task", lambda t: t is None)
            ui.button("STOP", color="red", on_click=runner.stop)
        with ui.row() as answers:
            for w in ("yes", "no", *COLORS):
                ui.button(w, on_click=lambda w=w: runner.answer(w))
        with ui.row() as ops:
            op_buttons = {c: ui.button(c, on_click=lambda c=c: runner.choose(c))
                          for c in ("retry", "skip", "abort", "accept")}
        log = ui.log(max_lines=300).classes("w-full h-96")
        shown = {"n": 0}

        def tick():
            p = runner.prompt
            status.text, prompt.text = runner.status, p["text"] if p else ""
            answers.visible, ops.visible = bool(p and p["kind"] == "answer"), bool(p and p["kind"] == "operator")
            for c, b in op_buttons.items():
                b.visible = bool(p and c in p["options"])
            for line in runner.log.tail[shown["n"]:]:
                log.push(line)
            shown["n"] = len(runner.log.tail)

        ui.timer(0.2, tick)

    app.on_shutdown(runner.arm.stop)
    ui.run(title=title, port=PORT, reload=False, dark=True, show=False)


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        world = SimWorld()

        def run(condition, auto, setup=lambda r: None):
            r = Runner(f"T{len(os.listdir(tmp))}", condition, FakeArm(), world, tmp, auto=auto, audio=False)
            world.reset()
            setup(r)
            asyncio.run(r.run())
            return r, [json.loads(line) for line in r.log.path.read_text().splitlines()]

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
            print(f"  {cond:9s} {len(ends)} trials, {len(asks)} asks, "
                  f"{sum(not e['correct'] for e in ends)} mistakes: ok")

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
        assert len(last(ev, "unplanned")) == 1 and last(ev, "trial_end")[0] == {"skipped": True, "step": None}
        assert len(last(ev, "trial_end")) == len(r.sc["order"])

        # the arm fails after the pick: no "accept" (nothing placed), skip logs how far it got
        class PlaceFailsOnce(FakeArm):
            failed = False

            async def do(self, op, x=None, y=None):
                if op == "place" and not self.failed:
                    self.failed = True
                    raise RuntimeError("controller error")
        offered = []
        r = Runner(f"T{len(os.listdir(tmp))}", "correct", PlaceFailsOnce(), world, tmp, audio=False,
                   auto=lambda kind, opts: offered.append(opts) or "skip")
        world.reset()
        asyncio.run(r.run())
        ev = [json.loads(line) for line in r.log.path.read_text().splitlines()]
        assert offered == [("retry", "skip", "abort")], offered
        assert last(ev, "trial_end")[0] == {"skipped": True, "step": "pick"}

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
    print("runner self-test ok")


def main():
    os.chdir(ROOT)  # calib_gui and robot_frame use paths relative to the repo
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant")
    ap.add_argument("--block", type=int, default=1, help="which of the participant's conditions (1-based)")
    ap.add_argument("--vision", choices=["synthetic", "cameras"], default="synthetic")
    ap.add_argument("--arm", choices=["ros", "fake"], default="ros")
    ap.add_argument("--host", default="localhost", help="rosbridge host (port 9090)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.participant:
        ap.error("--participant is required")
    with open(ROOT / "participants.csv") as f:
        rows = {r["participant"]: r["conditions"].split() for r in csv.DictReader(f)}
    if args.participant not in rows:
        sys.exit(f"{args.participant} is not in participants.csv")
    conditions = rows[args.participant]
    if not 1 <= args.block <= len(conditions):
        sys.exit(f"{args.participant} has blocks 1..{len(conditions)}")
    condition = conditions[args.block - 1]
    vision = SimWorld() if args.vision == "synthetic" else Vision()
    arm = RosArm(args.host) if args.arm == "ros" else FakeArm()
    runner = Runner(args.participant, condition, arm, vision)
    panel(runner, f"{args.participant} · block {args.block}/{len(conditions)} · {condition}")


if __name__ == "__main__":
    main()
