"""Speech-to-text for participant answers: Silero VAD + NVIDIA Canary-1B-v2 (GGUF), English or Hungarian.

The robot asks "Is this <colour>?" and the participant answers out loud ("yes"/"no", then a colour if the
robot was wrong). Listener opens the mic (16 kHz mono float32) and a worker thread runs Silero VAD on the
blocks; complete speech segments are transcribed with Canary and handed over via on_utterance(). match()
maps a transcript to the awaited answers; an ambiguous transcript returns None and the operator decides.
mute()/unmute() bracket the robot's voice: while muted the VAD keeps being fed (its state stays
continuous) but every event is dropped, and speech running across the unmute point gets that point as
its onset and is flagged Utterance.clipped.

Times: Utterance.t_onset / t_end are time.monotonic() from PortAudio's paired capture clock (its ADC-time
pairing applies when the block was sampled, inside the callback), so reaction times measured from the
robot's question are not skewed by buffering or by ASR itself. A block whose clock pair is unusable
continues linearly from the previous one and is flagged Utterance.clock="callback". t_onset is a VAD-frame
(32 ms) estimate: the saved audio plus Utterance.t_audio (time of audio[0], the start of the pre-roll)
let the true onset be refined offline. A processing error resets the pipeline on the same sample clock
and is reported via on_error — the bookkeeping never desyncs. The first ever Canary run compiles Vulkan
pipelines (~15 s once, cached on disk afterwards; per language and unseen input width), so start() pads
and transcribes silence once at every bucket length in every of LANGUAGES before it reports ready; later
utterances transcribe in ~50 ms.

Usage (from the repo root):
    venv/bin/python speech.py --list-devices
    venv/bin/python speech.py --test [--lang hu] [--device N]    # live mic: level + utterances
    venv/bin/python speech.py --file x.wav [--lang en]           # offline, the same VAD+ASR path
    venv/bin/python speech.py --self-test                        # match() + segmenter, no models needed
"""
import argparse
import bisect
import math
import os
import queue
import re
import sys
import threading
import time
import unicodedata
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "canary-1b-v2-Q8_0.gguf"
VAD_MODEL = ROOT / "models" / "silero-vad-v6.2-F32.gguf"
MODEL_URL = "https://huggingface.co/handy-computer/canary-1b-v2-gguf/resolve/main/canary-1b-v2-Q8_0.gguf"
VAD_URL = "https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/main/silero-vad-v6.2-F32.gguf"

SR = 16000                  # VAD, ASR and mic are all 16 kHz mono float32
FRAME = 512                 # Silero VAD frame: 32 ms
START_THR, END_THR = 0.5, 0.35          # VAD hysteresis
MIN_SPEECH_S = 0.15         # shorter blips are dropped (on_speech_start may have fired, that's ok)
END_SILENCE_S = 0.5         # utterance ends after this much below END_THR
MAX_UTT_S = 8.0             # longer speech is force-cut and continues as a new segment
PRE_ROLL_S, POST_ROLL_S = 0.25, 0.15    # audio around the segmented speech, given to ASR
BUCKET_STEP_S, BUCKET_MIN_S = 0.5, 1.0  # padding targets: next multiple of 0.5 s, at least 1.0 s
BUCKET_MAX_S = math.ceil((MAX_UTT_S + PRE_ROLL_S + POST_ROLL_S) / BUCKET_STEP_S) * BUCKET_STEP_S  # longest slice
BUCKETS = tuple(BUCKET_MIN_S + i * BUCKET_STEP_S
                for i in range(int((BUCKET_MAX_S - BUCKET_MIN_S) / BUCKET_STEP_S) + 1))
LEVEL_FALL_DB_S = 40.0      # the meter falls this fast, so peaks stick a bit
ONSET_BACK_THR, ONSET_BACK_FRAMES = 0.15, 8  # onset walks back: contiguous frames >= thr, at most 8
NEG_SCOPE = 4               # a hit stops counting when one of these sits NEG_SCOPE tokens before it
LANGUAGES = ("en", "hu")    # both warm up; the language is switched any time
COLORS = ("red", "blue", "green", "white")

WORDS = {   # answers -> words / phrases per language; multi-word = consecutive whole words. Only unambiguous
            # ones here: "right"/"ja"/"aha" and the like answer themselves ("Right, it's blue").
    "en": {"yes": ["yes", "yeah", "yep", "yup", "correct", "sure", "true", "of course",
                   "that is right", "that s right"],
           "no": ["no", "nope", "nah", "wrong", "incorrect", "false", "is not",
                  "not right", "not correct", "not true", "not it"],
           "red": ["red"], "blue": ["blue"], "green": ["green"], "white": ["white"]},
    "hu": {"yes": ["igen", "persze", "igaz", "helyes", "így van", "igy van"],
           "no": ["nem", "dehogy", "rossz", "hamis", "nem az"],
           "red": ["piros", "vörös", "voros"], "blue": ["kék", "kek"],
           "green": ["zöld", "zold"], "white": ["fehér", "feher"]},
}
NEGATORS = frozenset(("not", "never", "nem", "dehogy"))
UNCERTAIN = ("don t know", "do not know", "not sure", "no idea", "maybe", "perhaps", "talán", "talan",
             "lehet", "nem tudom", "nem vagyok biztos", "nem biztos", "fogalmam sincs")   # said at all: no answer


class SpeechUnavailable(Exception):
    """Cannot listen at this time: models, packages or the mic. The message says what to do."""


def _words(text):
    """One lowercase clause string -> tokens: 'it s blue' -> [it, s, blue]."""
    return "".join(c if unicodedata.category(c)[0] == "L" else " " for c in text).split()


# Hungarian "nem" is both the answer "no" and the ordinary negator: it only answers on its own or right
# before one of these ("nem az", "nem igaz", "nem piros"); anywhere else ("nem emlékszem") it is grammar
HU_NEM_NEXT = frozenset(("az",)) | frozenset(
    t for k in ("yes",) + COLORS for w in WORDS["hu"][k] for t in _words(w))


def _clauses(text):
    """Token lists per clause: punctuation splits before tokenizing (Canary punctuates, and a negation
    must not scope across a comma) with contractions expanded first: "No, it's blue" -> [[no],
    [it, s, blue]], "isn't" -> [is, not] (the typographic apostrophe too)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"n['’]t\b", " not", text)                    # isn't -> is not
    return [w for c in re.split(r"[,.;!?…–—-]+", text)          # it's -> it s
            if (w := _words(re.sub("['’]", " ", c)))]


def _scan(clauses, tokens):
    """Do the phrase tokens occur consecutively in one clause? -> (left standing, negated): a hit is
    negated when a negator sits within the NEG_SCOPE tokens before it in that clause."""
    plain = negated = False
    for w in clauses:
        for i in range(len(w) - len(tokens) + 1):
            if w[i:i + len(tokens)] == tokens:
                if any(w[j] in NEGATORS for j in range(max(0, i - NEG_SCOPE), i)):
                    negated = True
                else:
                    plain = True
    return plain, negated


def _nem_answers(clauses):
    """A clausal Hungarian "nem" is the answer "no" when it stands alone ("Nem.") or is immediately
    followed by az / a colour / a yes-word; before anything else it is the ordinary negator
    ("nem emlékszem", "nem látom") and answers nothing — the operator decides."""
    return any(w == ["nem"] or any(w[i + 1] in HU_NEM_NEXT for i, t in enumerate(w[:-1]) if t == "nem")
               for w in clauses)


def match(text: str, options: Sequence[str], language: str, asked=None) -> str | None:
    """The one awaited answer the transcript points to, else None (the operator decides).

    A negated yes-word counts as "no" ('not right', 'nem igaz'); words of doubt mean no answer at all;
    with `asked` given (yes/no question), a plain yes beside the asked colour negated ("Yes, not red")
    or a different un-negated colour contradicts -> None, while a no beside any colour stays "no".
    """
    clauses = _clauses(text)
    if any(_scan(clauses, _words(u))[0] for u in UNCERTAIN):
        return None
    plain, negated = {}, {}             # answer -> one of its phrases is left standing / is negated
    for option, phrases in WORDS[language].items():
        for phrase in phrases:
            if language == "hu" and phrase == "nem" and not _nem_answers(clauses):
                continue                # the grammar negator ("nem emlékszem"), not the answer
            found, hit_negated = _scan(clauses, _words(phrase))
            plain[option] = plain.get(option, False) or found
            negated[option] = negated.get(option, False) or hit_negated
            if hit_negated and option == "yes":
                plain["no"] = True      # "not right" denies the yes: no, not silence
    if asked is not None and plain.get("yes") and (negated.get(asked) or any(plain.get(c)
            for c in COLORS if c != asked)):
        return None                     # yes beside a negated asked colour / another colour: no answer
    hits = {o for o in options if plain.get(o)}     # a plain no beside any colour stays "no"
    return next(iter(hits)) if len(hits) == 1 else None


class _Timeline:
    """Stream sample index -> time.monotonic(): (first sample, t) per captured block, linear in between."""

    SPAN = 30 * SR          # kept sample span: far more than a slice can ever reach back (preroll + 8 s)

    def __init__(self):
        self.blocks = []    # [(first_sample, t)]

    def add(self, first, t):
        self.blocks.append((first, t))
        while len(self.blocks) > 2 and self.blocks[-1][0] - self.blocks[0][0] > self.SPAN:
            del self.blocks[0]

    def at(self, sample):
        """Time of one sample; the file path adds (i, i/SR) blocks, so this just yields i/SR there."""
        k = bisect.bisect_right([f for f, _ in self.blocks], sample) - 1
        first, t = self.blocks[max(k, 0)]
        return t + (sample - first) / SR


@dataclass
class Onset:
    at: int                 # first speech sample, stream-sample clock


@dataclass
class Segment:
    onset: int              # first sample with speech
    end: int                # first sample after the last voiced one
    audio_from: int         # onset - pre-roll
    audio_to: int           # end + post-roll
    clipped: bool = False   # speech already ran at the unmute point: the onset is not real


class Segmenter:
    """VAD frame probabilities -> Onset / Segment events on the stream-sample clock, no audio I/O of its
    own. The vad callable is fed a block and returns the probabilities of the VAD frames the block
    completes, in order (frame k covers samples base+k*FRAME..+FRAME-1; a too-short tail stays buffered
    inside the callable). Parameters are the module constants START_THR / END_THR / MIN_SPEECH_S / ...

    mute()/unmute(at) bracket the robot's voice: events are dropped in between, but the VAD keeps being
    fed so its state stays continuous. The onset walks back over contiguous frames at or above
    ONSET_BACK_THR (a 32 ms VAD-frame estimate either way) and never before the unmute point; speech
    running across `at` gets `at` as its onset and is flagged clipped.
    """

    def __init__(self, vad=None, base=0, floor=None):
        self.vad = vad
        self.base = base        # stream sample of the first frame (a reset stream continues, not restarts)
        self._frames = 0        # probabilities consumed so far = frames completed so far
        self._onset = None      # speech start sample while a segment is open
        self._last = 0          # end of the last frame at or above END_THR in the current segment
        self._hist = []         # the last ONSET_BACK_FRAMES+1 frame probabilities, for the walk-back
        self._floor = floor     # latest unmute point: onsets never go before it
        self._clip = False      # the open segment runs across the unmute point
        self._muted = False

    def feed(self, block):
        events = []
        for p in (self.vad(block) if self.vad else ()):
            a, self._frames = self.base + self._frames * FRAME, self._frames + 1
            b = a + FRAME
            self._hist = (self._hist + [p])[-(ONSET_BACK_FRAMES + 1):]
            if self._onset is None:
                if p >= START_THR:
                    self._onset, self._last = self._first(a), b
                    events.append(Onset(self._onset))
                continue
            if p >= END_THR:        # hysteresis: below START_THR still keeps the segment alive
                self._last = b
            if b - self._onset >= int(MAX_UTT_S * SR):          # force cut; speech continues in a new one
                if seg := self._close(b):
                    events.append(seg)
                self._onset, self._last = b, b
            elif b - self._last >= int(END_SILENCE_S * SR) and (seg := self._close(self._last)):
                events.append(seg)
        return [] if self._muted else events

    def _first(self, a):
        """Onset to declare for the frame that just crossed START_THR (start sample a)."""
        back = 0
        for p in reversed(self._hist[:-1]):
            if p < ONSET_BACK_THR:
                break
            back += 1
        onset = a - min(back, ONSET_BACK_FRAMES) * FRAME
        if self._floor is not None and onset < self._floor:
            onset, self._clip = self._floor, True   # speech was already running at the unmute point
        return onset

    def _close(self, cut):
        """Close the segment at `cut`; a too-short blip is dropped (its Onset already went out; that's ok)."""
        onset, self._onset, clip, self._clip = self._onset, None, self._clip, False
        if cut - onset < int(MIN_SPEECH_S * SR):
            return None
        return Segment(onset, cut, max(0, onset - int(PRE_ROLL_S * SR)),
                       cut + int(POST_ROLL_S * SR), clip)

    def mute(self):
        """The robot speaks: an open segment is dropped silently and events are suppressed until
        unmute(); feeding continues so the VAD state stays continuous."""
        self._onset, self._clip, self._muted = None, False, True

    def unmute(self, at):
        """Listening is back on from stream sample `at` on: onsets never go before it. Speech running
        across `at` gets `at` as its onset and is flagged clipped. Returns the Onset event for it (the
        one whose real onset was lost to the robot's voice) or None."""
        self._muted, self._floor = False, at
        onset = self._onset
        if onset is not None:                   # speech runs across the unmute: its onset isn't real
            self._onset, self._clip = at, True
            return Onset(at)
        return None


@dataclass
class Utterance:
    text: str               # Canary transcript ("" if ASR failed, see error)
    t_onset: float          # time.monotonic() of the first speech sample (VAD-frame estimate)
    t_end: float            # time.monotonic() of the end of speech
    asr_s: float            # transcription wall time
    audio: np.ndarray       # 16 kHz float32 incl. pre-/post-roll
    t_audio: float = 0.0    # time.monotonic() of audio[0] (pre-roll start), to refine the onset offline
    language: str = "en"    # the language Canary decoded it with
    clipped: bool = False   # speech was already going on when listening was unmuted: onset not real
    clock: str = "adc"      # "adc" | "callback": which clock t_onset's block came from
    error: str | None = None    # ASR failed; not the same as silence


_DEFS = {MODEL: MODEL_URL, VAD_MODEL: VAD_URL}


def _check_model(path, what):
    if not Path(path).is_file():
        raise SpeechUnavailable(f"{what} model not found: {path}"
                                + (f"\n  download: {_DEFS[Path(path)]}" if Path(path) in _DEFS else ""))


class _Engine:
    """Both models loaded once. run()/vad() come from one thread only (one compute call at a time per
    Model) — which both the Listener worker and the --file path hold to: they never overlap.
    """

    def __init__(self, model=MODEL, vad_model=VAD_MODEL):
        _check_model(model, "Canary-1B-v2")
        _check_model(vad_model, "Silero VAD")
        import transcribe_cpp as t      # the native lib spins up Vulkan; keep `import speech` fast
        self.t = t
        self.asr = t.Model(model)
        self.vad_m = t.Model(vad_model)
        if self.vad_m.vad_info.frame_samples != FRAME:
            raise SpeechUnavailable(f"VAD frame is {self.vad_m.vad_info.frame_samples} samples, expected {FRAME}")
        self.vads = self.vad_m.vad_session()

    def vad(self, chunk):
        return self.vads.stream_feed(chunk).probs

    def vad_reset(self):
        self.vads.stream_reset()

    def run(self, pcm, language):
        """(text, error): silence can decode into a token loop (not a failure); a real failure returns
        the error string and an unusable text."""
        try:
            with self.asr.session() as s:   # Canary requires the language; VAD said when, ASR says what
                return s.run(pcm, language=language, pnc="off").text.strip(), None
        except self.t.TranscribeError:      # silence can decode into a token loop; not a failure
            return "", None
        except Exception as e:              # native error / OOM: the utterance says so, not "silence"
            return "", f"{type(e).__name__}: {e}"

    def warmup(self):
        """Transcribe silence once at every bucket length in every language (LANGUAGES): exactly what
        keeps real utterances ~50 ms however the language is switched meanwhile."""
        for s in BUCKETS:
            for lang in LANGUAGES:
                self.run(np.zeros(int(s * SR), np.float32), lang)

    def close(self):
        for close in (self.vads.close, self.asr.close, self.vad_m.close):
            close()


class _Ear:
    """The pipeline the Listener and the --file path share: block -> segmenter (real or injected VAD) ->
    padded bucket -> transcription -> utterance events. Language is read at transcription time, so it can
    change until the moment an utterance is transcribed. Callbacks run on the caller's thread.
    """

    def __init__(self, engine, language="en", on_utterance=None, on_speech_start=None):
        self.engine = engine
        self.language = language if callable(language) else (lambda: language)
        self.on_utterance, self.on_speech_start = on_utterance, on_speech_start
        engine.vad_reset()          # fresh segmenter and fresh VAD stream must agree on the sample clock
        self.seg = Segmenter(engine.vad)
        self.timeline = _Timeline()
        self.blocks = []            # recent audio [(first_sample, np.float32, clk)] for slicing utterances
        self.n = 0                  # samples fed so far
        self._muted = False
        self._unmute_at = None      # last unmute point, kept for a reset

    def push(self, block, first, t, clk="adc"):
        """A 16 kHz float32 block; first = the block's first stream sample, t = monotonic time of it, clk
        = that block's clock kind. A processing error resets the pipeline and re-raises (the sample
        bookkeeping survives: the next block continues at first + len(block))."""
        try:
            self.timeline.add(first, t)
            self.blocks.append((first, block, clk))
            self.n = first + len(block)
            for ev in self.seg.feed(block):
                if isinstance(ev, Segment):
                    self._utterance(ev)
                elif self.on_speech_start:
                    self.on_speech_start(self.timeline.at(ev.at))
        except BaseException:
            self.reset(first + len(block))
            raise
        # keep audio since (preroll + max utterance) back from now: everything a future slice can need
        while len(self.blocks) > 1 and self.blocks[1][0] + len(self.blocks[1][1]) < self.n - int(20 * SR):
            del self.blocks[0]

    def reset(self, base):
        """After an error: fresh VAD stream and segmenter; the stream-sample clock continues at `base`
        (the next block's first sample), it does not restart at 0."""
        self.engine.vad_reset()
        self.seg = Segmenter(self.engine.vad, base=base, floor=self._unmute_at)
        if self._muted:
            self.seg.mute()

    def mute(self):
        """The robot starts speaking: events are dropped until unmute() (the VAD keeps being fed)."""
        self._muted = True
        self.seg.mute()

    def unmute(self):
        """The robot's line ended: speech already running is clipped at the unmute point."""
        at, self._muted, self._unmute_at = self.n, False, self.n
        ev = self.seg.unmute(at)
        if ev and self.on_speech_start:
            self.on_speech_start(self.timeline.at(ev.at))

    def _utterance(self, seg):
        a, b = max(seg.audio_from, self.blocks[0][0]), min(seg.audio_to, self.n)
        parts = [x[max(a - f, 0):b - f] for f, x, _ in self.blocks if f < b and f + len(x) > a]
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts)
        text, asr_s, lang, error = self.transcribe(audio)
        if self.on_utterance:
            clk = next((c for f, x, c in self.blocks if f <= seg.onset < f + len(x)), self.blocks[0][2])
            self.on_utterance(Utterance(text, self.timeline.at(seg.onset), self.timeline.at(seg.end), asr_s,
                                        audio, t_audio=self.timeline.at(a), language=lang,
                                        clipped=seg.clipped, clock=clk, error=error))

    def transcribe(self, pcm):
        """Pad with trailing silence up to the next bucket length (see module docstring), then ASR:
        (text, wall time, language used, error)."""
        target = int(max(BUCKET_MIN_S, math.ceil(len(pcm) / SR / BUCKET_STEP_S) * BUCKET_STEP_S) * SR)
        if len(pcm) < target:   # never longer than BUCKET_MAX_S: force-cut speech + rolls
            pcm = np.concatenate([pcm, np.zeros(target - len(pcm), np.float32)])
        t0 = time.monotonic()
        text, error = self.engine.run(pcm, self.language())
        return text, time.monotonic() - t0, self.language(), error


class Listener:
    """Live mic listener. start() is blocking (model load + warm-up, then the mic opens) and raises
    SpeechUnavailable with an actionable message; call it once, stop() is idempotent.

    Callbacks run on the worker thread: on_speech_start(t_onset) as soon as the VAD declares speech —
    a below-150 ms blip may never return as an utterance — and on_utterance(u) after ASR, ~50 ms after
    t_end; on_error(msg) for a runtime processing error (the pipeline reset itself and keeps running).
    language ("en"/"hu") can be changed any time and is used for the next utterance; level_db is the
    metered mic RMS in dBFS; ready is True once utterances can arrive. mute()/unmute() bracket the
    robot's voice: while muted, onsets and utterances are all dropped (see Segmenter).
    """

    def __init__(self, language="en", model=MODEL, vad_model=VAD_MODEL, device=None,
                 on_utterance=None, on_speech_start=None, on_error=None):
        self.language = language
        self.model, self.vad_model, self.device = Path(model), Path(vad_model), device
        self.on_utterance, self.on_speech_start, self.on_error = on_utterance, on_speech_start, on_error
        self.level_db, self.ready, self.load_s = -120.0, False, 0.0
        self.rate = SR          # if the device won't open at 16 kHz: opened at its default rate, resample
        self.mic = None         # input device index that ends up open
        self.mic_name = ""
        self.last_t, self.last_n = None, 0  # capture time of the last block's first sample, for liveness
        self.overflows = 0      # input-overflow status flags seen so far
        self._queue = queue.Queue()
        self._stop = True
        self._thread, self._stream, self._engine, self._ear = None, None, None, None
        self._prev_t = None     # capture time of the previous block, for a fallback clock
        self._prev_n = 0
        self._in_latency = 0.05 # stream input latency (measured on open), last-resort clock estimate

    def start(self):
        if self._thread and self._thread.is_alive():
            return              # start() once between stop()s
        try:
            import sounddevice as sd
        except ImportError:
            raise SpeechUnavailable("sounddevice not installed: venv/bin/pip install sounddevice") from None
        t0 = time.monotonic()
        try:
            self._engine = _Engine(self.model, self.vad_model)
            have = self._engine.asr.capabilities.languages
            if self.language not in have:
                raise SpeechUnavailable(f"language {self.language!r} missing from the model's languages "
                                        f"({', '.join(sorted(have))})")
            self._engine.warmup()
            self._ear = _Ear(self._engine, lambda: self.language, self.on_utterance, self.on_speech_start)
            self._open_mic(sd)
            self._fed, self._prev_t, self._prev_n = 0, None, 0    # no clock history from a former life
            self.last_t, self.overflows = None, 0
            self._stream.start()
            self._stop = False
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except BaseException:
            self.stop()
            raise
        self.ready = True
        self.load_s = time.monotonic() - t0

    def _open_mic(self, sd):
        try:
            dev = self.device if self.device is not None else sd.query_devices(kind="input")["index"]
            info = sd.query_devices(dev)
            if info["max_input_channels"] == 0:
                raise ValueError("has no input channels")
        except Exception:
            raise SpeechUnavailable("no such input device; run venv/bin/python speech.py --list-devices") from None
        self.mic, self.mic_name = dev, info["name"]
        try:
            self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=dev,
                                          callback=self._callback)
            self.rate = SR
        except Exception as e:
            try:    # a device that won't take 16 kHz: open at its own rate, resample in the callback
                self.rate = int(info["default_samplerate"])
                self._stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="float32", device=dev,
                                              callback=self._callback)
            except Exception:
                raise SpeechUnavailable(f"cannot open input device #{dev} ({info['name']}): {e}") from None
        try:
            self._in_latency = float(self._stream.latency)
        except Exception:
            pass    # Pa refused: the estimate stays

    def _callback(self, data, frames, tinfo, status):
        """Light: copy + timing + hand over to the worker. Capture time from PortAudio's paired clock:
        its ADC time says when the block's first sample was sampled, measured in PortAudio's own clock.
        An unusable pair continues from the previous block, or falls back to the stream input latency."""
        x = np.array(data[:, 0], dtype=np.float32)
        if self.rate != SR:                             # linear interpolation, no resampler dependency
            n = max(1, int(round(frames * SR / self.rate)))
            x = np.interp(np.linspace(0.0, frames - 1, n), np.arange(frames), x).astype(np.float32)
        first, self._fed = self._fed, self._fed + len(x)
        adc, cur = tinfo.inputBufferAdcTime, tinfo.currentTime
        if adc > 0 and cur > 0:                         # t = now, minus the Pa-clock age of the sample
            t, clk = time.monotonic() - (cur - adc), "adc"
        elif self._prev_t is not None:                  # unusable pair: continue linearly
            t, clk = self._prev_t + self._prev_n / SR, "callback"
        else:
            t, clk = time.monotonic() - len(x) / SR - self._in_latency, "callback"
        self._prev_t, self._prev_n = t, len(x)
        self.last_t, self.last_n = t, len(x)
        if status.input_overflow:                       # count them for the liveness heartbeat
            self.overflows += 1
        self._queue.put_nowait((x, first, t, clk))

    @property
    def age_s(self):
        """Seconds since the mic last captured a block (None: none captured yet)."""
        if self.last_t is None:
            return None
        return time.monotonic() - (self.last_t + self.last_n / SR)

    def _worker(self):
        while not self._stop:
            try:
                x, first, t, clk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                db = 20.0 * math.log10(max(float(np.sqrt(np.mean(x * x))), 1e-6))
                self.level_db = max(db, self.level_db - LEVEL_FALL_DB_S * len(x) / SR)
                self._ear.push(x, first, t, clk)
            except Exception as e:      # a bad block must not cost the listener; the ear reset itself
                msg = f"{type(e).__name__}: {e}"
                if self.on_error:
                    self.on_error(msg)
                else:
                    print(f"speech: {msg}", file=sys.stderr)

    def mute(self):
        """The robot starts speaking: everything until unmute() is discarded (the VAD stays warm)."""
        if self._ear:
            self._ear.mute()

    def unmute(self):
        """The robot's line ended: speech running across the unmute point comes out clipped."""
        if self._ear:
            self._ear.unmute()

    def stop(self):
        """Idempotent: closes the mic stream (no new blocks arrive), joins the worker so its in-flight
        call finishes, frees the models (start() reloads)."""
        self._stop = True
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread:
            self._thread.join()
            self._thread = None
        if self._engine:
            self._ear = None
            self._engine.close()
            self._engine = None
        self.ready = False


def _child(language, device, events, control):
    """ListenerProcess's child: a Listener whose events go back over a queue; exits when its parent is
    gone (nobody is left to read us)."""
    lst = Listener(language, device=device,
                   on_utterance=lambda u: events.put(("utterance", u)),
                   on_speech_start=lambda t: events.put(("onset", t)),
                   on_error=lambda msg: events.put(("error", msg)))
    parent = os.getppid()
    try:
        lst.start()
    except Exception as e:                  # models, packages, mic: what exactly failed, debuggable
        events.put(("error", f"{type(e).__name__}: {e}"))
        return
    events.put(("ready", lst.load_s, lst.mic_name))
    try:
        while True:
            if os.getppid() != parent:
                break
            try:
                msg = control.get(timeout=0.1)
            except queue.Empty:             # heartbeat ~10x/s: age of the last captured block
                events.put(("level", lst.level_db, lst.age_s, lst.overflows))
                continue
            if msg[0] == "stop":
                break
            if msg[0] == "language":
                lst.language = msg[1]
            elif msg[0] == "mute":
                lst.mute()
            elif msg[0] == "unmute":
                lst.unmute()
    finally:
        lst.stop()


class ListenerProcess:
    """A Listener in a child process, so a native crash in the speech engine can't take the experiment session
    down with it. Same callbacks, delivered on a reader thread here: on_utterance(u), on_speech_start(t),
    on_error(msg) for runtime processing errors plus on_state(state, detail) with state "loading" | "ready" |
    "error" | "crashed" | "stopped". mute()/unmute() forward over the control queue. warning says what
    listening can't currently do (None when all is well): "no audio from the microphone for N s" (the age of
    the last captured block, from the child's heartbeat), "speech process not responding" (a live child
    without an event for 3 s). time.monotonic() is system-wide on Linux and macOS, so the child's onset
    times compare directly with this process's clock. start() returns at once.
    """

    def __init__(self, language="en", device=None, on_utterance=None, on_speech_start=None, on_state=None,
                 on_error=None):
        self._language, self.device = language, device
        self.on_utterance, self.on_speech_start, self.on_state, self.on_error = \
            on_utterance, on_speech_start, on_state, on_error
        self.state, self.detail, self.level_db, self.mic_name = "stopped", "", -120.0, ""
        self.warning = None
        self._proc = None
        self._age, self._age_at, self._last_ev, self._overflows = None, None, None, 0

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        if self._proc:
            self._control.put(("language", value))

    def start(self):
        if self._proc and self._proc.is_alive():
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")       # a fresh interpreter: no forked threads, no inherited GPU state
        self._events, self._control = ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(target=_child, args=(self._language, self.device, self._events, self._control),
                                 daemon=True, name="speech")
        self._set("loading", "loading the speech models")
        self._proc.start()
        threading.Thread(target=self._reader, args=(self._proc, self._events), daemon=True).start()

    def _set(self, state, detail=""):
        self.state, self.detail = state, detail
        if self.on_state:
            self.on_state(state, detail)

    def mute(self):
        """The robot starts speaking: discard everything until unmute()."""
        if self._proc:
            self._control.put(("mute",))

    def unmute(self):
        """The robot's line ended."""
        if self._proc:
            self._control.put(("unmute",))

    def _warn(self):
        """warning is re-evaluated on every reader wake; state "ready" only, so it never talks over
        loading/crash states."""
        self.warning = None
        if self.state != "ready":
            return
        now = time.monotonic()
        if self._last_ev and now - self._last_ev > 3:
            self.warning = "speech process not responding"
        elif self._age is not None:
            age = self._age + now - self._age_at
            if age > 2:
                self.warning = f"no audio from the microphone for {age:.0f} s"
                self.level_db = -120.0      # a stale bar would lie

    def _reader(self, proc, events):
        while True:
            try:
                ev = events.get(timeout=0.5)
            except queue.Empty:
                if proc is not self._proc:
                    return          # stopped (or restarted): this reader's child is gone on purpose
                if not proc.is_alive():
                    if self.state in ("loading", "ready"):
                        self._set("crashed", f"speech process died (exit code {proc.exitcode})")
                    return
                self._warn()
                continue
            if proc is not self._proc:
                return              # a restart happened: never deliver the old child's events
            self._last_ev = time.monotonic()
            kind = ev[0]
            if kind == "level":
                self.level_db, self._age, self._overflows, self._age_at = ev[1], ev[2], ev[3], self._last_ev
            elif kind == "onset" and self.on_speech_start:
                self.on_speech_start(ev[1])
            elif kind == "utterance" and self.on_utterance:
                self.on_utterance(ev[1])
            elif kind == "ready":
                self.mic_name = ev[2]
                self._set("ready", f"listening on {ev[2]} (load + warm-up {ev[1]:.1f} s)")
            elif kind == "error":
                if self.on_error:
                    self.on_error(ev[1])
                if self.state == "loading":
                    self._set("error", ev[1])   # start-up failure (runtime errors keep the listener up)
            self._warn()

    def stop(self):
        proc, self._proc = self._proc, None
        if proc and proc.is_alive():
            self._control.put(("stop",))
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.kill()
        if proc:
            self.warning = None
            self._set("stopped")


def _devices():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        tag = "*" if i == sd.default.device[0] else " "
        print(f"{i:3d}{tag} {d['name']}  ({d['max_input_channels']} in, {d['max_output_channels']} out)")


def _file(path, language):
    """Offline: the file flows through the same segmenter + VAD + ASR path as the live listener."""
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            sys.exit(f"{path}: needs a 16 kHz mono 16-bit WAV, got {w.getframerate()} Hz, {w.getnchannels()}"
                     f" ch, {w.getsampwidth() * 8} bit")
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    engine = _Engine()
    ear = _Ear(engine, language,
               on_utterance=lambda u: print(f"{u.t_onset:6.2f}-{u.t_end:6.2f} s  \"{u.text}\""
                                            f"  yes/no={match(u.text, ('yes', 'no'), language) or '-'}"
                                            f"  colour={match(u.text, COLORS, language) or '-'}"
                                            f"  asr {u.asr_s * 1000:.0f} ms"
                                            + (f"  ERROR {u.error}" if u.error else "")))
    pcm = np.concatenate([pcm, np.zeros(SR, np.float32)])  # trailing silence closes speech running to the end
    for i in range(0, len(pcm), SR):    # 1 s blocks; t = seconds from file start doubles as the clock
        ear.push(pcm[i:i + SR], i, i / SR)
    engine.close()


def _test(args):
    """Live mic: level line every ~2 s, each utterance with its match and timings."""
    listener = Listener(args.lang, device=args.device)
    listener.on_utterance = lambda u: print(  # noqa: E731  (used once)
        f'"{u.text}"  yes/no={match(u.text, ("yes", "no"), listener.language) or "-"}'
        f"  colour={match(u.text, COLORS, listener.language) or '-'}"
        f"  latency {(time.monotonic() - u.t_onset) * 1000:.0f} ms"
        f"  asr {u.asr_s * 1000:.0f} ms  speech {u.t_end - u.t_onset:.2f} s"
        f"  {u.language}/{u.clock}" + ("  clipped" if u.clipped else "") + (f"  ERROR {u.error}" if u.error else ""))
    listener.start()
    print(f"ready (load + warm-up {listener.load_s:.1f} s)  mic #{listener.mic} ({listener.mic_name})")
    try:
        while True:
            time.sleep(2.0)
            print(f"level {listener.level_db:6.1f} dBFS")
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()


def self_test():
    # --- match(): clauses + contractions + negation scope + uncertainty + the asked consistency check
    m, y_n, colours = match, ("yes", "no"), COLORS
    assert m("Yes", y_n, "en") == "yes" and m("Nope.", y_n, "en") == "no"
    assert m("Of course!", y_n, "en") == "yes" and m("That's right", y_n, "en") == "yes"
    assert m("Is this red? Yes", y_n, "en") == "yes"        # punctuation and the stray question words
    assert m("totally false", y_n, "en") == "no" and m("Wrong.", y_n, "en") == "no"
    assert m("it's not right", y_n, "en") == "no"            # negated before a yes-word
    assert m("yes no", y_n, "en") is None and m("red or blue", ("red", "blue"), "en") is None
    assert m("please", colours, "en") is None and m("red", colours, "en") == "red"
    assert m("white.", colours, "en") == "white"
    # ambiguous yes-words are gone: "right" alone, "it is", jó/jo/ja, aha, yah
    assert m("Right, it's blue", y_n, "en") is None and m("it is blue", y_n, "en") is None
    assert m("Jó", y_n, "hu") is None and m("ja", y_n, "hu") is None and m("Aha.", y_n, "hu") is None
    # a hit with a negator up to 4 tokens back in the clause doesn't count as itself; yes-word -> no
    assert m("I don't think that's right", y_n, "en") == "no"
    assert m("I don't think it's red", colours, "en") is None and m("not really red", colours, "en") is None
    assert m("it isn't blue", colours, "en") is None and m("not white", colours, "en") is None
    assert m("No, not red, it's blue", colours, "en") == "blue"
    assert m("nem piros, kék", colours, "hu") == "blue" and m("nem is piros", colours, "hu") is None
    assert m("Nem igaz", y_n, "hu") == "no" and m("it isn't", y_n, "en") == "no"
    assert m("Ask not!", y_n, "en") is None                  # neither word is an answer
    # uncertainty means no answer at all
    assert m("Nem tudom.", y_n, "hu") is None and m("Nem vagyok biztos.", y_n, "hu") is None
    assert m("not sure", y_n, "en") is None and m("I have no idea", y_n, "en") is None
    assert m("maybe blue", colours, "en") is None and m("talán piros", colours, "hu") is None
    assert m("lehet, hogy kék", colours, "hu") is None
    # yes/no question with a known asked colour: contradicting it is not an answer
    assert m("Yes, it's red", y_n, "en", asked="red") == "yes"
    assert m("Yes, it's blue", y_n, "en", asked="red") is None
    assert m("No, it's blue", y_n, "en", asked="red") == "no"  # a no beside any colour stays no
    assert m("Yes, not red", y_n, "en", asked="red") is None   # a plain yes + the asked colour negated
    assert m("No, not red", y_n, "en", asked="red") == "no"
    assert m("Yes, not blue", y_n, "en", asked="red") == "yes"  # only the asked colour counts there
    assert m("Jó, de kék", y_n, "hu", asked="red") is None
    # the still-good basics
    assert m("Igen.", y_n, "hu") == "yes" and m("Nem.", y_n, "hu") == "no" and m("Persze", y_n, "hu") == "yes"
    assert m("így van", y_n, "hu") == "yes" and m("igy van", y_n, "hu") == "yes"
    assert m("Rossz!", y_n, "hu") == "no" and m("Nem az", y_n, "hu") == "no"
    # Hungarian "nem" answers only alone or before az / a colour / a yes-word; elsewhere (with a verb
    # after it) it is the ordinary negator and matches nothing, so the operator decides
    assert m("igen", y_n, "hu") == "yes" and m("Nem, kék.", y_n, "hu") == "no"
    assert m("nem igaz", y_n, "hu") == "no" and m("nem az", y_n, "hu") == "no"
    assert m("nem emlékszem", y_n, "hu") is None and m("nem látom", y_n, "hu") is None
    assert m("Nem biztos.", y_n, "hu") is None and m("nem biztos", y_n, "hu") is None
    assert m("Biztos nem", y_n, "hu") is None  # not via UNCERTAIN ("nem biztos"): no "nem"-answer here
    assert m("Nem, nem piros, kék", colours, "hu") == "blue"    # a colour question: negated colours are no hits
    for opt, said in (("blue", "kék"), ("blue", "kek"), ("green", "zöld"), ("green", "zold"),
                      ("white", "fehér"), ("white", "feher"), ("red", "piros"), ("red", "voros")):
        assert m(said, (opt,) + colours, "hu") == opt, said
    assert m("igen, nem", y_n, "hu") is None

    # --- Segmenter with scripted VAD probabilities (frame k = probability of samples base+k*512..)
    class Scripted:
        probs, fed, done = None, None, None

        def __init__(self, probs):
            self.probs, self.fed, self.done = probs, 0, 0

        def __call__(self, chunk):      # the callable contract, like the real VAD stream prefix
            self.fed += len(chunk)
            n = min(self.fed // FRAME, len(self.probs)) - self.done
            out, self.done = self.probs[self.done:self.done + n], self.done + n
            return out

    def run(probs, chunk=1600, base=0):
        seg = Segmenter(Scripted(probs), base=base)
        ev = []
        for i in range(0, (len(probs) + 26) * FRAME, chunk):
            ev += seg.feed(np.zeros(chunk, np.float32))
        return ev

    ev = run([0.05] * 5 + [0.9] * 30 + [0.1] * 20)                  # normal utterance with rolls
    onset, seg = ev
    assert onset.at == 5 * FRAME, ev
    assert (seg.onset, seg.end, seg.audio_from, seg.audio_to, seg.clipped) == \
        (5 * FRAME, 35 * FRAME, max(0, 5 * FRAME - int(PRE_ROLL_S * SR)), 35 * FRAME + int(POST_ROLL_S * SR),
         False), seg
    ev = run([0.05] * 5 + [0.9] * 2 + [0.1] * 20)                   # 64 ms blip: no utterance
    assert len(ev) == 1 and isinstance(ev[0], Onset) and ev[0].at == 5 * FRAME, ev
    ev = run([0.05] * 2 + [0.9] + [0.4] * 4 + [0.1] * 17)           # 0.4 keeps speech, no new onset
    assert len(ev) == 2 and ev[0].at == 2 * FRAME and (ev[1].onset, ev[1].end) == (2 * FRAME, 7 * FRAME), ev
    ev = run([0.9] * 300 + [0.1] * 20)                              # 9.6 s: force-cut at 8.0 s
    on0, seg1, seg2 = ev
    assert (seg1.onset, seg1.end) == (0, int(MAX_UTT_S * SR)), seg1
    assert (seg2.onset, seg2.end) == (int(MAX_UTT_S * SR), 300 * FRAME), seg2
    ev = run([0.05] * 5 + [0.9] * 30 + [0.1] * 20, base=7 * SR)     # a rebased stream keeps its clock
    onset, seg = ev
    assert onset.at == 7 * SR + 5 * FRAME and seg.onset == 7 * SR + 5 * FRAME, ev
    ev = run([0.05, 0.2, 0.3, 0.9] + [0.9] * 26 + [0.1] * 20)       # onset walks back to the 0.2 frame
    assert ev[0].at == 1 * FRAME and ev[1].onset == 1 * FRAME, ev
    ev = run([0.3] * 12 + [0.9] * 18 + [0.1] * 20)                  # the walk-back caps at 8 frames
    assert ev[0].at == 4 * FRAME, ev

    # --- mute/unmute: events are dropped while the robot speaks; speech running at unmute is clipped
    probs = [0.05] * 3 + [0.9] * 8 + [0.1] * 12 + [0.05] * 4 + [0.9] * 14 + [0.1] * 20
    seg = Segmenter(Scripted(probs))
    zeros, ev, fed = np.zeros(2048, np.float32), [], 0
    while not ev:                       # speech starts after the previous unmute, with real onset
        ev += seg.feed(zeros)
        fed += len(zeros)
    assert isinstance(ev[0], Onset) and ev[0].at == 3 * FRAME, (ev, fed)
    seg.mute()                          # the robot speaks: the open segment is dropped silently
    while fed < 29 * FRAME:             # silence, its close and the next onset all suppressed
        assert seg.feed(zeros) == []
        fed += len(zeros)
    got = seg.unmute(32 * FRAME)        # mid-speech of the burst: the participant was already talking
    assert got is not None and got.at == 32 * FRAME, got
    while len(ev) < 2:
        ev += seg.feed(zeros)
        fed += len(zeros)
    assert isinstance(ev[1], Segment) and ev[1].clipped and ev[1].onset == 32 * FRAME, ev
    # speech starting later: its onset stays real, only the unmute point is the earliest allowed
    probs = [0.05] * 2 + [0.9] * 10 + [0.1] * 25 + [0.05] * 5 + [0.9] * 8 + [0.1] * 20
    seg = Segmenter(Scripted(probs))
    ev, fed = [], 0
    while not ev:
        ev += seg.feed(zeros)
        fed += len(zeros)
    assert ev[0].at == 2 * FRAME, ev
    seg.mute()
    while fed < 38 * FRAME:             # unmute in the 0.05 stretch, long before any speech
        assert seg.feed(zeros) == []
        fed += len(zeros)
    assert seg.unmute(38 * FRAME) is None
    while len(ev) < 3:
        ev += seg.feed(zeros)
        fed += len(zeros)
    assert isinstance(ev[1], Onset) and ev[1].at == 42 * FRAME, ev   # never before the unmute sample
    assert isinstance(ev[2], Segment) and not ev[2].clipped and ev[2].onset == 42 * FRAME, ev
    print("speech self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--test", action="store_true", help="live mic: ready line, level, utterances")
    ap.add_argument("--file", help="offline: a 16 kHz mono 16-bit WAV through the same pipeline")
    ap.add_argument("--lang", default="en", choices=("en", "hu"), help="Canary language to decode with")
    ap.add_argument("--device", type=int, help="input device index (from --list-devices)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    try:
        if args.list_devices:
            return _devices()
        if args.file:
            return _file(args.file, args.lang)
        if args.test:
            return _test(args)
    except SpeechUnavailable as e:
        sys.exit(str(e))
    if args.self_test:
        return self_test()
    ap.error("give one of --list-devices / --test / --file / --self-test")


if __name__ == "__main__":
    main()
