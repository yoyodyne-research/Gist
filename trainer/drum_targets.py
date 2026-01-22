from __future__ import annotations
import numpy as np
import mido
from typing import Dict, Any, Tuple


def exp_kernel(sr: float, attack_ms: float, decay_ms: float, length_s: float) -> np.ndarray:
    n = int(length_s * sr)
    t = np.arange(n) / sr
    a = max(1e-3, attack_ms / 1000.0)
    d = max(1e-3, decay_ms / 1000.0)
    k = np.where(t < a, t / a, np.exp(-(t - a) / d))
    return k.astype(np.float32)


def midi_to_events(midi_path: str, ticks_per_beat: float) -> Dict[str, np.ndarray]:
    mid = mido.MidiFile(midi_path)
    tempo = 500000  # default 120 BPM
    time = 0.0
    events = {"kick": [], "snare": [], "hats": [], "ride": [], "toms": []}
    for msg in mid:  # type: ignore
        time += mido.tick2second(msg.time, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            ch = msg.channel if hasattr(msg, "channel") else 9
            note = msg.note
            vel = msg.velocity / 127.0
            # naive mapping; customize as needed
            tgt = None
            if note in (35, 36):
                tgt = "kick"
            elif note in (38, 40):
                tgt = "snare"
            elif 42 <= note <= 46:
                tgt = "hats"
            elif note in (51, 59):
                tgt = "ride"
            elif 41 <= note <= 50:
                tgt = "toms"
            if tgt:
                events[tgt].append((time, vel))
    return {k: np.array(v, dtype=np.float32) if v else np.zeros((0, 2), dtype=np.float32) for k, v in events.items()}


def synthesize_curves(events: Dict[str, np.ndarray], sr: float, length_s: float, bpm: float) -> Dict[str, np.ndarray]:
    curves: Dict[str, np.ndarray] = {}
    # tempo-aware kernel lengths
    beat_s = 60.0 / max(1e-3, bpm)
    kernels = {
        "kick": exp_kernel(sr, attack_ms=5, decay_ms=max(80, 200 * beat_s), length_s=2 * beat_s),
        "snare": exp_kernel(sr, attack_ms=5, decay_ms=max(100, 250 * beat_s), length_s=2 * beat_s),
        "hats": exp_kernel(sr, attack_ms=2, decay_ms=max(40, 120 * beat_s), length_s=1.5 * beat_s),
        "ride": exp_kernel(sr, attack_ms=3, decay_ms=max(60, 180 * beat_s), length_s=2 * beat_s),
        "toms": exp_kernel(sr, attack_ms=4, decay_ms=max(100, 220 * beat_s), length_s=2 * beat_s),
    }
    n = int(length_s * sr)
    for name, ev in events.items():
        y = np.zeros(n, dtype=np.float32)
        k = kernels[name]
        for t, v in ev:
            i = int(t * sr)
            j = min(n, i + len(k))
            seg = k[: j - i] * v
            y[i:j] += seg
        # normalize per instrument to [0,1]
        if y.max() > 1e-6:
            y /= y.max()
        curves[name] = y
    return curves


def merge_to_control(curves: Dict[str, np.ndarray]) -> np.ndarray:
    # Example mapping to 4 control lanes: [drive, punch, sparkle, space]
    kick = curves.get("kick"); snare = curves.get("snare"); hats = curves.get("hats"); ride = curves.get("ride"); toms = curves.get("toms")
    # Ensure arrays exist
    L = max((len(v) for v in curves.values()), default=0)
    def g(x):
        return x if x is not None and len(x) == L else np.zeros(L, dtype=np.float32)
    kick = g(kick); snare = g(snare); hats = g(hats); ride = g(ride); toms = g(toms)
    drive   = 0.8 * kick + 0.2 * toms
    punch   = 0.7 * snare + 0.3 * toms
    sparkle = 0.7 * hats + 0.3 * ride
    space   = 0.5 * ride + 0.3 * toms + 0.2 * hats
    Y = np.stack([drive, punch, sparkle, space], axis=-1)
    Y = np.clip(Y, 0.0, 1.0)
    return Y.astype(np.float32)


def midi_to_curves(midi_path: str, sr: float, length_s: float, bpm: float, ticks_per_beat: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    events = midi_to_events(midi_path, ticks_per_beat)
    curves = synthesize_curves(events, sr, length_s, bpm)
    control = merge_to_control(curves)
    return curves, control

