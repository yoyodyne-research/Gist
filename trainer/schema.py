from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class Norm:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_data(cls, x: np.ndarray, eps: float = 1e-6) -> "Norm":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < eps] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def invert(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean


def load_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    # info can be a 0-d array containing JSON string
    if "info" in d and isinstance(d["info"], np.ndarray) and d["info"].ndim == 0:
        try:
            d["info"] = json.loads(str(d["info"]))
        except Exception:
            pass
    return d


def save_npz(path: str, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)


def is_active_level(level: np.ndarray, thresh_db: float = -40.0) -> bool:
    """Return True if signal level (RMS in dBFS) exceeds threshold for a short window."""
    # level is linear; convert to dBFS safely
    eps = 1e-9
    db = 20.0 * np.log10(np.maximum(level, eps))
    # active if > thresh for at least 100 ms
    n = len(db)
    if n == 0:
        return False
    win = max(1, int(0.1 * n))
    return (db[-win:] > thresh_db).mean() > 0.2


def drums_present(midi_activity: np.ndarray, min_events: int = 3) -> bool:
    """Return True if recent MIDI activity exceeds a tiny count (kick/snare/hat hits)."""
    return int(midi_activity[-1]) >= min_events if len(midi_activity) else False

