from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def export_flat(model: nn.Module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = {}
    sd = model.state_dict()
    for k, v in sd.items():
        blob[k] = tensor_to_np(v)

    # Save weights in NPZ
    np.savez_compressed(out_dir / "weights.npz", **blob)

    # Save lightweight model meta
    meta = {
        "use_gru": getattr(model, "use_gru", False),
        "shapes": {k: list(v.shape) for k, v in sd.items()},
        "keys": list(sd.keys()),
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))

