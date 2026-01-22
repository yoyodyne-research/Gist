from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from schema import load_npz, Norm
from model import TinyController


class SeqDataset(Dataset):
    def __init__(self, npzs, seq_len: int = 512):
        self.data = [load_npz(p) for p in npzs]
        self.seq_len = seq_len

    def __len__(self):
        return sum(max(0, d["features"].shape[0] - self.seq_len) for d in self.data)

    def __getitem__(self, idx):
        # naive: iterate files until idx falls in range
        for d in self.data:
            T = d["features"].shape[0]
            n = max(0, T - self.seq_len)
            if idx < n:
                s = idx
                e = s + self.seq_len
                X = d["features"][s:e]
                M = d.get("macros", np.zeros((T, 0), np.float32))[s:e]
                C = d.get("meta", np.zeros((T, 0), np.float32))[s:e]
                Y = d["targets"][s:e]
                return torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(C), torch.from_numpy(Y)
            idx -= n
        raise IndexError


def tv_loss(y: torch.Tensor) -> torch.Tensor:
    return (y[1:] - y[:-1]).abs().mean()


def range_slew_penalty(y: torch.Tensor, ymin=0.0, ymax=1.0, dmax=0.1) -> torch.Tensor:
    p1 = (y - ymax).clamp_min(0).mean() + (ymin - y).clamp_min(0).mean()
    dy = (y[1:] - y[:-1]).abs()
    p2 = (dy - dmax).clamp_min(0).mean()
    return p1 + p2


def train(args):
    npz_paths = [str(p) for p in Path(args.data).glob("*.npz")] if os.path.isdir(args.data) else [args.data]
    ds = SeqDataset(npz_paths, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # dims (infer from first sample)
    X0, M0, C0, Y0 = next(iter(dl))
    in_dim = X0.shape[-1]
    macro_dim = M0.shape[-1]
    cond_dim = C0.shape[-1]
    out_dim = Y0.shape[-1]

    device = torch.device(args.device)
    model = TinyController(in_dim, macro_dim, cond_dim, hid=args.hid, enc_dim=args.enc_dim, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    l1 = nn.L1Loss()

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for X, M, C, Y in dl:
            X = X.to(device).transpose(0,1)  # [T,B,D]
            M = M.to(device)
            C = C.to(device)
            Y = Y.to(device).transpose(0,1)

            # broadcast macros/cond to [T,B,*]
            if M.dim() == 2:
                M = M.unsqueeze(0).expand(X.size(0), -1, -1)
            if C.dim() == 2:
                C = C.unsqueeze(0).expand(X.size(0), -1, -1)

            h = None
            pred, _ = model(X, h, M[0], C[0])  # FiLM on outputs per batch (macro stable over clip)
            loss_main = l1(pred, Y)
            loss_tv = tv_loss(pred)
            loss_rng = range_slew_penalty(pred)
            loss = loss_main + args.tv_w*loss_tv + args.rng_w*loss_rng

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()

        avg = total / max(1, len(dl))
        print(f"epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

        # checkpoint
        torch.save(model.state_dict(), run_dir / f"ckpt_{epoch:03d}.pt")

    # final export
    from export import export_flat
    export_dir = Path("exports") / args.run
    export_dir.mkdir(parents=True, exist_ok=True)
    export_flat(model, export_dir)
    print(f"Exported to {export_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="NPZ file or directory of NPZs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--hid", type=int, default=12)
    ap.add_argument("--enc_dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tv_w", type=float, default=0.05)
    ap.add_argument("--rng_w", type=float, default=0.05)
    ap.add_argument("--run", type=str, default="demo")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train(args)
