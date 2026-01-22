from __future__ import annotations
import argparse
import torch
from pathlib import Path
from model import TinyController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--in_dim", type=int, required=True)
    ap.add_argument("--macro_dim", type=int, default=0)
    ap.add_argument("--cond_dim", type=int, default=0)
    ap.add_argument("--hid", type=int, default=12)
    ap.add_argument("--enc_dim", type=int, default=16)
    ap.add_argument("--out_dim", type=int, required=True)
    args = ap.parse_args()

    m = TinyController(args.in_dim, args.macro_dim, args.cond_dim, hid=args.hid, enc_dim=args.enc_dim, out_dim=args.out_dim)
    sd = torch.load(args.ckpt, map_location="cpu")
    m.load_state_dict(sd)
    print("State dict keys and shapes:")
    for k, v in m.state_dict().items():
        print(f"{k:30s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()

