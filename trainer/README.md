Trainer scaffold for NCP/CfC-based control (Mac mini)

Overview
- Trains a tiny recurrent controller (CfC via ncps, or GRU fallback) to map your feature stream (Gist + Flenser) to smooth control curves derived from MIDI drum grooves.
- Supports FiLM conditioning from metadata/macros (e.g., BPM, genre, Toussaint rhythm descriptors) and artist macros (Brightness, Density, Sustain, Aggression).
- Gate training so updates happen only when both bass and drums are active.
- Exports flat weights + normalization for a minimal C++ inference stub on Bela.

Repo layout
- model.py: Tiny controller (CfC via ncps) + FiLM conditioning + multi-head readout (no GRU fallback)
- drum_targets.py: MIDI→curves generator (kick/snare/hats buses → smooth envelopes)
- schema.py: NPZ schema, normalization helpers, gating (bass+drums) utilities
- train.py: Training loop with gating, losses (L1/L2, TV, range/slew), logging, checkpoints
- export.py: Serialize weights and normalization to a compact JSON/NPZ blob

Environment
1) Python venv
   python3 -m venv .venv && source .venv/bin/activate

2) Install dependencies
   pip install -r requirements.txt
   # If ncps is cloned alongside Gist (recommended):
   pip install -e ../ncps

Data schema (NPZ)
- features: float32 [T, D_in]  (normalized per training stats)
- macros:   float32 [T, D_m]   (artist controls; may be zero if unused)
- meta:     float32 [T, D_c]   (conditioning: BPM, beat phase sin/cos, rhythm descriptors)
- targets:  float32 [T, D_out] (drum-derived curves mapped to control lanes)
- ts:       int64   [T]        (timestamps or frame indices)
- info:     JSON/str metadata  (tempo, genre tag, sample rate, hop, etc.)

Targets
- Use drum_targets.py:midi_to_curves() to convert MIDI into instrument envelopes then to control lanes.
- Kernels are tempo-aware and genre-tunable; envelopes are normalized to [0,1].

Training
- Edit train.py opts (input dims, hidden width, heads, loss weights).
- Run: python train.py --data your.npz --epochs 50 --device cpu
- Checkpoints save to runs/; best JSON/NPZ for Bela in exports/.

Export
- export.py writes:
  - weights.npz: flat arrays for encoder, rnn (CfC/GRU), heads, FiLM
  - norm.json: input/output normalization (mean/std/min/max)
  - model.json: shapes, architecture flags

Deployment (Bela)
- C++ inference stub (examples/bela/ncp_infer.*) consumes weights and norm, ticks at feature cadence.
- Post-processing (clamp, slew, one-pole) applied per lane.
- Hot-swap: copy updated weights to project and trigger a light reload (TBD; see notes in ncp_infer.h).

Notes
- Start tiny: hidden=8–12, heads=4–8 lanes; add instrument aux heads later if desired.
- FiLM: keep γ≈1, β≈0 at init; apply to hidden and/or output head.
- Losses: combine L1/L2 with TV (smoothness) and hinge range/slew penalties for well-behaved outputs.
