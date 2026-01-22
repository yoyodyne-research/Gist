# Gist: Bela + Features + NCP — Progress and Next Steps

This document captures the current status of the Bela audio_features project, the feature extraction front‑end(s), and the neural controller plan; plus a concrete roadmap of next steps.

## Status Summary (Sept 2025)
- Build/integration
  - Cross‑platform CMake builds (Torch trainer scaffold added in `trainer/`)
  - Bela deploy script (`scripts/deploy_bela.sh`) robust: stages sources, CARFAC, vendored Eigen, and runs the project
  - CI/IDE: project recognized in Bela; settings expose defines/flags and per‑project parameters

- FFT backend and SIMD
  - Bela examples default to PFFFT; NEON active on Bela (verified via runtime log)
  - Core library default remains KISS FFT (see `src/CMakeLists.txt` options)
  - You can switch backends in two places:
    - Core/desktop: CMake options `-DUSE_KISS_FFT|USE_PFFFT|USE_FFTW|USE_ACCELERATE_FFT`
    - Bela examples: `examples/bela/settings.json` defines/cflags

- Primary feature front‑end(s)
  - Flenser (SVF) multiband envelopes: per‑band fast/slow AR envelopes + transient; heuristic per‑band timing; presets for guitar/bass
  - CARFAC (cochlear model) front‑end integrated (Apache‑2.0 upstream vendored as submodule)
    - Wrapper (`examples/bela/carfac_frontend.*`) runs CAR→IHC→AGC at hop cadence via `RunSegment` and publishes per‑band envelopes/transients
    - Eigen headers vendored into repo and staged project for easy Bela builds
    - Frontend A/B: `ENABLE_CARFAC_FRONTEND` selects CARFAC vs Flenser; logs show the active path

- Pitch
  - Bitstream Autocorrelation (BACF) implemented (float-only) with fast path; now used in aux task; pitch cadence at every 4 hops by default
  - Optional per‑band BACF refinement left as a future enhancement

- MFCC
  - Disabled by default at compile time (`ENABLE_MFCC=0`); clean toggling and aux‑task computation path retained

- NCP/CfC training (PyTorch)
  - `trainer/` scaffold: data schema (NPZ), drum→curves target generator, Tiny CfC + FiLM model, training loop (MAE + TV + range/slew), exporter (flat weights)
  - Model uses ncps.torch CfC only (no GRU fallback)
  - Checkpoint inspector to list keys/shapes for C++ mapping
  - Export manifest: `trainer/export.py` writes a stable set of tensor names/shapes used by `examples/bela/ncp_infer.*` for C++ mapping

- Bela NCP runtime
  - Inference stub present (`examples/bela/ncp_infer.*`) with FiLM and post‑processing; currently contains a GRU placeholder to validate the pipeline shape; to be replaced with CfC closed‑form forward

## Current Bela Runtime (defaults)
- Frame/hop: 256/64 samples
- CARFAC front end (if enabled): 16 bands, per‑sample buffering, `RunSegment` every publish (default every 256 samples)
- Published features (@ ~172 Hz at 44.1 kHz; 44100/256 ≈ 172.3): per‑band env_fast/env_slow/transient
- Global features: RMS every hop; centroid/rolloff/HFC every 4 hops; MFCC disabled
- Pitch: BACF on aux task every 4 hops; decoupled from RT thread
- CPU: ~29–32% with CARFAC at 16 bands (measured) on Bela NEON

## Core Library Build/Tests
- Configure and build (desktop):
  - `mkdir build && cd build`
  - `cmake .. -D BUILD_TESTS=ON` (defaults to KISS FFT)
  - `cmake --build .`
- Run tests:
  - `cd build && ctest -C Debug -VV`
- FFT backend (core/desktop): select exactly one via CMake options
  - `-DUSE_KISS_FFT` (default) | `-DUSE_PFFFT` | `-DUSE_FFTW` | `-DUSE_ACCELERATE_FFT`

## Testing (doctest)
- Framework: doctest (header‑only) with tests in `tests/`
- Recommended near‑term coverage additions:
  - BACF: lag search, cadence control, edge cases (silence/low SNR)
  - MFCC: toggled disabled/enabled behavior parity (no RT regressions when off)
  - Frontends: envelope invariants (non‑negative, bounded, monotonic decay on silence) for CARFAC and Flenser

## Desirable Next Steps (Prioritized)

1) CfC (NCP) on‑device inference (replace GRU stub)
   - Implement CfC closed‑form forward in `ncp_infer.*` to match ncps.torch export
   - Export mapping: add CfC tensors to `trainer/export.py`; finalize manifest
   - Add `ENABLE_NCP` flag in `render.cpp`; tick model every publish; map to MIDI/OSC lanes; preserve safety (clamp/slew/one‑pole)

2) Feature logger + dataset tooling (Mac)
   - UDP/OSC/TCP logger pulling Bela features/macros + drum MIDI (time-aligned)
   - NPZ writer producing {features, macros, meta(BPM/phase/genre/rhythm), targets}
   - Small CLI: record, visualize, and sanity‑check clips

3) Online training gate and swapper
   - “Learn” toggle; only update when bass+drums active
   - Mini‑batches from a rolling buffer; Adam + trunc. BPTT; replay weight
   - Safety checks on outputs (range/slew/silence stability) before hot‑swap
   - Versioned weights + atomic swap on Bela; A/B fallback

4) Conditioning (FiLM) metadata
   - Build conditioning vector c = [norm_bpm, sin/cos beat phase, genre embed, Toussaint rhythm descriptors]
   - Train CfC + FiLM with c present; use c from live drum machine when available; freeze c otherwise

5) CARFAC tuning and features
   - Channels: 16→24 with CPU checks; expose per‑band AGC gain as optional features
   - Envelope timing: adjust wrapper AR times to taste; evaluate IHC vs BM as envelope source
   - Warm‑up strategy (AGC): 200–500 ms before feature confidence gates

6) Model head design and mapping
   - Define initial control lanes (4–8) and mapping to rig (e.g., drive, punch, sparkle, space)
   - Optional instrument aux heads (kick/snare/hats envelopes) to multi-task supervise CfC trunk

7) Feature cadence/latency experiments
   - Try publish at 128 samples (~344 Hz) or 64 samples (~688 Hz) if CPU allows
   - Explore per‑sample CAR/IHC/AGC stepping path (Ear::Step methods) if lower latency envelopes needed

8) Per‑band BACF refinement (optional)
   - Use a few candidate lags from full‑band BACF; validate across bands; weighted vote to refine pitch; publish per‑band periodicity as confidence

## Troubleshooting & Notes
- Enabling CARFAC
  - Set `HAVE_CARFAC=1` and `ENABLE_CARFAC_FRONTEND=1` (settings.json)
  - Auto‑detect fallback: if `carfac/upstream/cpp/carfac.h` is staged, frontend enables automatically
  - Eigen headers are required; deploy script vendors Eigen into staged project and repo (`libs/carfac/eigen/Eigen`)
- FFT defaults and toggles
  - Core library default: KISS FFT; switch via CMake options in `src/CMakeLists.txt`
  - Bela examples default: PFFFT; switch via `examples/bela/settings.json` defines/cflags
- Platform flags
  - NEON flags in `examples/bela/settings.json` are ARM‑specific. For non‑ARM builds, remove the NEON defines/flags (see `examples/bela/README.md`).
- CI scope
  - CI builds the desktop library and runs `ctest`. Bela example projects are not executed in CI; use `scripts/deploy_bela.sh` for on‑device smoke tests.
- Logs
  - Frontend selection prints: “Frontend: CARFAC” or “Frontend: Flenser (SVF)”
  - Bands print: “CARFAC bands=N” or “Flenser bands=N” once at setup
- MFCC
  - Toggle via `ENABLE_MFCC=0|1`; when enabled runs in aux task only
- Pitch
  - BACF cadence and range controlled via macros; Flenser/carfac front‑end changes do not affect BACF unless per‑band refinement is added later

## Open Questions
- How many bands and which envelope time constants feel best per preset (Guitar, Bass4, Bass6, Baritone, etc.)?
- Do we want AGC gain per band exposed to the NCP as context features?
- What is the minimal, robust control lane set for v1 (4 vs 6 vs 8 lanes)?
- Preferred conditioning set (BPM/phase/genre/rhythm) and macro knobs (Brightness, Density, Sustain, Aggression)?

---
For any integration or build issues, see:
- `scripts/deploy_bela.sh` (staging & vendoring logic)
- `examples/bela/settings.json` (defines/flags/paths)
- `examples/bela/carfac_frontend.*` (cochlear front‑end wrapper)
- `trainer/` (PyTorch CfC training pipeline)
