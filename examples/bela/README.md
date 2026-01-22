# Bela + Gist Skeleton

This is a minimal Bela project skeleton that runs Gist in real time for guitar/bass features. It defaults to the PFFFT backend and enables NEON on ARM (Bela).

Quick start (on the Bela board)
- Create a new Bela project (e.g., `gist-bela`).
- Preferred: use `scripts/deploy_bela.sh` from this repo to assemble and sync a ready-to-run project (copies sources, PFFFT, and optional CARFAC/Eigen).
- Manual alternative: copy these files into the project: `render.cpp` and `settings.json`.
- Also copy the Gist sources and PFFFT into the same project folder:
  - From this repo `src/`: `Gist.cpp`, `CoreTimeDomainFeatures.cpp`, `CoreFrequencyDomainFeatures.cpp`, `OnsetDetectionFunction.cpp`, `MFCC.cpp`, `WindowFunctions.cpp`, `Yin.cpp` and all corresponding headers.
  - From `libs/pffft/`: `pffft.c` and `pffft.h`.
- Open the project in the Bela IDE or run the command-line build to compile and run.

Notes
- Backend: `USE_PFFFT` is enabled via `settings.json` and is float-only.
- Frame/hop: defaults to 256/64 (good trade-off for low latency and stability). Adjust in `render.cpp`.
- Real-time safety: no allocations or logging in `render()`. All buffers are preallocated in `setup()`.
- Features: example computes RMS, HFC, spectral centroid/rolloff, and YIN pitch.

FFT backends (Bela)
- Default: PFFFT via `examples/bela/settings.json` (`defines`, `cflags`, `cxxflags`) and bundled `pffft.c/h` sources.
- Switching away from PFFFT on Bela requires adapting sources and flags (e.g., adding KissFFT sources). For simplicity and performance on ARM, keep PFFFT enabled.

NEON flags (ARM-specific)
- `settings.json` enables NEON on Bela with defines/flags like `USE_ARM_NEON`, `PFFFT_ENABLE_NEON`, `__ARM_NEON`, and `-mfpu=neon-vfpv3 -mcpu=cortex-a8 -mfloat-abi=hard`.
- On non-ARM builds, remove these defines/flags to avoid build errors. See also the troubleshooting note below.

CARFAC frontend (optional)
- Enable by defining `HAVE_CARFAC=1` and `ENABLE_CARFAC_FRONTEND=1` in `settings.json`. The wrapper lives in `examples/bela/carfac_frontend.*`.
- Ensure CARFAC headers are present under `carfac/upstream/cpp` inside the project. The deploy script vendors these from `libs/carfac`.
- Eigen headers are required by CARFAC; the deploy script attempts to vendor them (or copies from `libs/carfac/eigen/Eigen`).

Troubleshooting
- If you see build errors related to NEON flags on non-ARM hosts, remove NEON-specific defines/flags in `settings.json` (`USE_ARM_NEON`, `PFFFT_ENABLE_NEON`, `__ARM_NEON`, and related `-mfpu/-mcpu/-mfloat-abi`).
- Make sure you copied all `.cpp` and `.h` files; missing sources will cause link errors.
