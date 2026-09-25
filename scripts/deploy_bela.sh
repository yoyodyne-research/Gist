#!/usr/bin/env bash
set -euo pipefail

# Deploy a Bela project using Gist + PFFFT for real-time audio features.
# - Assembles a local project folder with required sources
# - rsyncs to the Bela board
# - Optionally builds and runs the project (with --run flag)
#
# Usage:
#   ./deploy_bela.sh                       # Sync code only
#   ./deploy_bela.sh --run                 # Sync, build, and run
#   ./deploy_bela.sh --run --scope         # With Bela Scope enabled
#   ./deploy_bela.sh --run --carfac-rate 22050  # CARFAC at half sample rate

RUN_PROJECT=false
USE_SCOPE=false
USE_LATENT=false
CARFAC_RATE=0  # 0 = same as audio rate
LATENT_MODEL=""  # Path to latent_model.json

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run|-r)
            RUN_PROJECT=true
            shift
            ;;
        --scope|--use-scope|--use_scope)
            USE_SCOPE=true
            shift
            ;;
        --latent|--use-latent|--use_latent)
            USE_LATENT=true
            shift
            ;;
        --latent-model)
            LATENT_MODEL="$2"
            USE_LATENT=true
            shift 2
            ;;
        --carfac-rate|--carfac_rate)
            CARFAC_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--run|-r] [--scope] [--latent] [--latent-model PATH] [--carfac-rate RATE]" >&2
            exit 1
            ;;
    esac
done

# Configurable via env vars
REMOTE_HOST=${REMOTE_HOST:-bela.local}
REMOTE_USER=${REMOTE_USER:-root}
REMOTE_DEV_DIR=${REMOTE_DEV_DIR:-/root/dev}
REMOTE_PROJECT_ROOT=${REMOTE_PROJECT_ROOT:-/root/Bela/projects}
REMOTE_PROJECT_NAME=${REMOTE_PROJECT_NAME:-audio_features}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# Default local project folder inside this repo (override with LOCAL_PROJECT)
LOCAL_PROJECT=${LOCAL_PROJECT:-"$REPO_ROOT/.bela_project"}
REMOTE_PROJECT_DIR="$REMOTE_PROJECT_ROOT/$REMOTE_PROJECT_NAME"

echo "Assembling local project at: $LOCAL_PROJECT"
mkdir -p "$LOCAL_PROJECT"

# Copy Bela example files
cp -v "$REPO_ROOT/examples/bela/render.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/examples/bela/settings.json" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/examples/bela/carfac_frontend.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/examples/bela/carfac_frontend.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/examples/bela/pffft_alloc.c" "$LOCAL_PROJECT/"

# Copy latent layer files (if enabled)
if $USE_LATENT; then
    cp -v "$REPO_ROOT/examples/bela/latent_layer.h" "$LOCAL_PROJECT/"
    cp -v "$REPO_ROOT/examples/bela/latent_layer.cpp" "$LOCAL_PROJECT/"
    echo "Latent layer: ENABLED"

    # Copy latent model if specified
    if [ -n "$LATENT_MODEL" ] && [ -f "$LATENT_MODEL" ]; then
        cp -v "$LATENT_MODEL" "$LOCAL_PROJECT/latent_model.json"
        echo "Latent model: $LATENT_MODEL"
    else
        echo "Warning: No latent model specified. Use --latent-model PATH to include trained model."
    fi
fi

# Copy Gist sources (avoid Apple Accelerate files)
cp -v "$REPO_ROOT/src/Gist.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/CoreTimeDomainFeatures.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/CoreFrequencyDomainFeatures.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/OnsetDetectionFunction.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/MFCC.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/WindowFunctions.cpp" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/Yin.cpp" "$LOCAL_PROJECT/"

# Headers
cp -v "$REPO_ROOT/src/Gist.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/CoreTimeDomainFeatures.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/CoreFrequencyDomainFeatures.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/OnsetDetectionFunction.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/MFCC.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/WindowFunctions.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/Yin.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/BitstreamPitch.h" "$LOCAL_PROJECT/"
cp -v "$REPO_ROOT/src/NeonOps.h" "$LOCAL_PROJECT/" || true

# PFFFT (float-only NEON backend)
mkdir -p "$LOCAL_PROJECT/pffft/simd"
# Copy original C source and a C++ wrapper that forces NEON macros when compiling under Bela
cp -v "$REPO_ROOT/libs/pffft/pffft.c" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/examples/bela/pffft_neon.cpp" "$LOCAL_PROJECT/pffft/pffft.cpp"
cp -v "$REPO_ROOT/libs/pffft/pffft.h" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/libs/pffft/pffft_priv_impl.h" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/libs/pffft/simd/pf_"*".h" "$LOCAL_PROJECT/pffft/simd/"

# CARFAC (if vendored). Copy headers to carfac/ subfolder, but sources to project root for Bela to compile.
if [ -d "$REPO_ROOT/libs/carfac" ]; then
  EIGEN_VENDORED=""
  mkdir -p "$LOCAL_PROJECT/carfac"

  # Copy headers to carfac/ subfolder (preserving structure for includes)
  while IFS= read -r -d '' src; do
    rel_path="${src#"$REPO_ROOT/libs/carfac/"}"
    dest="$LOCAL_PROJECT/carfac/$rel_path"
    mkdir -p "$(dirname "$dest")"
    cp -v "$src" "$dest"
  done < <(find "$REPO_ROOT/libs/carfac" -maxdepth 4 \( -name "*.h" -o -name "*.hpp" \) -print0)

  # Copy CARFAC source AND header files to project ROOT so Bela compiles them
  # Sources use local includes like #include "carfac.h"
  # Rename .cc to .cpp since Bela only compiles .cpp files
  for src in car.cc carfac.cc ear.cc; do
    if [ -f "$REPO_ROOT/libs/carfac/upstream/cpp/$src" ]; then
      dst="${src%.cc}.cpp"
      cp -v "$REPO_ROOT/libs/carfac/upstream/cpp/$src" "$LOCAL_PROJECT/$dst"
    fi
  done
  for hdr in car.h carfac.h ear.h agc.h ihc.h common.h carfac_util.h; do
    if [ -f "$REPO_ROOT/libs/carfac/upstream/cpp/$hdr" ]; then
      cp -v "$REPO_ROOT/libs/carfac/upstream/cpp/$hdr" "$LOCAL_PROJECT/"
    fi
  done
  echo "CARFAC sources (.cpp) and headers copied to project root"

  # Try to vendor Eigen headers if available on this machine so CARFAC builds on Bela.
  # Check EIGEN3_INCLUDE_DIR first, then common Homebrew/system locations.
  EIGEN_CANDIDATES=(
    "${EIGEN3_INCLUDE_DIR:-}"
    "/opt/homebrew/include/eigen3"
    "/usr/local/include/eigen3"
    "/usr/include/eigen3"
  )
  for root in "${EIGEN_CANDIDATES[@]}"; do
    if [ -n "$root" ] && [ -d "$root/Eigen" ] && [ -f "$root/Eigen/Core" ]; then
      echo "Vendoring Eigen headers from: $root"
      mkdir -p "$LOCAL_PROJECT/carfac/eigen"
      # Prefer rsync, fallback to cp -R
      if command -v rsync >/dev/null 2>&1; then
        rsync -av "$root/Eigen" "$LOCAL_PROJECT/carfac/eigen/" || true
      else
        cp -R "$root/Eigen" "$LOCAL_PROJECT/carfac/eigen/" || true
      fi
      if [ -f "$LOCAL_PROJECT/carfac/eigen/Eigen/Core" ]; then
        echo "Eigen vendored to: $LOCAL_PROJECT/carfac/eigen/Eigen"
        # Also vendor into the repo so it persists (optional, skip if already exists)
        if [ ! -f "$REPO_ROOT/libs/carfac/eigen/Eigen/Core" ]; then
          echo "Copying Eigen headers into repo at libs/carfac/eigen/Eigen (one-time vendor)"
          mkdir -p "$REPO_ROOT/libs/carfac/eigen"
          if command -v rsync >/dev/null 2>&1; then
            rsync -av "$root/Eigen" "$REPO_ROOT/libs/carfac/eigen/" || true
          else
            cp -R "$root/Eigen" "$REPO_ROOT/libs/carfac/eigen/" || true
          fi
        fi
        EIGEN_VENDORED=1
        break
      else
        echo "Warning: attempted to vendor Eigen, but Core not found at destination" >&2
      fi
    fi
  done
  if [ -z "$EIGEN_VENDORED" ]; then
    echo "Warning: Eigen headers not found on host. If CARFAC build fails on Bela due to Eigen/Core, set EIGEN3_INCLUDE_DIR or install eigen3 on this machine to vendor headers." >&2
  fi

  # Ensure any repo-vendored Eigen is also copied to the staged project
  if [ -d "$REPO_ROOT/libs/carfac/eigen/Eigen" ]; then
    echo "Copying repo-vendored Eigen to staged project"
    mkdir -p "$LOCAL_PROJECT/carfac/eigen"
    if command -v rsync >/dev/null 2>&1; then
      rsync -av "$REPO_ROOT/libs/carfac/eigen/Eigen" "$LOCAL_PROJECT/carfac/eigen/" || true
    else
      cp -R "$REPO_ROOT/libs/carfac/eigen/Eigen" "$LOCAL_PROJECT/carfac/eigen/" || true
    fi
    # Also place Eigen at project root so #include <Eigen/Core> resolves via -I.
    mkdir -p "$LOCAL_PROJECT/Eigen"
    if command -v rsync >/dev/null 2>&1; then
      rsync -av "$REPO_ROOT/libs/carfac/eigen/Eigen/" "$LOCAL_PROJECT/Eigen/" || true
    else
      cp -R "$REPO_ROOT/libs/carfac/eigen/Eigen/"* "$LOCAL_PROJECT/Eigen/" || true
    fi
  fi
fi
# Also place a top-level shim so "#include \"pffft.h\"" resolves reliably
cp -v "$REPO_ROOT/libs/pffft/pffft.h" "$LOCAL_PROJECT/"

# Sanity check
if [ ! -s "$LOCAL_PROJECT/render.cpp" ] || [ ! -s "$LOCAL_PROJECT/Gist.cpp" ]; then
  echo "Error: Project assembly failed (missing sources)." >&2
  exit 1
fi

echo "Syncing to Bela projects: $REMOTE_HOST:$REMOTE_PROJECT_DIR"
if $USE_SCOPE; then
    echo "Bela Scope: ENABLED (--scope flag)"
else
    echo "Bela Scope: disabled (use --scope to enable)"
fi
if $USE_LATENT; then
    echo "Latent layer: ENABLED (--latent flag)"
else
    echo "Latent layer: disabled (use --latent to enable)"
fi
if [[ "$CARFAC_RATE" -gt 0 ]]; then
    echo "CARFAC rate: ${CARFAC_RATE} Hz (decimated)"
else
    echo "CARFAC rate: full (same as audio rate)"
fi
# Ensure a real directory (not a symlink) exists for the project
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $REMOTE_PROJECT_ROOT; if [ -L '$REMOTE_PROJECT_DIR' ]; then rm -f '$REMOTE_PROJECT_DIR'; fi; mkdir -p '$REMOTE_PROJECT_DIR'"
rsync -avz --delete --exclude ".git" "$LOCAL_PROJECT/" "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_PROJECT_DIR/"

# Compiler flags for Bela make (settings.json is only for IDE)
# Set ENABLE_SCOPE based on --scope flag
if $USE_SCOPE; then
    SCOPE_FLAG="-DENABLE_SCOPE=1"
else
    SCOPE_FLAG="-DENABLE_SCOPE=0"
fi

# Set ENABLE_LATENT based on --latent flag
if $USE_LATENT; then
    LATENT_FLAG="-DENABLE_LATENT=1"
else
    LATENT_FLAG="-DENABLE_LATENT=0"
fi

BELA_CPPFLAGS="-DUSE_PFFFT -DUSE_ARM_NEON -DPFFFT_ENABLE_NEON -D__ARM_NEON -D__arm__ -DUSE_BITSTREAM_PITCH -DPITCH_PRESET=6 -DENABLE_MFCC=0 $SCOPE_FLAG $LATENT_FLAG -DCARFAC_RATE=$CARFAC_RATE -DHAVE_CARFAC -DPFFFT_SILENCE_SIMD_MSG -DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_DEBUG -DEIGEN_MALLOC_ALREADY_ALIGNED=0"
BELA_CFLAGS="-march=armv7-a -O3 -ffast-math -fno-math-errno -ftree-vectorize -Wno-#pragma-messages -mfpu=neon-vfpv3 -mcpu=cortex-a8 -mfloat-abi=hard"
BELA_CXXFLAGS="-std=c++11 $BELA_CFLAGS"
BELA_INCLUDES="-I. -Ipffft -Ipffft/simd -Icarfac -Icarfac/upstream -Icarfac/upstream/cpp -Icarfac/eigen"

# Runtime arguments (block size, etc.)
BELA_RUN_ARGS=${BELA_RUN_ARGS:-"-p 16 -B 16 -C 8 -N 1 -G 1"}

if $RUN_PROJECT; then
    echo "Building and running project on Bela..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd /root/Bela && make PROJECT=$REMOTE_PROJECT_NAME CPPFLAGS='$BELA_CPPFLAGS $BELA_INCLUDES' CFLAGS='$BELA_CFLAGS' CXXFLAGS='$BELA_CXXFLAGS' CL='$BELA_RUN_ARGS' run"
    echo "Done. Project running on Bela as: $REMOTE_PROJECT_NAME"
else
    echo "Done. Code synced to Bela. To build and run:"
    echo "  ssh root@$REMOTE_HOST 'cd /root/Bela && make PROJECT=$REMOTE_PROJECT_NAME CPPFLAGS=\"$BELA_CPPFLAGS $BELA_INCLUDES\" CFLAGS=\"$BELA_CFLAGS\" CXXFLAGS=\"$BELA_CXXFLAGS\" CL=\"$BELA_RUN_ARGS\" run'"
    echo "Or re-run with: $0 --run"
fi
