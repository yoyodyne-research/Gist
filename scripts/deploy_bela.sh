#!/usr/bin/env bash
set -euo pipefail

# Deploy a Bela project using Gist + PFFFT for real-time audio features.
# - Assembles a local project folder with required sources
# - rsyncs to the Bela board
# - Optionally builds and runs the project (with --run flag)
#
# Usage:
#   ./deploy_bela.sh         # Sync code only
#   ./deploy_bela.sh --run   # Sync, build, and run

RUN_PROJECT=false
if [[ "${1:-}" == "--run" || "${1:-}" == "-r" ]]; then
    RUN_PROJECT=true
fi

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
cp -v "$REPO_ROOT/src/NeonOps.h" "$LOCAL_PROJECT/" || true

# PFFFT (float-only NEON backend)
mkdir -p "$LOCAL_PROJECT/pffft/simd"
# Copy original C source and a C++ wrapper that forces NEON macros when compiling under Bela
cp -v "$REPO_ROOT/libs/pffft/pffft.c" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/examples/bela/pffft_neon.cpp" "$LOCAL_PROJECT/pffft/pffft.cpp"
cp -v "$REPO_ROOT/libs/pffft/pffft.h" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/libs/pffft/pffft_priv_impl.h" "$LOCAL_PROJECT/pffft/"
cp -v "$REPO_ROOT/libs/pffft/simd/pf_"*".h" "$LOCAL_PROJECT/pffft/simd/"

# CARFAC (if vendored). Copy any headers/sources into project carfac/ folder.
if [ -d "$REPO_ROOT/libs/carfac" ]; then
  EIGEN_VENDORED=""
  mkdir -p "$LOCAL_PROJECT/carfac"
  # Copy headers/sources from the vendored/submodule tree into the staged project
  # Avoid shell parameter expansion pitfalls inside xargs by using a while loop
  while IFS= read -r -d '' src; do
    rel_path="${src#"$REPO_ROOT/libs/carfac/"}"
    dest="$LOCAL_PROJECT/carfac/$rel_path"
    mkdir -p "$(dirname "$dest")"
    cp -v "$src" "$dest"
  done < <(find "$REPO_ROOT/libs/carfac" -maxdepth 4 \( -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cpp" \) -print0)

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
# Ensure a real directory (not a symlink) exists for the project
ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $REMOTE_PROJECT_ROOT; if [ -L '$REMOTE_PROJECT_DIR' ]; then rm -f '$REMOTE_PROJECT_DIR'; fi; mkdir -p '$REMOTE_PROJECT_DIR'"
rsync -avz --delete --exclude ".git" "$LOCAL_PROJECT/" "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_PROJECT_DIR/"

if $RUN_PROJECT; then
    echo "Building and running project on Bela..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd /root/Bela && make PROJECT=$REMOTE_PROJECT_NAME run"
    echo "Done. Project running on Bela as: $REMOTE_PROJECT_NAME"
else
    echo "Done. Code synced to Bela. To build and run:"
    echo "  ssh root@$REMOTE_HOST 'cd /root/Bela && make PROJECT=$REMOTE_PROJECT_NAME run'"
    echo "Or re-run with: $0 --run"
fi
