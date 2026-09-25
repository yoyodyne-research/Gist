# Kitsune Setup Guide

This guide covers building and using the CARFAC cochlear frontend and Kitsune latent layer for real-time expressive audio feature extraction.

## System Overview

```
Audio Input → CARFAC (71 bands) → Feature Vector (150 dims) → Latent Layer → Control Signals
                                                                    ↓
                                                            PCA (8 latents)
                                                            GMM (14 modes)
```

**CARFAC**: Cochlear filterbank providing perceptually-weighted frequency analysis with fast/slow envelope tracking.

**Kitsune Latent Layer**: PCA + GMM model that compresses 150-dim features into 8 continuous latent axes and 14 soft "technique mode" responsibilities.

## Dependencies

### Desktop (Linux/macOS)

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libeigen3-dev

# Arch
sudo pacman -S base-devel cmake eigen

# macOS
brew install cmake eigen
```

Python (for training):
```bash
pip install numpy scikit-learn
```

### Bela

No additional dependencies—the deploy script bundles everything including Eigen headers.

## Building for Desktop

Desktop builds let you test the full pipeline without Bela hardware.

```bash
cd /home/jay/projects/Gist

# Create build directory
mkdir -p examples/desktop/build
cd examples/desktop/build

# Configure and build
cmake ..
cmake --build .
```

This produces:
- `kitsune_test` — Full audio processing pipeline
- `latent_test` — Latent layer unit tests

## Training the Latent Layer

Training happens offline on a desktop machine. The workflow:

1. Collect feature data from playing sessions
2. Train PCA (with Varimax rotation) on the features
3. Train GMM in PCA space
4. Export model to JSON for C++ runtime

### Quick Start (Synthetic Data)

For testing without real data:

```bash
cd /home/jay/projects/Gist

python3 << 'EOF'
import numpy as np
from trainer.latent import train_pca, train_gmm, export_model

# Generate synthetic features (replace with real data)
X = np.random.randn(5000, 150).astype(np.float32)

# Train PCA with Varimax rotation (8 components)
pca = train_pca(X, n_components=8, apply_varimax=True)
print(f"Explained variance: {sum(pca.ipca.explained_variance_ratio_):.1%}")

# Project to latent space
Z = pca.transform(X)

# Train GMM (14 components)
gmm = train_gmm(Z, n_components=14)

# Export for C++ runtime
export_model(pca, gmm, 'latent_model.json', smoothing_ms=30.0)
EOF
```

### Training with Real Data

Feature data should be collected from Bela during playing sessions. The expected format is a 150-dimensional vector per frame:

| Dims | Content |
|------|---------|
| 0–70 | `env_fast` (71 CARFAC band envelopes) |
| 71–141 | `delta` (fast − slow envelope difference) |
| 142 | Total energy |
| 143 | Spectral centroid |
| 144 | Spectral spread |
| 145 | Peak band |
| 146 | Attack breadth |
| 147–149 | Band ratios (low/mid/high) |

```python
from trainer.collect_features import load_features, load_multiple
from trainer.latent import train_pca, train_gmm, export_model

# Load collected data
X = load_multiple([
    'session1_features.npz',
    'session2_features.npz',
])
print(f"Loaded {len(X)} samples")

# Train
pca = train_pca(X, n_components=8, apply_varimax=True)
Z = pca.transform(X)
gmm = train_gmm(Z, n_components=14)

# Export
export_model(pca, gmm, 'latent_model.json')
```

### Choosing Component Counts

**PCA dimensions (default: 8)**: Check cumulative explained variance. 8 components typically capture 85–95% of variance in guitar signals.

**GMM components (default: 14)**: Use BIC to select:

```python
from trainer.latent.gmm import select_n_components

best_k = select_n_components(Z, min_k=8, max_k=20, criterion='bic')
print(f"Optimal components: {best_k}")
```

## Desktop Testing

### Process Audio Files

```bash
cd examples/desktop/build

# Process a WAV file, output latent signals to CSV
./kitsune_test process /path/to/audio.wav ../../../latent_model.json output.csv

# Print model info
./kitsune_test info ../../../latent_model.json
```

Output CSV columns:
```
frame, z0..z7, r0..r13, mode_id, mode_strength, entropy, log_likelihood
```

### Validate Against Python

Generate reference outputs from Python, then validate C++ matches:

```python
from trainer.latent.export import export_reference_outputs

# After training pca and gmm...
export_reference_outputs(pca, gmm, X, 'reference.npz', n_samples=100)
```

```bash
# Convert NPZ to CSV format expected by validator (or modify validator to read NPZ)
./kitsune_test validate ../../../latent_model.json reference.csv
```

### Unit Tests

```bash
./latent_test ../../../latent_model.json
```

Tests verify:
- PCA projection produces valid outputs
- GMM responsibilities sum to 1 and are in [0, 1]
- Smoothing behaves correctly
- Inference is fast enough for real-time (~230 Hz)

## Deploying to Bela

### Basic Deployment (CARFAC only)

```bash
cd /home/jay/projects/Gist

# Sync code to Bela
./scripts/deploy_bela.sh

# Sync, build, and run
./scripts/deploy_bela.sh --run
```

### With Latent Layer

```bash
# Deploy with latent layer enabled and trained model
./scripts/deploy_bela.sh --run --latent --latent-model latent_model.json
```

### With Scope Visualization

```bash
# Enable Bela Scope to visualize CARFAC bands
./scripts/deploy_bela.sh --run --scope
```

### CARFAC Rate Decimation

For reduced CPU usage, run CARFAC at half the audio sample rate:

```bash
./scripts/deploy_bela.sh --run --carfac-rate 22050
```

### All Options

| Flag | Description |
|------|-------------|
| `--run` | Build and run after syncing |
| `--latent` | Enable latent layer |
| `--latent-model PATH` | Path to trained model JSON |
| `--scope` | Enable Bela Scope visualization |
| `--carfac-rate N` | CARFAC processing rate (0 = full rate) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REMOTE_HOST` | `bela.local` | Bela hostname/IP |
| `REMOTE_USER` | `root` | SSH user |
| `REMOTE_PROJECT_NAME` | `audio_features` | Project name on Bela |

Example:
```bash
REMOTE_HOST=192.168.1.50 ./scripts/deploy_bela.sh --run
```

## Latent Layer Outputs

Once running, the latent layer produces these signals at ~230 Hz:

### Continuous Signals

| Signal | Range | Description |
|--------|-------|-------------|
| `z[0..7]` | unbounded | PCA latent axes (signed, zero-mean) |
| `r[0..13]` | [0, 1] | GMM responsibilities (sum to 1) |

### Discrete/Meta Signals

| Signal | Description |
|--------|-------------|
| `mode_id` | Index of strongest mode (argmax of r) |
| `mode_strength` | Confidence in current mode (max of r) |
| `entropy` | Uncertainty across modes (high = ambiguous) |
| `log_likelihood` | How well current frame fits the model |

### Typical Latent Interpretations

After Varimax rotation, PCA axes often align with interpretable concepts:

| Axis | Typical Meaning |
|------|-----------------|
| z0 | Overall energy/effort |
| z1 | Brightness (high frequencies) |
| z2 | Attack vs sustain |
| z3 | Harmonic vs noisy |
| z4–z7 | Secondary correlations |

GMM modes typically cluster into technique categories:
- Clean sustain
- Palm mute
- Pick attack
- Slide/bend
- Vibrato
- Harmonics
- Near-silence

## Troubleshooting

### Desktop Build Fails: Eigen Not Found

```
Could not find a package configuration file provided by "Eigen3"
```

Install Eigen3 or set the path manually:
```bash
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen3
```

### Bela Build Fails: Eigen/Core Not Found

The deploy script should vendor Eigen automatically. If it fails:
```bash
# Set path to Eigen on your machine
export EIGEN3_INCLUDE_DIR=/usr/include/eigen3
./scripts/deploy_bela.sh --run
```

### Latent Model Not Loading

Check the model path in `render.cpp`:
```cpp
const char* model_path = "/root/Bela/projects/audio_features/latent_model.json";
```

Verify the file exists on Bela:
```bash
ssh root@bela.local "ls -la /root/Bela/projects/audio_features/latent_model.json"
```

### High CPU Usage on Bela

- Use `--carfac-rate 22050` to halve CARFAC processing rate
- The latent layer runs at ~230 Hz and should add <1ms per tick
- Check with Bela's CPU monitor in the IDE

## File Reference

| File | Description |
|------|-------------|
| `examples/bela/carfac_frontend.h` | CARFAC wrapper + LatentInput struct |
| `examples/bela/carfac_frontend.cpp` | CARFAC processing + feature extraction |
| `examples/bela/latent_layer.h` | Latent layer class + LatentOutput struct |
| `examples/bela/latent_layer.cpp` | PCA/GMM inference implementation |
| `examples/bela/render.cpp` | Bela audio callback with integration |
| `examples/desktop/main.cpp` | Desktop test harness |
| `trainer/latent/pca.py` | PCA training with Varimax |
| `trainer/latent/gmm.py` | GMM training |
| `trainer/latent/export.py` | Model export to JSON |
| `trainer/collect_features.py` | Feature collection utilities |
| `scripts/deploy_bela.sh` | Deployment script |
| `docs/LATENT_LAYER_DESIGN.md` | Design rationale and specifications |
