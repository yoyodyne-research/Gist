# Kitsune Latent Layer Design (PCA + GMM)

## Purpose

This document describes the **latent expression layer** for Kitsune: a real‑time, hardware‑friendly abstraction layer that converts high‑dimensional audio features into **continuous and discrete expressive control signals** suitable for:

* MIDI CC / Program Change
* Expression pedal emulation (digital potentiometers)
* Control Voltage (CV)

The goal is **stable, interpretable, low‑latency expressive signals**, not classification or end‑to‑end ML control.

---

## System Overview

```
Audio → Feature Extraction → PCA Projection → GMM → Latent Signals → Mapping → Hardware Outputs
```

Key principle:

> **Machine learning stays descriptive; musical meaning stays designed.**

---

## Input Features (Summary)

Primary features are derived from **CARFAC envelope outputs** (71 bands) and related statistics.

### Per‑Band Features

* `env_fast` — fast AR envelope (≈5 ms attack / 40 ms release)
* `delta` — signed fast–slow difference

  * positive → attack
  * negative → decay

### Cross‑Band Summary Statistics

* Total energy
* Spectral centroid
* Spectral spread
* Peak band
* Attack breadth
* Band energy ratios (low / mid / high)

### Pitch Features (Optional but Recommended)

* Pitch estimate (YIN or BACF)
* Pitch confidence / periodicity
* |df0/dt| (pitch motion)

**Total dimensionality:** ~140–150 dims (before projection)

---

## Timing & Rates

* Audio hop size: **64 samples @ 44.1 kHz**
* Feature frame rate: **~689 Hz**
* Latent / control update rate: **~200–250 Hz** (downsample by 3–4 hops)

This captures:

* pick rakes
* vibrato
* bends & slides
* articulation nuance

---

## Latent Layer (Core Design)

### 1. PCA Projection (Continuous Gesture Axes)

* PCA trained **offline** on a curated guitar dataset
* Inputs standardized using global μ / σ
* Retain **k = 8** dimensions (recommended range: 6–10)
* Optional: **Varimax rotation** for interpretability

**Runtime:** single matrix multiply per control tick

#### Properties

* Signed, continuous
* Zero‑mean
* Stable across sessions
* Extremely cheap to compute

#### Musical Interpretation (Typical)

| Latent | Interpretable Axis           |
| ------ | ---------------------------- |
| z1     | Overall excitation / effort  |
| z2     | Bright ↔ dark                |
| z3     | Transient ↔ sustained        |
| z4     | Harmonic ↔ noisy             |
| z5     | Low ↔ high register emphasis |
| z6     | Articulation motion          |
| z7–8   | Secondary correlations       |

These are **gesture axes**, not user controls.

---

### 2. Gaussian Mixture Model (Soft Technique Modes)

* GMM trained offline in PCA space
* **K = 12–16** mixture components recommended
* Diagonal covariances
* Fixed means / covariances for session stability

At runtime compute **responsibilities**:

```
r_i(t) = p(mode_i | z(t))
```

#### Properties

* Range: [0, 1]
* Sum to 1
* Smooth and blendable
* Interpretable as "how much this gesture is present"

Typical emergent modes:

* clean sustain
* muted / palm‑muted
* bright pick attack
* rake / strum
* slide
* vibrato sustain
* harmonics
* noise / scrape
* silence / near‑floor

Modes are *not* hard‑labeled; usefulness comes from soft activation.

---

## Additional Latent Signals

### Mode Selection

* `mode_id = argmax(r)`
* `mode_strength = max(r)`

Useful for discrete switching or gating.

### Confidence / Novelty

* Log‑likelihood: `log p(z)`
* Entropy: `H = -Σ r_i log r_i`

Uses:

* detect out‑of‑distribution gestures
* freeze mappings
* drive meta‑modulation

---

## Output Signals Available to Mapping Layer

The latent layer exposes the following **raw expressive signals**:

### Continuous

* PCA latents: `z[1..8]`
* GMM responsibilities: `r[1..K]`

### Semi‑Discrete

* `mode_id`
* `mode_strength`

### Meta

* entropy
* log‑likelihood

These are **not user controls** yet.

---

## Hardware‑Friendly Characteristics

All latent outputs are:

* bounded or easily scaled
* continuous (no discontinuities)
* low‑noise with light smoothing (20–40 ms)
* deterministic & real‑time safe

They map cleanly to:

* MIDI CC (0–127)
* bipolar or unipolar CV
* expression pedal resistance curves

---

## Separation of Concerns

| Layer              | Role                               |
| ------------------ | ---------------------------------- |
| Feature extraction | Descriptive audio analysis         |
| PCA / GMM          | Gesture & technique representation |
| Mapping layer      | Musical intent & UI design         |
| Hardware output    | Physical control signals           |

This separation allows:

* stable ML models
* flexible UI / mapping changes
* no retraining for new patches

---

## Recommended Defaults (v1)

* PCA dims: **8**
* GMM components: **14**
* Control rate: **~230 Hz**
* Responsibility smoothing: **30 ms**

---

## Key Design Philosophy

> Kitsune does not decide *what the musician means*.
> It measures *how they are playing* and provides expressive signals.

Musical meaning is created downstream.

---

## Open Design Questions

* Final feature subset for PCA training
* Exact GMM component count (BIC‑guided)
* Default latent → hardware mappings
* Cross‑instrument normalization strategies

---

*This document is intended for sharing with code agents and collaborators as a high‑level design reference.*
