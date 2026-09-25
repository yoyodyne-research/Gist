# CARFAC Decimation Design Analysis

This document captures the analysis and tradeoffs for sample rate decimation in the CARFAC cochlear frontend, specifically for real-time guitar processing on Bela (Cortex-A8, ARMv7).

---

## Problem Statement

Running CARFAC at full audio rate (44100 Hz) with full spectral resolution (83 bands, erb_per_step=1.0) exceeds 100% CPU on Bela's Cortex-A8, causing audio underruns.

**Goal**: Reduce CARFAC's compute load while preserving signal fidelity for envelope-following and transient detection.

---

## Current Implementation (as of 2026-01-26)

The current approach uses **integer decimation** with a **first-order IIR (one-pole) anti-aliasing filter**:

```cpp
// carfac_frontend.cpp

// Cutoff at 0.4 * Nyquist of target rate
float fc = 0.4f * (carfac_fs_ / 2.0f);
float rc = 1.0f / (2.0f * 3.14159f * fc);
float dt = 1.0f / fs_;
aa_coeff_ = dt / (rc + dt);

// Per-sample processing
aa_state_ += aa_coeff_ * (x - aa_state_);  // One-pole LPF
if (++decim_cnt_ < decim_) return;         // Decimate
decim_cnt_ = 0;
// ... run CARFAC at reduced rate
```

### Parameters for 2x Decimation (44100 -> 22050 Hz)

| Parameter | Value |
|-----------|-------|
| Input sample rate | 44100 Hz |
| CARFAC sample rate | 22050 Hz |
| Decimation factor | 2 |
| Filter cutoff | 4410 Hz (0.4 * 11025) |
| Filter order | 1 (one-pole) |
| Roll-off | -6 dB/octave |

### Results

| Config | Bands | CPU |
|--------|-------|-----|
| 44100 Hz, erb=1.0 | 83 | >100% (underrun) |
| 22050 Hz, erb=1.0 | 71 | ~55% |

---

## Analysis: One-Pole Filter Limitations

### Stopband Attenuation

A first-order filter rolls off at **-6 dB/octave** (-20 dB/decade). For our configuration:

| Frequency | Octaves Above Cutoff | Attenuation |
|-----------|---------------------|-------------|
| 4410 Hz (cutoff) | 0 | -3 dB |
| 8820 Hz | 1 | -9 dB |
| 11025 Hz (new Nyquist) | 1.32 | ~-11 dB |
| 17640 Hz | 2 | -15 dB |
| 22050 Hz (old Nyquist) | 2.32 | ~-17 dB |

**Problem**: Content between 11-22 kHz folds back into 0-11 kHz with only 11-17 dB attenuation. For typical anti-aliasing, 40-60 dB is preferred.

### Impact on Guitar Signals

Guitar fundamentals and primary harmonics are below 5 kHz, but:

- **Pick attack transients** have broadband energy extending to 15+ kHz
- **String buzz and fret noise** contain high-frequency content
- **Distortion/overdrive** generates dense harmonics

These high-frequency components will alias into lower bands, potentially causing:
- Spurious energy in mid-frequency cochlear channels
- False transient detections
- Envelope modulation artifacts

### CARFAC-Specific Considerations

The CARFAC v2 paper (Lyon, 2024) notes:

> "CARFAC does better with a higher over-sampling ratio... Our current recommended sample rate is 48 kHz for modeling human hearing."

CARFAC's nonlinearities (IHC, AGC) generate intermodulation products. At lower sample rates, these products can alias back into the passband. However, we're already accepting this tradeoff by running at 22050 Hz.

---

## Alternative Approaches

### Option 1: Second-Order Butterworth (Biquad)

Replace one-pole with a biquad lowpass at ~8 kHz cutoff.

**Characteristics**:
- Roll-off: -12 dB/octave
- Attenuation at 11 kHz: ~16-20 dB
- Per-sample cost: 5 multiply-adds (vs 2 for one-pole)

**Pros**: Minimal code change, 2x better rejection
**Cons**: Still modest attenuation; nonlinear phase near cutoff

```cpp
// Biquad state
float x1, x2, y1, y2;
float b0, b1, b2, a1, a2;  // Butterworth coefficients

// Per-sample
float y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2;
x2 = x1; x1 = x;
y2 = y1; y1 = y;
```

### Option 2: Cascaded Biquads (4th-order)

Two biquad sections in series.

**Characteristics**:
- Roll-off: -24 dB/octave
- Attenuation at 11 kHz: ~32-40 dB
- Per-sample cost: 10 multiply-adds

**Pros**: Good stopband rejection
**Cons**: Higher CPU cost; phase distortion accumulates

### Option 3: Polyphase Halfband FIR (Recommended)

A **polyphase FIR** exploits the decimation structure: for M-fold decimation, only 1/M of the filter outputs need to be computed.

For 2x decimation with a 14-tap halfband FIR:
- Only odd-phase samples computed (7 taps per output)
- Linear phase (symmetric impulse response)
- Stopband attenuation: ~40-50 dB (depending on design)

**Characteristics**:
- Per-output cost: 7 multiply-adds
- Amortized per-input cost: 3.5 multiply-adds
- Linear phase, no group delay variation

**Pros**: Best attenuation/cost ratio; linear phase; no feedback
**Cons**: Requires delay line (7 samples); slightly more complex implementation

```cpp
// Polyphase halfband FIR for 2x decimation
// Coefficients designed for 0.4*Nyquist cutoff, ~45 dB stopband
static const float h[7] = {
    0.0072f, -0.0315f, 0.0918f, 0.4325f,  // h[0], h[2], h[4], h[6]
    0.4325f,  0.0918f, -0.0315f           // h[8], h[10], h[12] (symmetric)
};
// Note: h[1], h[3], etc. are zero for halfband

float delay[14];  // Delay line
int phase = 0;

// Per input sample
void push(float x) {
    // Shift delay line and insert new sample
    for (int i = 13; i > 0; --i) delay[i] = delay[i-1];
    delay[0] = x;
    phase ^= 1;
}

// Call every 2 input samples (when phase == 0)
float decimate() {
    float y = 0.0f;
    for (int i = 0; i < 7; ++i) {
        y += h[i] * (delay[2*i] + delay[14 - 2*i]);  // Exploit symmetry
    }
    return y;
}
```

### Option 4: CIC Filter (Cascaded Integrator-Comb)

Efficient for large decimation factors (8x, 16x), but for 2x decimation the frequency response has poor stopband characteristics without compensation filtering.

**Not recommended** for 2x decimation.

---

## Comparison Summary

| Approach | Roll-off | Atten @ 11 kHz | Ops/Input | Phase | Complexity |
|----------|----------|----------------|-----------|-------|------------|
| One-pole (current) | -6 dB/oct | ~11 dB | 2 | Nonlinear | Trivial |
| Biquad | -12 dB/oct | ~18 dB | 5 | Nonlinear | Low |
| 2x Biquad | -24 dB/oct | ~36 dB | 10 | Nonlinear | Low |
| Polyphase FIR | Sharp | ~45 dB | 3.5* | Linear | Medium |

*Amortized: 7 ops every 2 samples

---

## Recommendation

**Use polyphase halfband FIR** for production:

1. **Best cost/quality ratio**: 3.5 ops/sample amortized vs 2 for one-pole, but 45 dB vs 11 dB rejection
2. **Linear phase**: No group delay variation across frequency; important for transient detection
3. **No feedback**: Simpler to debug, no stability concerns
4. **Standard technique**: Well-understood, easy to find reference implementations

**For quick improvement** (minimal code change): Replace one-pole with a biquad Butterworth at 8 kHz. This doubles attenuation with modest CPU increase.

---

## Implementation Notes

### Coefficient Design

Use MATLAB/Octave or scipy to design coefficients:

```python
from scipy.signal import firwin, remez

# Halfband FIR for 2x decimation
# Passband: 0 to 0.4*Nyquist, Stopband: 0.6*Nyquist to Nyquist
numtaps = 15  # Odd for halfband
h = firwin(numtaps, 0.5, window='hamming')  # Simple approach

# Or use remez for equiripple design with explicit stopband
h = remez(numtaps, [0, 0.4, 0.6, 1.0], [1, 0], Hz=2)
```

### NEON Optimization

The polyphase FIR inner loop is ideal for NEON vectorization:

```cpp
// Process 4 symmetric pairs at once
float32x4_t sum = vdupq_n_f32(0.0f);
for (int i = 0; i < 4; ++i) {
    float32x4_t coef = vld1q_f32(&h[i*4]);
    float32x4_t d0 = vld1q_f32(&delay[i*4]);
    float32x4_t d1 = vld1q_f32(&delay[14 - i*4 - 3]);
    sum = vmlaq_f32(sum, coef, vaddq_f32(d0, d1));
}
// Horizontal sum of 'sum' vector
```

### Memory Layout

Delay line should be aligned to 16 bytes for NEON. Consider circular buffer with mask instead of shifting:

```cpp
alignas(16) float delay[16];  // Power of 2 for masking
int write_idx = 0;

void push(float x) {
    delay[write_idx] = x;
    write_idx = (write_idx + 1) & 15;
}
```

---

## References

- Lyon, R. F. (2024). *The CARFAC v2 Cochlear Model*. arXiv:2404.17490
- Crochiere & Rabiner. *Multirate Digital Signal Processing*. Prentice-Hall.
- Vaidyanathan, P. P. *Multirate Systems and Filter Banks*. Prentice-Hall.
- Smith, J. O. *Introduction to Digital Filters*. https://ccrma.stanford.edu/~jos/filters/

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-26 | Claude/Jay | Initial analysis; documented current one-pole approach and alternatives |
