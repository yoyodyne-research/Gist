// LatentInput: Feature vector structure for latent layer inference
//
// This defines the 150-dimensional input format expected by LatentLayer::process()
// Shared between CarfacFrontend (producer) and LatentLayer (consumer)

#pragma once

#include <array>

struct LatentInput {
    static constexpr int kNumBands = 71;
    static constexpr int kSummaryDims = 8;
    static constexpr int kDims = kNumBands * 2 + kSummaryDims;  // 150

    // Envelope features (71 bands)
    std::array<float, kNumBands> env_fast;   // Fast envelope (attack-smoothed)
    std::array<float, kNumBands> delta;      // Envelope derivative

    // Summary statistics (8 dims)
    float total_energy;       // Sum of env_fast
    float spectral_centroid;  // Weighted mean band index
    float spectral_spread;    // Weighted std of band index
    float peak_band;          // argmax(env_fast)
    float attack_breadth;     // How many bands are rising
    std::array<float, 3> band_ratios;  // Low/mid/high energy ratios

    // Convert to flat vector for PCA projection
    void toVector(float* out) const {
        for (int i = 0; i < kNumBands; ++i) out[i] = env_fast[i];
        for (int i = 0; i < kNumBands; ++i) out[kNumBands + i] = delta[i];
        out[142] = total_energy;
        out[143] = spectral_centroid;
        out[144] = spectral_spread;
        out[145] = peak_band;
        out[146] = attack_breadth;
        out[147] = band_ratios[0];
        out[148] = band_ratios[1];
        out[149] = band_ratios[2];
    }
};
