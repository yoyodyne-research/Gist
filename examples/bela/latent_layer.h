// Latent Layer: PCA + GMM inference for expressive control signals
//
// This class implements the runtime inference for the latent expression layer:
// - Input: 150-dim feature vector from CARFAC envelopes + summary statistics
// - Output: 8 PCA latents (z[0..7]) + 14 GMM responsibilities (r[0..13]) + meta signals
//
// All parameters are loaded from a JSON file trained offline in Python.
// Processing is deterministic, real-time safe, and uses only std:: (no BLAS/etc).

#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "latent_input.h"

// Output structure from latent layer processing
struct LatentOutput {
    static constexpr int kPcaDims = 8;
    static constexpr int kGmmComponents = 14;

    std::array<float, kPcaDims> z;           // PCA latents (continuous, signed)
    std::array<float, kGmmComponents> r;     // GMM responsibilities [0,1], sum to 1
    int mode_id;                             // argmax(r)
    float mode_strength;                     // max(r)
    float entropy;                           // -sum(r_i * log(r_i))
    float log_likelihood;                    // log p(z) under GMM
};

class LatentLayer {
public:
    static constexpr int kInputDims = 150;
    static constexpr int kPcaDims = 8;
    static constexpr int kGmmComponents = 14;

    LatentLayer() = default;

    // Load model parameters from JSON file
    // Returns true on success, false on parse error or dimension mismatch
    bool loadParams(const char* json_path);

    // Set smoothing time constant (default 30ms, applied to responsibilities)
    void setSmoothingMs(float ms, float control_rate_hz = 230.0f);

    // Main inference (call at control rate ~230 Hz)
    // Input: feature vector from CarfacFrontend::publishFeatures()
    // Output: latent signals for mapping layer
    void process(const LatentInput& input, LatentOutput& output);

    // Check if parameters have been loaded
    bool isLoaded() const { return loaded_; }

    // Get explained variance ratios from PCA (for diagnostics)
    const std::array<float, kPcaDims>& getExplainedVariance() const {
        return explained_variance_;
    }

private:
    // Compute log probability of z under diagonal Gaussian k
    float logGaussianDiag(const std::array<float, kPcaDims>& z, int k) const;

    // Compute entropy of probability distribution
    static float computeEntropy(const std::array<float, kGmmComponents>& r);

    bool loaded_ = false;

    // PCA parameters (standardization + projection)
    std::array<float, kInputDims> mean_;    // Feature means
    std::array<float, kInputDims> std_;     // Feature stds
    std::array<float, kPcaDims * kInputDims> pca_components_;  // Row-major [k, d]

    // GMM parameters (diagonal covariance)
    std::array<float, kGmmComponents * kPcaDims> gmm_means_;   // [K, k]
    std::array<float, kGmmComponents * kPcaDims> gmm_covs_;    // [K, k] variances
    std::array<float, kGmmComponents> gmm_weights_;            // [K] mixture weights
    std::array<float, kGmmComponents> gmm_log_weights_;        // log(weights) cached

    // Pre-computed terms for Gaussian evaluation
    std::array<float, kGmmComponents> gmm_log_norm_;           // -0.5*k*log(2pi) - 0.5*sum(log(cov))

    // Smoothing state
    std::array<float, kGmmComponents> r_smooth_;
    float smooth_coef_ = 0.0f;  // Exponential smoothing coefficient

    // Diagnostics
    std::array<float, kPcaDims> explained_variance_;
};
