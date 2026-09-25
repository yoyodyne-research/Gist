#include "latent_layer.h"
#include "latent_input.h"

#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>
#include <algorithm>

// Simple JSON parsing (no external dependencies)
// This is a minimal parser for the specific model format we export

namespace {

// Skip whitespace
const char* skipWs(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
        ++p;
    return p;
}

// Parse a number (float)
const char* parseNumber(const char* p, float& out) {
    p = skipWs(p);
    char* end;
    out = std::strtof(p, &end);
    return end;
}

// Parse an array of floats
const char* parseFloatArray(const char* p, float* out, int n) {
    p = skipWs(p);
    if (*p != '[') return nullptr;
    ++p;

    for (int i = 0; i < n; ++i) {
        p = parseNumber(p, out[i]);
        if (!p) return nullptr;
        p = skipWs(p);
        if (i < n - 1) {
            if (*p != ',') return nullptr;
            ++p;
        }
    }

    p = skipWs(p);
    if (*p != ']') return nullptr;
    return ++p;
}

// Parse a 2D array of floats (row-major)
const char* parseFloatArray2D(const char* p, float* out, int rows, int cols) {
    p = skipWs(p);
    if (*p != '[') return nullptr;
    ++p;

    for (int r = 0; r < rows; ++r) {
        p = parseFloatArray(p, out + r * cols, cols);
        if (!p) return nullptr;
        p = skipWs(p);
        if (r < rows - 1) {
            if (*p != ',') return nullptr;
            ++p;
        }
    }

    p = skipWs(p);
    if (*p != ']') return nullptr;
    return ++p;
}

// Find a key in JSON object (very simple, assumes no nested strings with colons)
const char* findKey(const char* json, const char* key) {
    std::string needle = "\"";
    needle += key;
    needle += "\"";

    const char* p = std::strstr(json, needle.c_str());
    if (!p) return nullptr;

    p += needle.length();
    p = skipWs(p);
    if (*p != ':') return nullptr;
    return skipWs(p + 1);
}

}  // namespace

bool LatentLayer::loadParams(const char* json_path) {
    // Read entire file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::fprintf(stderr, "LatentLayer: Failed to open %s\n", json_path);
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    const char* json = content.c_str();

    // Parse PCA section
    const char* pca_section = findKey(json, "pca");
    if (!pca_section) {
        std::fprintf(stderr, "LatentLayer: Missing 'pca' section\n");
        return false;
    }

    const char* p;

    // PCA mean
    p = findKey(pca_section, "mean");
    if (!p || !parseFloatArray(p, mean_.data(), kInputDims)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse pca.mean\n");
        return false;
    }

    // PCA std
    p = findKey(pca_section, "std");
    if (!p || !parseFloatArray(p, std_.data(), kInputDims)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse pca.std\n");
        return false;
    }

    // PCA components [k, d]
    p = findKey(pca_section, "components");
    if (!p || !parseFloatArray2D(p, pca_components_.data(), kPcaDims, kInputDims)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse pca.components\n");
        return false;
    }

    // Explained variance (optional)
    p = findKey(pca_section, "explained_variance_ratio");
    if (p) {
        parseFloatArray(p, explained_variance_.data(), kPcaDims);
    } else {
        explained_variance_.fill(0.0f);
    }

    // Parse GMM section
    const char* gmm_section = findKey(json, "gmm");
    if (!gmm_section) {
        std::fprintf(stderr, "LatentLayer: Missing 'gmm' section\n");
        return false;
    }

    // GMM means [K, k]
    p = findKey(gmm_section, "means");
    if (!p || !parseFloatArray2D(p, gmm_means_.data(), kGmmComponents, kPcaDims)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse gmm.means\n");
        return false;
    }

    // GMM covariances [K, k]
    p = findKey(gmm_section, "covariances");
    if (!p || !parseFloatArray2D(p, gmm_covs_.data(), kGmmComponents, kPcaDims)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse gmm.covariances\n");
        return false;
    }

    // GMM weights [K]
    p = findKey(gmm_section, "weights");
    if (!p || !parseFloatArray(p, gmm_weights_.data(), kGmmComponents)) {
        std::fprintf(stderr, "LatentLayer: Failed to parse gmm.weights\n");
        return false;
    }

    // Precompute log weights and normalization constants
    constexpr float kLogTwoPi = 1.8378770664093454835606594728112f;  // log(2*pi)
    for (int k = 0; k < kGmmComponents; ++k) {
        gmm_log_weights_[k] = std::log(gmm_weights_[k] + 1e-10f);

        // Log normalization: -0.5 * (k * log(2pi) + sum(log(cov)))
        float sum_log_cov = 0.0f;
        for (int j = 0; j < kPcaDims; ++j) {
            sum_log_cov += std::log(gmm_covs_[k * kPcaDims + j] + 1e-10f);
        }
        gmm_log_norm_[k] = -0.5f * (kPcaDims * kLogTwoPi + sum_log_cov);
    }

    // Parse config for smoothing
    const char* config_section = findKey(json, "config");
    if (config_section) {
        p = findKey(config_section, "smoothing_ms");
        if (p) {
            float ms;
            parseNumber(p, ms);
            setSmoothingMs(ms);
        }
    }

    // Initialize smoothing state
    r_smooth_.fill(1.0f / kGmmComponents);

    loaded_ = true;
    std::fprintf(stderr, "LatentLayer: Loaded model from %s\n", json_path);
    return true;
}

void LatentLayer::setSmoothingMs(float ms, float control_rate_hz) {
    // Exponential smoothing: y[n] = a*y[n-1] + (1-a)*x[n]
    // Time constant tau = -1 / (fs * log(a))
    // So a = exp(-1 / (tau * fs)) where tau = ms / 1000
    float tau_s = ms / 1000.0f;
    smooth_coef_ = std::exp(-1.0f / (tau_s * control_rate_hz));
}

float LatentLayer::logGaussianDiag(const std::array<float, kPcaDims>& z, int k) const {
    // Log probability under diagonal Gaussian:
    // log N(z | mu, diag(cov)) = -0.5 * (k*log(2pi) + sum(log(cov)) + sum((z-mu)^2/cov))
    // The first two terms are precomputed in gmm_log_norm_

    float maha = 0.0f;  // Mahalanobis distance squared
    for (int j = 0; j < kPcaDims; ++j) {
        float diff = z[j] - gmm_means_[k * kPcaDims + j];
        maha += diff * diff / (gmm_covs_[k * kPcaDims + j] + 1e-10f);
    }

    return gmm_log_norm_[k] - 0.5f * maha;
}

float LatentLayer::computeEntropy(const std::array<float, kGmmComponents>& r) {
    float h = 0.0f;
    for (int k = 0; k < kGmmComponents; ++k) {
        if (r[k] > 1e-10f) {
            h -= r[k] * std::log(r[k]);
        }
    }
    return h;
}

void LatentLayer::process(const LatentInput& input, LatentOutput& out) {
    if (!loaded_) {
        // Zero output if not loaded
        out.z.fill(0.0f);
        out.r.fill(1.0f / kGmmComponents);
        out.mode_id = 0;
        out.mode_strength = 1.0f / kGmmComponents;
        out.entropy = std::log(static_cast<float>(kGmmComponents));
        out.log_likelihood = 0.0f;
        return;
    }

    // 1. Flatten input to vector
    float x[kInputDims];
    input.toVector(x);

    // 2. Standardize: x_norm = (x - mean) / std
    float x_norm[kInputDims];
    for (int i = 0; i < kInputDims; ++i) {
        x_norm[i] = (x[i] - mean_[i]) / (std_[i] + 1e-8f);
    }

    // 3. PCA projection: z = W @ x_norm  (W is [k, d], row-major)
    for (int j = 0; j < kPcaDims; ++j) {
        out.z[j] = 0.0f;
        for (int i = 0; i < kInputDims; ++i) {
            out.z[j] += pca_components_[j * kInputDims + i] * x_norm[i];
        }
    }

    // 4. GMM responsibilities: r_k ∝ π_k * N(z | μ_k, Σ_k)
    float log_probs[kGmmComponents];
    float max_log = -1e30f;

    for (int k = 0; k < kGmmComponents; ++k) {
        log_probs[k] = logGaussianDiag(out.z, k) + gmm_log_weights_[k];
        if (log_probs[k] > max_log) {
            max_log = log_probs[k];
        }
    }

    // Log-sum-exp for numerical stability
    float sum = 0.0f;
    for (int k = 0; k < kGmmComponents; ++k) {
        out.r[k] = std::exp(log_probs[k] - max_log);
        sum += out.r[k];
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int k = 0; k < kGmmComponents; ++k) {
        out.r[k] *= inv_sum;
    }

    // 5. Smooth responsibilities
    for (int k = 0; k < kGmmComponents; ++k) {
        r_smooth_[k] = smooth_coef_ * r_smooth_[k] + (1.0f - smooth_coef_) * out.r[k];
    }

    // Use smoothed values for output
    std::copy(r_smooth_.begin(), r_smooth_.end(), out.r.begin());

    // 6. Derived signals
    out.mode_id = 0;
    out.mode_strength = out.r[0];
    for (int k = 1; k < kGmmComponents; ++k) {
        if (out.r[k] > out.mode_strength) {
            out.mode_id = k;
            out.mode_strength = out.r[k];
        }
    }

    out.entropy = computeEntropy(out.r);

    // Log-likelihood: log(sum(exp(log_probs)))
    out.log_likelihood = max_log + std::log(sum + 1e-10f);
}
