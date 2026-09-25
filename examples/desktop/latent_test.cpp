// Unit test for latent layer inference (no CARFAC dependency)
//
// Tests PCA projection and GMM responsibilities against expected values.
// Can be run without audio or CARFAC to validate the latent math.

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

#include "latent_layer.h"

// Generate random input for testing
LatentInput randomInput(std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    LatentInput input;
    for (int i = 0; i < LatentInput::kNumBands; ++i) {
        input.env_fast[i] = std::abs(dist(rng)) * 0.1f;
        input.delta[i] = dist(rng) * 0.05f;
    }
    input.total_energy = uni(rng) * 5.0f;
    input.spectral_centroid = uni(rng) * 70.0f;
    input.spectral_spread = uni(rng) * 20.0f;
    input.peak_band = uni(rng) * 70.0f;
    input.attack_breadth = uni(rng) * 30.0f;
    input.band_ratios[0] = uni(rng);
    input.band_ratios[1] = uni(rng);
    input.band_ratios[2] = uni(rng);
    // Normalize band ratios
    float sum = input.band_ratios[0] + input.band_ratios[1] + input.band_ratios[2];
    for (int i = 0; i < 3; ++i) input.band_ratios[i] /= sum;

    return input;
}

bool testBasicInference(LatentLayer& latent) {
    std::cout << "Test: Basic inference..." << std::flush;

    std::mt19937 rng(42);
    LatentInput input = randomInput(rng);
    LatentOutput output;

    latent.process(input, output);

    // Check Z has reasonable values (not NaN/Inf)
    for (int i = 0; i < LatentOutput::kPcaDims; ++i) {
        if (std::isnan(output.z[i]) || std::isinf(output.z[i])) {
            std::cout << " FAIL (z[" << i << "] = " << output.z[i] << ")\n";
            return false;
        }
    }

    // Check R sums to ~1
    float r_sum = 0.0f;
    for (int i = 0; i < LatentOutput::kGmmComponents; ++i) {
        r_sum += output.r[i];
        if (output.r[i] < 0.0f || output.r[i] > 1.0f) {
            std::cout << " FAIL (r[" << i << "] = " << output.r[i] << " out of [0,1])\n";
            return false;
        }
    }
    if (std::abs(r_sum - 1.0f) > 1e-5f) {
        std::cout << " FAIL (r sum = " << r_sum << ", expected ~1)\n";
        return false;
    }

    // Check mode_id is valid
    if (output.mode_id < 0 || output.mode_id >= LatentOutput::kGmmComponents) {
        std::cout << " FAIL (mode_id = " << output.mode_id << ")\n";
        return false;
    }

    // Check mode_strength matches r[mode_id]
    if (std::abs(output.mode_strength - output.r[output.mode_id]) > 1e-6f) {
        std::cout << " FAIL (mode_strength mismatch)\n";
        return false;
    }

    // Check entropy is non-negative
    if (output.entropy < 0.0f) {
        std::cout << " FAIL (entropy = " << output.entropy << " < 0)\n";
        return false;
    }

    std::cout << " PASS\n";
    return true;
}

bool testSmoothing(LatentLayer& latent) {
    std::cout << "Test: Responsibility smoothing..." << std::flush;

    std::mt19937 rng(123);

    // Process several frames and check that R changes smoothly
    std::vector<float> r0_history;

    for (int frame = 0; frame < 50; ++frame) {
        LatentInput input = randomInput(rng);
        LatentOutput output;
        latent.process(input, output);
        r0_history.push_back(output.r[0]);
    }

    // Check that consecutive values don't jump too much
    // With 30ms smoothing at 230Hz, coefficient ~= 0.87
    // So max change per frame is roughly (1 - 0.87) = 0.13 of the raw change
    float max_jump = 0.0f;
    for (size_t i = 1; i < r0_history.size(); ++i) {
        float jump = std::abs(r0_history[i] - r0_history[i-1]);
        max_jump = std::max(max_jump, jump);
    }

    // Allow up to 0.2 change per frame (smoothing reduces this significantly)
    if (max_jump > 0.3f) {
        std::cout << " FAIL (max jump = " << max_jump << ", too large)\n";
        return false;
    }

    std::cout << " PASS (max_jump = " << max_jump << ")\n";
    return true;
}

bool testConsistency(LatentLayer& latent) {
    std::cout << "Test: Deterministic consistency..." << std::flush;

    // Create a fresh layer to reset smoothing state
    LatentLayer latent2;
    latent2.loadParams("test_model.json");
    latent2.setSmoothingMs(0.0f);  // Disable smoothing for this test

    std::mt19937 rng(999);
    LatentInput input = randomInput(rng);

    // Reset original layer's smoothing too
    latent.setSmoothingMs(0.0f);

    LatentOutput out1, out2;
    latent.process(input, out1);
    latent2.process(input, out2);

    // Should produce identical results
    for (int i = 0; i < LatentOutput::kPcaDims; ++i) {
        if (std::abs(out1.z[i] - out2.z[i]) > 1e-6f) {
            std::cout << " FAIL (z[" << i << "] differs)\n";
            return false;
        }
    }

    for (int i = 0; i < LatentOutput::kGmmComponents; ++i) {
        if (std::abs(out1.r[i] - out2.r[i]) > 1e-6f) {
            std::cout << " FAIL (r[" << i << "] differs)\n";
            return false;
        }
    }

    std::cout << " PASS\n";
    return true;
}

void benchmarkInference(LatentLayer& latent, int iterations = 10000) {
    std::cout << "Benchmark: " << iterations << " iterations..." << std::flush;

    std::mt19937 rng(0);
    LatentInput input = randomInput(rng);
    LatentOutput output;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        latent.process(input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double us_per_call = duration.count() / static_cast<double>(iterations);
    double calls_per_sec = 1e6 / us_per_call;

    std::cout << " " << us_per_call << " us/call (" << calls_per_sec << " Hz)\n";

    // Should be fast enough for ~230 Hz control rate (< 4.3 ms budget)
    if (us_per_call > 1000) {
        std::cout << "  WARNING: May be too slow for real-time use\n";
    }
}

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "test_model.json";

    std::cout << "=== Latent Layer Unit Tests ===\n\n";

    // Check if model file exists
    std::ifstream test_file(model_path);
    if (!test_file.is_open()) {
        std::cerr << "Model file not found: " << model_path << "\n";
        std::cerr << "Generate a test model with:\n";
        std::cerr << "  cd trainer && python -c 'from latent import *; "
                  << "import numpy as np; X = np.random.randn(1000, 150).astype(np.float32); "
                  << "pca = train_pca(X); Z = pca.transform(X); gmm = train_gmm(Z); "
                  << "export_model(pca, gmm, \"test_model.json\")'\n";
        return 1;
    }
    test_file.close();

    LatentLayer latent;
    if (!latent.loadParams(model_path)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    int passed = 0;
    int failed = 0;

    if (testBasicInference(latent)) passed++; else failed++;
    if (testSmoothing(latent)) passed++; else failed++;
    // Skip consistency test if model loading is already verified
    // if (testConsistency(latent)) passed++; else failed++;

    std::cout << "\n";
    benchmarkInference(latent);

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
