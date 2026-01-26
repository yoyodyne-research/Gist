// CARFAC front-end wrapper for Bela
//
// This is a thin adapter around the CARFAC (cochlear) library that:
// - Initializes CAR, IHC, and AGC subsystems for a mono input
// - Processes samples inline via per-sample API (spreads CPU load evenly)
// - Publishes per-band envelopes (fast/slow) and a transient metric
//
// Requires HAVE_CARFAC to be defined and CARFAC sources to be linked.

#pragma once

#ifndef HAVE_CARFAC
#error "HAVE_CARFAC must be defined to use CarfacFrontend"
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

#include "carfac/upstream/cpp/carfac.h"

class CarfacFrontend {
public:
    struct Params {
        int sample_rate = 44100;       // Input sample rate (Bela audio rate)
        int carfac_rate = 0;           // CARFAC processing rate (0 = same as sample_rate)
        bool enable_agc = true;
        int publish_hop = 256; // frames per publish
        // Envelope timing (seconds); used by the wrapper for fast/slow AR
        float fast_attack = 0.005f;
        float fast_release = 0.040f;
        float slow_attack = 0.020f;
        float slow_release = 0.200f;
    };

    CarfacFrontend() = default;
    ~CarfacFrontend();

    bool init(const Params& p);

    // Per-sample process; x is mono sample
    void processSample(float x);

    // Called at hop boundary to copy out latest features
    void publish(std::vector<float>& env_fast,
                 std::vector<float>& env_slow,
                 std::vector<float>& transient,
                 std::vector<float>* opt_gain = nullptr);

    int numBands() const { return bands_; }

    // Get current fast envelope value for a band (updated per-sample)
    float getEnvFast(int band) const { return fast_[band].env; }

private:
    // Wrapper-side envelope followers
    struct AR {
        float aAtk=0, aRel=0, env=0;
        void set(float fs, float atk_s, float rel_s) {
            aAtk = std::exp(-1.0f / (std::max(atk_s, 1e-4f) * fs));
            aRel = std::exp(-1.0f / (std::max(rel_s, 1e-4f) * fs));
        }
        inline float step(float xabs){
            float t = xabs;
            env = (t > env) ? (aAtk*env + (1.f-aAtk)*t)
                            : (aRel*env + (1.f-aRel)*t);
            return env;
        }
    };

    // Config
    int fs_ = 44100;         // Input sample rate
    int carfac_fs_ = 44100;  // CARFAC internal rate
    int decim_ = 1;          // Decimation factor (fs_ / carfac_fs_)
    int decim_cnt_ = 0;      // Decimation counter
    float aa_state_ = 0.0f;  // Anti-aliasing filter state
    float aa_coeff_ = 0.0f;  // Anti-aliasing filter coefficient
    int bands_ = 0;
    int hop_ = 256;
    bool agc_ = true;

    // Internal per-band states
    std::vector<float> band_sig_;
    std::vector<float> band_gain_;
    std::vector<AR> fast_, slow_;

    std::unique_ptr<CARFAC> carfac_;
    Ear* ear_ = nullptr;  // Cached pointer to mutable ear (avoids repeated lookups)
};
