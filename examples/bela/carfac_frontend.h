// CARFAC front-end wrapper for Bela
//
// This is a thin adapter around a CARFAC (cochlear) implementation that:
// - Initializes CAR, IHC, and AGC subsystems for a mono input
// - Steps the processor per sample (CAR -> IHC -> AGC, with loop closure when required)
// - Publishes per-band envelopes (fast/slow) and a transient metric at a hop boundary
//
// Integration notes:
// - By default this file builds a stub (no real CARFAC) so the project compiles.
//   To use the real library, define HAVE_CARFAC and include the appropriate headers
//   in the implementation (see carfac_frontend.cpp), then link the CARFAC sources.
// - Use ENABLE_CARFAC_FRONTEND in settings.json to switch render.cpp to this frontend.

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

#if defined(HAVE_CARFAC)
#include "carfac/upstream/cpp/carfac.h"
#endif

class CarfacFrontend {
public:
    struct Params {
        int sample_rate = 44100;
        int num_channels = 16; // number of cochlear channels (bands)
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
    int fs_ = 44100;
    int bands_ = 0;
    int hop_ = 256;
    bool agc_ = true;

    // Internal per-band states (for the stub and envelope publish)
    std::vector<float> band_sig_;   // latest per-band sample (from CAR/IHC)
    std::vector<float> band_gain_;  // optional per-band gain (AGC)
    std::vector<AR> fast_, slow_;

#ifdef HAVE_CARFAC
    std::unique_ptr<CARFAC> carfac_;
    std::vector<float> buffer_;
#else
    std::vector<float> buffer_;
#endif
};
