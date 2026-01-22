// Minimal multiband envelope filterbank ("Flenser") for Bela
// TPT SVF engine (topology-preserving transform) for stable HF behavior
// - Bands defined by split edges: [0,f0], (f0,f1], ..., (f_{M-1}, Nyq]
// - For inner bands, use bandpass with fc=sqrt(f1*f2), Q=fc/(f2-f1)
// - Optional cascade (order 4) per band via FLENSER_SVF_ORDER (2 or 4)
// - Dual envelope followers per band: fast and slow; transient = max(0, fast - slow)
// - Designed to run per-sample in the audio thread

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

// TPT SVF (VA) per Zavalishin; stable near Nyquist
struct TPTSVF {
    float g=0.f, k=0.f; // g = tan(pi*fc/fs), k = 1/Q
    float ic1=0.f, ic2=0.f; // integrator states (equalized)
    void set(float fs, float fc, float Q) {
        fc = std::max(10.0f, fc);
        Q  = std::max(0.1f, Q);
        const float w = static_cast<float>(M_PI) * fc / fs;
        g = std::tan(w);
        k = 1.0f / Q;
        ic1 = ic2 = 0.0f;
    }
    inline void process(float x, float& lp, float& bp, float& hp) {
        const float a = 1.0f / (1.0f + g*(g + k));
        const float v1 = a * (ic1 + g*(x - ic2));
        const float v2 = ic2 + g*v1;
        hp = x - k*v1 - v2;
        bp = v1;
        lp = v2;
        ic1 = 2.0f*v1 - ic1;
        ic2 = 2.0f*v2 - ic2;
    }
};

struct ARFollower {
    float env = 0.0f;
    float aAtk = 0.0f;
    float aRel = 0.0f;
    void set(float fs, float atk_s, float rel_s) {
        aAtk = std::exp(-1.0f / std::max(atk_s, 1e-4f) / fs);
        aRel = std::exp(-1.0f / std::max(rel_s, 1e-4f) / fs);
    }
    inline float process(float xabs) {
        float target = xabs;
        if (target > env) env = aAtk*env + (1.0f - aAtk)*target;
        else              env = aRel*env + (1.0f - aRel)*target;
        return env;
    }
};

class Flenser {
public:
    // splitFreqs: ascending array of crossover frequencies, size M → bands = M+1
    // Envelope timing heuristic (per‑band):
    // - We derive a representative band frequency fc (Hz):
    //   low  band → fc ≈ 0.5 * f0
    //   high band → fc ≈ sqrt(f_{M-1} * fs/2)
    //   inner band → fc = sqrt(f1 * f2) (geometric mean)
    // - Base timing scales with 1/fc to keep low bands slightly slower and
    //   high bands snappy, with sensible clamps:
    //   fast attack  ≈ clamp(fastAtk * (fRef/fc),  2ms .. 10ms)
    //   fast release ≈ clamp(fastRel * (fRef/fc), 20ms .. 80ms)
    //   slow attack  ≈ clamp(slowAtk * (fRef/fc), 15ms .. 50ms)
    //   slow release ≈ clamp(slowRel * (fRef/fc),150ms .. 400ms)
    // - The provided fast/slow times are interpreted as mid‑band (fRef) base
    //   values; scaling factors (…Scale*) allow global adjustment of attack
    //   and release heuristics without changing band shapes.
    // - Intent: preserve transient articulation across bands while avoiding
    //   excessive pumping in low bands and sluggishness in high bands.
    void configure(float fs, const std::vector<float>& splitFreqs,
                   float fastAtk=0.005f, float fastRel=0.040f,
                   float slowAtk=0.020f, float slowRel=0.200f,
                   float fastAtkScale=1.0f, float fastRelScale=1.0f,
                   float slowAtkScale=1.0f, float slowRelScale=1.0f)
    {
        sampleRate = fs;
        splits = splitFreqs;
        const int M = (int)splits.size();
        const int N = M + 1;
        bandStages.resize(N);
        bandFast.resize(N); bandSlow.resize(N);
        bandVal.assign(N, 0.0f);
        envFast.assign(N, 0.0f);
        envSlow.assign(N, 0.0f);
        transient.assign(N, 0.0f);

        // Configure per-band SVFs
        for (int b = 0; b < N; ++b) {
            if (b == 0) {
                // Low band: LP at f0
                const float fc = splits[0];
                float dummyQ = 0.7071f;
                bandStages[b].type = BandType::Low;
                bandStages[b].stage1.set(sampleRate, fc, dummyQ);
#if !defined(FLENSER_SVF_ORDER) || (FLENSER_SVF_ORDER==4)
                bandStages[b].stage2.set(sampleRate, fc, dummyQ);
                bandStages[b].cascade = true;
#else
                bandStages[b].cascade = false;
#endif
            } else if (b == N-1) {
                // High band: HP at f_{M-1}
                const float fc = splits[M-1];
                float dummyQ = 0.7071f;
                bandStages[b].type = BandType::High;
                bandStages[b].stage1.set(sampleRate, fc, dummyQ);
#if !defined(FLENSER_SVF_ORDER) || (FLENSER_SVF_ORDER==4)
                bandStages[b].stage2.set(sampleRate, fc, dummyQ);
                bandStages[b].cascade = true;
#else
                bandStages[b].cascade = false;
#endif
            } else {
                // Inner band: BP with fc=sqrt(f1*f2), Q=fc/(f2-f1)
                const float f1 = splits[b-1];
                const float f2 = splits[b];
                const float fc = std::sqrt(f1 * f2);
                float Q = fc / std::max(10.0f, (f2 - f1));
                Q = std::max(0.2f, std::min(Q, 10.0f));
                bandStages[b].type = BandType::Band;
                bandStages[b].stage1.set(sampleRate, fc, Q);
#if !defined(FLENSER_SVF_ORDER) || (FLENSER_SVF_ORDER==4)
                bandStages[b].stage2.set(sampleRate, fc, Q);
                bandStages[b].cascade = true;
#else
                bandStages[b].cascade = false;
#endif
            }
        }

        // Heuristic per‑band envelope timing
        auto clampf = [](float v, float lo, float hi){ return std::max(lo, std::min(v, hi)); };
        const float fRef = 1000.0f; // reference freq for base timings
        const float nyq  = sampleRate * 0.5f;
        for (int b = 0; b < N; ++b) {
            float fc = 1000.0f;
            if (bandStages[b].type == BandType::Low)
                fc = std::max(10.0f, 0.5f * splits[0]);
            else if (bandStages[b].type == BandType::High)
                fc = std::sqrt(std::max(10.0f, splits[M-1]) * nyq);
            else {
                const float f1 = splits[b-1];
                const float f2 = splits[b];
                fc = std::sqrt(std::max(10.0f, f1) * std::max(10.0f, f2));
            }

            const float scale = fRef / std::max(10.0f, fc);
            // Base times scaled and clamped per heuristic; then apply global scales
            const float fa = clampf(fastAtk * scale, 0.002f, 0.010f) * fastAtkScale;
            const float fr = clampf(fastRel * scale, 0.020f, 0.080f) * fastRelScale;
            const float sa = clampf(slowAtk * scale, 0.015f, 0.050f) * slowAtkScale;
            const float sr = clampf(slowRel * scale, 0.150f, 0.400f) * slowRelScale;
            bandFast[b].set(sampleRate, fa, fr);
            bandSlow[b].set(sampleRate, sa, sr);
        }
    }

    inline void processSample(float x) {
        const int N = (int)bandStages.size();
        for (int b = 0; b < N; ++b) {
            float lp1=0, bp1=0, hp1=0;
            float y=0;
            bandStages[b].stage1.process(x, lp1, bp1, hp1);
            switch (bandStages[b].type) {
                case BandType::Low:
                    y = lp1;
                    break;
                case BandType::High:
                    y = hp1;
                    break;
                case BandType::Band:
                default:
                    y = bp1;
                    break;
            }
            if (bandStages[b].cascade) {
                float lp2=0, bp2=0, hp2=0;
                bandStages[b].stage2.process(y, lp2, bp2, hp2);
                switch (bandStages[b].type) {
                    case BandType::Low:   y = lp2; break;
                    case BandType::High:  y = hp2; break;
                    case BandType::Band:  y = bp2; break;
                }
            }
            bandVal[b] = y;
        }

        // envelopes
        for (int b = 0; b < N; ++b) {
            float a = std::fabs(bandVal[b]);
            envFast[b] = bandFast[b].process(a);
            envSlow[b] = bandSlow[b].process(a);
            float t = envFast[b] - envSlow[b];
            transient[b] = (t > 0.f ? t : 0.f);
        }
    }

    const std::vector<float>& bands()     const { return bandVal; }
    const std::vector<float>& env_fast()  const { return envFast; }
    const std::vector<float>& env_slow()  const { return envSlow; }
    const std::vector<float>& transients() const { return transient; }

private:
    float sampleRate = 44100.0f;
    std::vector<float> splits;
    enum class BandType { Low, Band, High };
    struct Band {
        BandType type = BandType::Band;
        TPTSVF stage1;
        TPTSVF stage2;
        bool cascade = false;
    };
    std::vector<Band> bandStages;
    std::vector<float> bandVal, envFast, envSlow, transient;
    std::vector<ARFollower> bandFast, bandSlow;
};
