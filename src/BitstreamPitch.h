//=======================================================================
/** @file BitstreamPitch.h
 *  @brief Bitstream autocorrelation pitch estimator (q-inspired, C++14)
 */
//=======================================================================

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

class BitstreamPitch
{
public:
    // Configure with sample rate and frequency bounds.
    // window is computed as ~2 * period(lowest) * fs per q.
    BitstreamPitch(int frameSize, int sampleRate,
                   float lowestHz = 60.0f,
                   float highestHz = 1000.0f,
                   float hysteresisLinear = 0.02f)
        : fs(sampleRate)
        , nFrame(frameSize)
        , fLow(lowestHz)
        , fHigh(highestHz)
        , hyst(hysteresisLinear)
    {
        configure(fs);
    }

    void setFrameSize(int frameSize) { nFrame = frameSize; }
    void setSampleRate(int sampleRate) { fs = sampleRate; configure(fs); }
    void setFrequencyRange(float lowestHz, float highestHz)
    {
        fLow = lowestHz; fHigh = highestHz; configure(fs);
    }
    void setHysteresisLinear(float h) { hyst = h; }

    // Feed a frame (processes sample-by-sample internally). Returns latest
    // pitch in Hz if a new estimate is ready, otherwise returns the last value.
    float processFrame(const float* frame, int count)
    {
        for (int i = 0; i < count; ++i)
            feed(frame[i]);
        return lastHz;
    }

    float lastPitchHz() const { return lastHz; }
    float lastPeriodicity() const { return lastPeriodicity_; }

private:
    struct Edge {
        float peak;
        int lead;
        int trail;
        Edge() : peak(0.f), lead(-1), trail(-1) {}
        Edge(float p, int l, int t) : peak(p), lead(l), trail(t) {}
    };

    // State
    int fs = 44100;
    int nFrame = 0;
    float fLow = 60.f;
    float fHigh = 1000.f;
    float hyst = 0.02f; // linear amplitude hysteresis for ZC collector

    // Sliding window and thresholds
    int window = 0;
    int midPoint = 0;
    int minPeriod = 0;
    float weight = 0.f; // 2.0 / window
    float periodDiffThreshold = 0.f; // midPoint * 0.008
    float pulseThresholdScale = 0.6f; // fraction of max peak

    // Zero-crossing collector
    float prev = 0.f;
    bool state = false;
    int frameIdx = 0;
    int edgeCount = 0;
    float peak = 0.f;
    float peakUpdate = 0.f;
    std::vector<Edge> edges; // ring-like, newest at back

    // Bitset for pulses
    std::vector<uint64_t> bits;

    // Outputs
    float lastHz = 0.f;
    float lastPeriodicity_ = 0.f;

    // Helpers
    static inline int popcnt64(uint64_t x)
    {
    #if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcountll(x);
    #else
        int c = 0; while (x) { x &= (x - 1); ++c; } return c;
    #endif
    }

    void configure(int sampleRate)
    {
        fs = sampleRate;
        window = std::max(32, (int)std::round(2.0f * (fs / std::max(fLow, 1.0f))));
        midPoint = window / 2;
        minPeriod = std::max(1, (int)std::round(fs / std::max(fHigh, 1.0f)));
        weight = 2.0f / (float)window;
        periodDiffThreshold = midPoint * 0.008f;
        edges.clear(); edges.reserve(128);
        bits.assign(((window + 63) >> 6), 0);
        resetWindow();
    }

    void resetWindow()
    {
        prev = 0.f; state = false; frameIdx = 0; edgeCount = 0;
        peak = 0.f; peakUpdate = 0.f; edges.clear();
        std::fill(bits.begin(), bits.end(), 0ull);
    }

    void pushEdge(int leading, float firstSample)
    {
        if ((int)edges.size() >= 512) // cap to something sane
            edges.erase(edges.begin());
        edges.push_back(Edge{firstSample, leading, -1});
        ++edgeCount;
    }

    void updatePeak(float s)
    {
        if (!edges.empty())
        {
            Edge& e = edges.back();
            if (s > e.peak) e.peak = s;
        }
        if (s > peakUpdate) peakUpdate = s;
    }

    void feed(float sIn)
    {
        float s = sIn + hyst * 0.5f; // center zero-crossing on actual zero

        // Ensure we reset if ring too full
        if ((int)edges.size() >= 512) resetWindow();

        // rising edge
        if (s > 0.f)
        {
            if (!state)
            {
                pushEdge(frameIdx, s);
                state = true;
            }
            else
            {
                updatePeak(s);
            }
        }
        else if (state && s < hyst)
        {
            // falling edge
            state = false;
            if (!edges.empty()) edges.back().trail = frameIdx;
            if (peak == 0.f) peak = peakUpdate;
        }

        prev = s;

        // Slide window check
        if (++frameIdx >= window && !state)
        {
            frameIdx -= midPoint; // slide by half window
            slideEdges(midPoint);

            if (edgeCount > 1)
            {
                // Ready: build bitset and run ACF
                setBitstream();
                estimatePeriod();
            }
            else
            {
                resetWindow();
            }
        }
    }

    void slideEdges(int n)
    {
        // Shift indices; drop edges with trail < 0
        std::vector<Edge> kept;
        kept.reserve(edges.size());
        for (auto& e : edges)
        {
            e.lead -= n; if (e.trail >= 0) e.trail -= n;
            if (e.trail >= 0 || state) kept.push_back(e);
        }
        edges.swap(kept);
        edgeCount = (int)edges.size();
        peak = peakUpdate; peakUpdate = 0.f;
        std::fill(bits.begin(), bits.end(), 0ull);
    }

    void setBitstream()
    {
        const float thr = std::max(peak, peakUpdate) * pulseThresholdScale;
        std::fill(bits.begin(), bits.end(), 0ull);
        auto setRange = [&](int start, int len){
            if (len <= 0) return;
            int end = std::min(window, start + len);
            int i = std::max(0, start);
            while (i < end)
            {
                int word = i >> 6; int off = i & 63;
                int chunk = std::min(64 - off, end - i);
                uint64_t mask = (chunk == 64) ? ~0ull : ((1ull << chunk) - 1ull);
                bits[word] |= (mask << off);
                i += chunk;
            }
        };
        for (auto const& e : edges)
        {
            if (e.peak >= thr && e.lead >= 0 && e.trail > e.lead)
                setRange(e.lead, e.trail - e.lead);
        }
    }

    int acCountAt(int lag) const
    {
        int arrays = (int)bits.size();
        int wordsHalf = std::max(1, arrays / 2 - 1);
        int index = lag >> 6; int shift = lag & 63;
        int count = 0;
        const uint64_t* p1 = bits.data();
        const uint64_t* p2 = bits.data() + index;
        if (shift == 0)
        {
            for (int i = 0; i < wordsHalf; ++i)
                count += popcnt64(p1[i] ^ p2[i]);
        }
        else
        {
            int shift2 = 64 - shift;
            for (int i = 0; i < wordsHalf; ++i)
            {
                uint64_t v = (p2[i] >> shift) | (p2[i+1] << shift2);
                count += popcnt64(p1[i] ^ v);
            }
        }
        return count;
    }

    void estimatePeriod()
    {
        if (edgeCount < 2) return;

        const float thr = std::max(peak, peakUpdate) * pulseThresholdScale;
        float bestPeriodicity = 0.f;
        int bestPeriod = -1;

        bool first = true;

        for (int i = 0; i < edgeCount - 1; ++i)
        {
            const Edge& a = edges[i]; if (a.peak < thr) continue;
            for (int j = i + 1; j < edgeCount; ++j)
            {
                const Edge& b = edges[j]; if (b.peak < thr) continue;
                int period = b.lead - a.lead;
                if (period <= 0 || period < minPeriod) continue;
                if (period > midPoint) break;

                int count = acCountAt(period);
                if (first && count == 0)
                {
                    int p2 = period / 2; if (p2 > minPeriod && acCountAt(p2) == 0)
                        { first = false; continue; }
                }

                if (period < 32)
                {
                    int start = period;
                    // upwards
                    for (int p = start + 1; p < midPoint; ++p)
                    {
                        int c = acCountAt(p);
                        if (c > count) break; count = c; period = p;
                    }
                    // downwards
                    for (int p = start - 1; p > minPeriod; --p)
                    {
                        int c = acCountAt(p);
                        if (c > count) break; count = c; period = p;
                    }
                }

                float periodicity = 1.0f - (count * weight);
                if (periodicity > bestPeriodicity + 1e-6f &&
                    (bestPeriod == -1 || std::fabs((float)period - (float)bestPeriod) > periodDiffThreshold))
                {
                    bestPeriodicity = periodicity;
                    bestPeriod = period;
                }
                first = false;
                if (count == 0) break; // perfect correlation
            }
        }

        if (bestPeriod > 0)
        {
            lastPeriodicity_ = bestPeriodicity;
            lastHz = fs / (float)bestPeriod;
        }
        else
        {
            lastPeriodicity_ = 0.f;
            // keep lastHz unchanged
        }
    }
};
