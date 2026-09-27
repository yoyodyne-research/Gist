// The superfly filter bench's output chain, native: a Web Audio BiquadFilterNode (spec coefficients, Q in dB
// for lowpass/highpass and linear otherwise), a VCA, a dry/wet mix, a volume and a peak limiter.
//   out = limiter(volume * (mix * vca * biquad(x) + (1 - mix) * x))
// Filter parameters arrive once per control frame and ramp linearly, per sample, over one frame (256
// samples), like the page's linearRampToValueAtTime. Coefficients are recomputed every sample, in double,
// matching superfly.modfilter._biquad. Real-time safe: no allocation after construction.
#pragma once

#include <algorithm>
#include <cmath>

namespace fly {

enum FilterKind { kLowpass = 0, kBandpass = 1, kHighpass = 2, kNotch = 3, kPeak = 4 };

struct FilterParams {
    double cutoff = 1000.0, q = 0.0, gain_db = 0.0, vca = 1.0;
};

class Biquad {
public:
    double process(double x, int kind, double f0, double q, double gain_db, double fs) {
        const double w0 = 2.0 * M_PI * std::min(f0, 0.5 * fs * 0.999) / fs;
        const double cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = (kind == kLowpass || kind == kHighpass) ? sw / (2.0 * std::pow(10.0, q / 20.0))
                                                                      : sw / (2.0 * q);
        double b0, b1, b2, a0, a2;
        switch (kind) {
        case kLowpass:  b0 = (1 - cw) / 2; b1 = 1 - cw;    b2 = (1 - cw) / 2; a0 = 1 + alpha; a2 = 1 - alpha; break;
        case kHighpass: b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; a0 = 1 + alpha; a2 = 1 - alpha; break;
        case kBandpass: b0 = alpha;        b1 = 0.0;       b2 = -alpha;       a0 = 1 + alpha; a2 = 1 - alpha; break;
        case kNotch:    b0 = 1.0;          b1 = -2 * cw;   b2 = 1.0;          a0 = 1 + alpha; a2 = 1 - alpha; break;
        default: {
            const double A = std::pow(10.0, gain_db / 40.0);
            b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A; a0 = 1 + alpha / A; a2 = 1 - alpha / A;
        }
        }
        // normalise first, in the same order as superfly.modfilter._coeffs / _biquad
        const double nb0 = b0 / a0, nb1 = b1 / a0, nb2 = b2 / a0, na1 = -2 * cw / a0, na2 = a2 / a0;
        const double y = nb0 * x + nb1 * x1_ + nb2 * x2_ - na1 * y1_ - na2 * y2_;
        x2_ = x1_; x1_ = x; y2_ = y1_; y1_ = y;
        return y;
    }
    void reset() { x1_ = x2_ = y1_ = y2_ = 0.0; }

private:
    double x1_ = 0, x2_ = 0, y1_ = 0, y2_ = 0;
};

// Feed-forward peak limiter in the spirit of the page's DynamicsCompressor (-1 dB, 20:1, hard knee,
// 2 ms attack, 100 ms release). No look-ahead, so a sharp transient can overshoot by a few samples.
class Limiter {
public:
    void init(double fs, double threshold_db = -1.0, double ratio = 20.0, double attack_s = 0.002,
              double release_s = 0.1) {
        thr_ = threshold_db; slope_ = 1.0 - 1.0 / ratio;
        att_ = std::exp(-1.0 / (attack_s * fs)); rel_ = std::exp(-1.0 / (release_s * fs));
    }
    double process(double x) {
        env_ = std::max(std::fabs(x), rel_ * env_);  // peak detector: instant rise, release_s fall
        const double level = 20.0 * std::log10(std::max(env_, 1e-9));
        const double want = level > thr_ ? -(level - thr_) * slope_ : 0.0;  // gain change in dB, <= 0
        const double a = want < gr_ ? att_ : rel_;
        gr_ = a * gr_ + (1.0 - a) * want;
        return x * std::pow(10.0, gr_ / 20.0);
    }
    double reduction_db() const { return gr_; }

private:
    double thr_ = -1.0, slope_ = 0.95, att_ = 0.0, rel_ = 0.0, gr_ = 0.0, env_ = 0.0;
};

class FilterChain {
public:
    static constexpr int kRamp = 256;  // one control frame

    void init(double fs) { fs_ = fs; lim_.init(fs); }

    // New targets (one control frame's values): ramp from where we are to them over one frame.
    void setTarget(int kind, const FilterParams& p) {
        if (kind != kind_) {  // a new type reinterprets Q: jump rather than ramp across types
            kind_ = kind; cur_ = p; from_ = p; to_ = p; left_ = 0;
            return;
        }
        from_ = cur_; to_ = p; left_ = kRamp;
    }
    void setMix(double mix) { mix_ = mix; }
    void setVolume(double gain) { volume_ = gain; }
    void setLimiter(bool on) { limit_ = on; }
    bool hasTarget() const { return kind_ >= 0; }

    double process(double x) {
        if (left_ > 0) {
            const double t = 1.0 - static_cast<double>(--left_) / kRamp;  // 1/256 .. 1
            cur_.cutoff = from_.cutoff + t * (to_.cutoff - from_.cutoff);
            cur_.q = from_.q + t * (to_.q - from_.q);
            cur_.gain_db = from_.gain_db + t * (to_.gain_db - from_.gain_db);
            cur_.vca = from_.vca + t * (to_.vca - from_.vca);
        }
        double wet = 0.0;
        if (kind_ >= 0) wet = biquad_.process(x, kind_, cur_.cutoff, cur_.q, cur_.gain_db, fs_) * cur_.vca;
        double y = volume_ * (mix_ * wet + (1.0 - mix_) * x);
        return limit_ ? lim_.process(y) : y;
    }
    const FilterParams& current() const { return cur_; }
    double reduction_db() const { return lim_.reduction_db(); }

private:
    double fs_ = 44100.0, mix_ = 1.0, volume_ = 1.0;
    bool limit_ = true;
    int kind_ = -1, left_ = 0;  // no filter until the first target: until then only the dry share passes
    FilterParams cur_, from_, to_;
    Biquad biquad_;
    Limiter lim_;
};

}  // namespace fly
