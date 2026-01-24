#include "carfac_frontend.h"
#include <cmath>
#include <cstring>

// Enable the real CARFAC path when upstream sources are present.
// The submodule places headers under libs/carfac/upstream/cpp, which the
// deploy script copies into the staged project's carfac/ folder.
// Adjust include as needed based on your layout.
#if defined(HAVE_CARFAC)
#include "carfac/upstream/cpp/carfac.h"
#endif

bool CarfacFrontend::init(const Params& p)
{
    fs_ = p.sample_rate;
    bands_ = p.num_channels;
    hop_ = p.publish_hop;
    agc_ = p.enable_agc;
    if (bands_ <= 0 || fs_ <= 0) return false;

    band_sig_.assign(bands_, 0.0f);
    band_gain_.assign(bands_, 1.0f);
    fast_.assign(bands_, {});
    slow_.assign(bands_, {});
    for (int b = 0; b < bands_; ++b) {
        fast_[b].set(fs_, p.fast_attack, p.fast_release);
        slow_[b].set(fs_, p.slow_attack, p.slow_release);
    }

#ifdef HAVE_CARFAC
    // Build default CAR/IHC/AGC params; CARFAC will design coeffs and allocate state.
    CARParams carp; IHCParams ihcp; AGCParams agcp;
    // Tune a few key params to match requested bands and AGC on/off
    // CARFAC computes num_channels from params; we can adjust erb_per_step to hit band count.
    // Here we let CARFAC pick and read back num_channels after design.
    const int num_ears = 1;
    carfac_.reset(new CARFAC(num_ears, static_cast<FPType>(fs_), carp, ihcp, agcp));
    carfac_->Reset();
    bands_ = carfac_->num_channels();
    band_sig_.assign(bands_, 0.0f);
    band_gain_.assign(bands_, 1.0f);
#endif
    return true;
}

CarfacFrontend::~CarfacFrontend() = default;

void CarfacFrontend::processSample(float x)
{
#ifdef HAVE_CARFAC
    // Buffer samples and process in publish(), using CARFAC::RunSegment for the hop.
    buffer_.push_back(x);
#else
    // Stub path: distribute the sample energy into a crude spectral tilt across bands.
    // Step envelope followers per-sample for correct timing.
    float a = std::fabs(x);
    for (int b = 0; b < bands_; ++b) {
        float w = 1.0f - (float)b / std::max(1, bands_-1); // simple tilt
        float sig = w * a;
        band_sig_[b] = sig;
        // Step envelopes per-sample (stored in fast_/slow_ for publish to read)
        fast_[b].step(sig);
        slow_[b].step(sig);
    }
#endif
}

void CarfacFrontend::publish(std::vector<float>& env_fast,
                             std::vector<float>& env_slow,
                             std::vector<float>& transient,
                             std::vector<float>* opt_gain)
{
    env_fast.resize(bands_);
    env_slow.resize(bands_);
    transient.resize(bands_);
    if (opt_gain) opt_gain->assign(bands_, 1.0f);

    #ifdef HAVE_CARFAC
    // Build a [num_ears x num_samples] segment from buffered samples
    const int T = static_cast<int>(buffer_.size());
    if (T > 0) {
        ArrayXX seg(1, T);
        for (int t = 0; t < T; ++t) seg(0, t) = static_cast<FPType>(buffer_[t]);
        CARFACOutput out(/*store_nap=*/false, /*store_bm=*/true, /*store_ohc=*/false, /*store_agc=*/false);
        carfac_->RunSegment(seg, /*open_loop=*/false, &out);
        // Process BM frames through our AR followers to yield fast/slow/transient
        const auto& bm_ears = out.bm();
        const ArrayXX& bm = bm_ears[0]; // [num_channels x T]
        for (int t = 0; t < T; ++t) {
            for (int b = 0; b < bands_; ++b) {
                float a = std::fabs(static_cast<float>(bm(b, t)));
                float ef = fast_[b].step(a);
                float es = slow_[b].step(a);
                // keep last values; we'll copy at the end
                env_fast[b] = ef;
                env_slow[b] = es;
            }
        }
        for (int b = 0; b < bands_; ++b) {
            float tval = env_fast[b] - env_slow[b];
            transient[b] = (tval > 0.f ? tval : 0.f);
        }
        buffer_.clear();
    } else {
        // No new samples; just keep previous envelope values
        for (int b = 0; b < bands_; ++b) {
            float tval = env_fast[b] - env_slow[b];
            transient[b] = (tval > 0.f ? tval : 0.f);
        }
    }
    #else
    // Stub: read envelope values (already stepped per-sample in processSample)
    for (int b = 0; b < bands_; ++b) {
        env_fast[b] = fast_[b].env;
        env_slow[b] = slow_[b].env;
        float t = env_fast[b] - env_slow[b];
        transient[b] = (t > 0.f ? t : 0.f);
    }
    #endif
}
