#include "carfac_frontend.h"
#include <cmath>
#include <cstring>

#ifndef HAVE_CARFAC
#error "HAVE_CARFAC must be defined - stub mode no longer supported"
#endif

#include "carfac.h"
#include "ear.h"

bool CarfacFrontend::init(const Params& p)
{
    fs_ = p.sample_rate;
    hop_ = p.publish_hop;
    agc_ = p.enable_agc;
    if (fs_ <= 0) return false;

    // Compute decimation factor (CARFAC runs at reduced rate if carfac_rate specified)
    carfac_fs_ = (p.carfac_rate > 0) ? p.carfac_rate : fs_;
    decim_ = fs_ / carfac_fs_;
    if (decim_ < 1) decim_ = 1;
    carfac_fs_ = fs_ / decim_;  // Actual rate after integer division
    decim_cnt_ = 0;

    // Simple one-pole lowpass for anti-aliasing before decimation
    // Cutoff at ~0.4 * Nyquist of target rate
    if (decim_ > 1) {
        float fc = 0.4f * (carfac_fs_ / 2.0f);
        float rc = 1.0f / (2.0f * 3.14159f * fc);
        float dt = 1.0f / fs_;
        aa_coeff_ = dt / (rc + dt);
    } else {
        aa_coeff_ = 1.0f;  // No filtering needed
    }
    aa_state_ = 0.0f;

    // Build CAR/IHC/AGC params; CARFAC will design coeffs and allocate state.
    CARParams carp; IHCParams ihcp; AGCParams agcp;
    // erb_per_step = 1.0 (default) for full resolution; higher values reduce bands
    ihcp.just_half_wave_rectify = true;  // Bypass expensive cube/divide nonlinearity
    const int num_ears = 1;
    carfac_.reset(new CARFAC(num_ears, static_cast<FPType>(carfac_fs_), carp, ihcp, agcp));
    carfac_->Reset();

    // Read actual band count from CARFAC and allocate envelope arrays
    bands_ = carfac_->num_channels();
    band_sig_.assign(bands_, 0.0f);
    band_gain_.assign(bands_, 1.0f);
    fast_.assign(bands_, {});
    slow_.assign(bands_, {});
    for (int b = 0; b < bands_; ++b) {
        fast_[b].set(carfac_fs_, p.fast_attack, p.fast_release);
        slow_[b].set(carfac_fs_, p.slow_attack, p.slow_release);
    }

    // Cache ear pointer for inline per-sample processing
    ear_ = carfac_->mutable_ear(0);

    return true;
}

CarfacFrontend::~CarfacFrontend() = default;

void CarfacFrontend::processSample(float x)
{
    // Anti-aliasing lowpass filter (runs at input rate)
    aa_state_ += aa_coeff_ * (x - aa_state_);

    // Decimation: only run CARFAC every decim_ samples
    if (++decim_cnt_ < decim_) {
        return;
    }
    decim_cnt_ = 0;

    // Run CARFAC pipeline at reduced rate
    ear_->CARStep(static_cast<FPType>(aa_state_));
    ear_->IHCStep(ear_->car_out());
    bool agc_updated = ear_->AGCStep(ear_->ihc_out());
    if (agc_updated) {
        ear_->CloseAGCLoop(false);
    }

    // Update envelope followers (at CARFAC rate)
    const ArrayX& bm = ear_->zy_memory();
    for (int b = 0; b < bands_; ++b) {
        float a = std::fabs(static_cast<float>(bm(b)));
        fast_[b].step(a);
        slow_[b].step(a);
    }
}

void CarfacFrontend::publish(std::vector<float>& env_fast,
                             std::vector<float>& env_slow,
                             std::vector<float>& delta,
                             std::vector<float>* opt_gain)
{
    // Just copy current envelope state (processing already done in processSample)
    env_fast.resize(bands_);
    env_slow.resize(bands_);
    delta.resize(bands_);
    if (opt_gain) opt_gain->assign(bands_, 1.0f);

    for (int b = 0; b < bands_; ++b) {
        env_fast[b] = fast_[b].env;
        env_slow[b] = slow_[b].env;
        // Signed delta: positive = attack (energy rising), negative = decay (energy falling)
        delta[b] = fast_[b].env - slow_[b].env;
    }
}

void CarfacFrontend::publishFeatures(LatentInput& out)
{
    // Band boundaries for low/mid/high energy ratios
    // 71 bands split roughly into thirds: 0-23 (low), 24-47 (mid), 48-70 (high)
    constexpr int kLowEnd = 24;
    constexpr int kMidEnd = 48;
    constexpr float kAttackThreshold = 0.01f;  // Delta threshold for attack detection

    float total_energy = 0.0f;
    float weighted_sum = 0.0f;
    float low_energy = 0.0f;
    float mid_energy = 0.0f;
    float high_energy = 0.0f;
    float max_env = 0.0f;
    int peak_band = 0;
    int attack_count = 0;

    // First pass: collect env_fast, delta, and accumulate statistics
    for (int b = 0; b < bands_ && b < LatentInput::kNumBands; ++b) {
        float env = fast_[b].env;
        float d = fast_[b].env - slow_[b].env;

        out.env_fast[b] = env;
        out.delta[b] = d;

        total_energy += env;
        weighted_sum += static_cast<float>(b) * env;

        if (env > max_env) {
            max_env = env;
            peak_band = b;
        }

        if (d > kAttackThreshold)
            attack_count++;

        // Accumulate band energies
        if (b < kLowEnd)
            low_energy += env;
        else if (b < kMidEnd)
            mid_energy += env;
        else
            high_energy += env;
    }

    // Zero-pad if CARFAC has fewer than 71 bands
    for (int b = bands_; b < LatentInput::kNumBands; ++b) {
        out.env_fast[b] = 0.0f;
        out.delta[b] = 0.0f;
    }

    // Compute summary statistics
    out.total_energy = total_energy;

    // Spectral centroid: weighted average of band indices
    out.spectral_centroid = (total_energy > 1e-10f)
        ? weighted_sum / total_energy
        : static_cast<float>(bands_) / 2.0f;

    // Spectral spread: standard deviation around centroid
    float variance_sum = 0.0f;
    for (int b = 0; b < bands_ && b < LatentInput::kNumBands; ++b) {
        float diff = static_cast<float>(b) - out.spectral_centroid;
        variance_sum += diff * diff * out.env_fast[b];
    }
    out.spectral_spread = (total_energy > 1e-10f)
        ? std::sqrt(variance_sum / total_energy)
        : 0.0f;

    out.peak_band = static_cast<float>(peak_band);
    out.attack_breadth = static_cast<float>(attack_count);

    // Band ratios (normalized by total energy)
    float inv_total = (total_energy > 1e-10f) ? 1.0f / total_energy : 0.0f;
    out.band_ratios[0] = low_energy * inv_total;
    out.band_ratios[1] = mid_energy * inv_total;
    out.band_ratios[2] = high_energy * inv_total;
}
