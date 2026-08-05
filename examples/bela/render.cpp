#include <Bela.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <atomic>
#include <memory>

// Feature extraction method toggles (set via CPPFLAGS or settings.json)
#if !defined(ENABLE_CARFAC)
#define ENABLE_CARFAC 1
#endif
#if !defined(ENABLE_GIST_SPECTRAL)
#define ENABLE_GIST_SPECTRAL 0
#endif
#if !defined(ENABLE_GIST_RMS)
#define ENABLE_GIST_RMS 0
#endif
#if !defined(ENABLE_PITCH)
#define ENABLE_PITCH 0
#endif
#if !defined(ENABLE_MFCC)
#define ENABLE_MFCC 0
#endif
#if !defined(ENABLE_SCOPE)
#define ENABLE_SCOPE 0
#endif
#if !defined(ENABLE_LATENT)
#define ENABLE_LATENT 0
#endif
#if !defined(CARFAC_RATE)
#define CARFAC_RATE 0  // 0 = same as audio rate; set to e.g. 22050 for half-rate
#endif

// Conditional includes
#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH || ENABLE_MFCC
#include "Gist.h"
#include "pffft.h"
#endif
#if ENABLE_CARFAC
#include "carfac_frontend.h"
#endif
#if ENABLE_LATENT && ENABLE_CARFAC
#include "latent_layer.h"
#endif
#if ENABLE_SCOPE
#include <libraries/Scope/Scope.h>
#endif

#if ENABLE_PITCH && defined(USE_BITSTREAM_PITCH)
// Compile-time tunables for bitstream pitch presets
#ifndef PITCH_MIN_HZ
#define PITCH_MIN_HZ 40.0f     // bass low B ≈ 30.87 Hz; 4-string low E ≈ 41.2 Hz
#endif
#ifndef PITCH_MAX_HZ
#define PITCH_MAX_HZ 1000.0f
#endif
#ifndef PITCH_HYST
#define PITCH_HYST   0.03f     // linear hysteresis; raise for noisier inputs
#endif

// Optional preset selector (numeric). 0 = custom (use PITCH_MIN_HZ/etc.)
// 1=BASS6 (low B), 2=BASS4 (low E), 3=PICCOLO_BASS, 4=ALTO_BASS,
// 5=BARITONE_GUITAR, 6=GUITAR
#ifndef PITCH_PRESET
#define PITCH_PRESET 0
#endif

static inline void applyPitchPreset(Gist<float>& gist)
{
    float minHz = PITCH_MIN_HZ;
    float maxHz = PITCH_MAX_HZ;
    float hyst  = PITCH_HYST;
    switch (PITCH_PRESET)
    {
        case 1: /* BASS6 (low B ≈ 30.87 Hz) */
            minHz = 30.0f; break;
        case 2: /* BASS4 (low E ≈ 41.2 Hz) */
            minHz = 40.0f; break;
        case 3: /* PICCOLO_BASS (≈ one octave up) */
            minHz = 80.0f; break;
        case 4: /* ALTO_BASS (between bass and bari) */
            minHz = 60.0f; break;
        case 5: /* BARITONE_GUITAR (A1 ≈ 55 Hz) */
            minHz = 50.0f; break;
        case 6: /* GUITAR (E2 ≈ 82.4 Hz) */
            minHz = 80.0f; break;
        default: break; // custom values above
    }
    gist.setBitstreamPitchRange(minHz, maxHz);
    gist.setBitstreamPitchHysteresis(hyst);
    rt_printf("Bitstream pitch preset=%d min=%.2fHz max=%.2fHz hyst=%.3f\n",
              PITCH_PRESET, minHz, maxHz, hyst);
}
#endif

// Configuration
#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH || ENABLE_CARFAC
static constexpr int kHopSize   = 64;    // hop size for frame-based processing
#endif

#if ENABLE_SCOPE
static int gNumScopeBands = 0;  // Set dynamically from CARFAC band count
static std::vector<float> gScopeBuffer;  // Buffer for scope logging
#endif

// Analysis objects
#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS
static std::unique_ptr<Gist<float>> gGist;
#endif
#if ENABLE_PITCH
static std::unique_ptr<Gist<float>> gGistPitch;  // separate instance for aux pitch
static std::vector<float> gPitchFrame;           // frame handed to aux pitch task
static AuxiliaryTask gPitchTask;
static std::atomic<bool> gPitchPending{false};
static std::atomic<float> gPitchHz{0.f};
static int gPitchDecim = 0;
#endif
#if ENABLE_CARFAC
static std::unique_ptr<CarfacFrontend> gCarfac;
static std::vector<float> gEnvFast, gEnvSlow, gDelta;
static int gCarfacDecim = 0;
#endif
#if ENABLE_LATENT && ENABLE_CARFAC
static LatentLayer gLatentLayer;
static LatentOutput gLatentOutput;
static int gLatentDecim = 0;
// Control rate decimation: 689 Hz (feature rate) / 3 = ~230 Hz
static constexpr int kLatentControlDecim = 3;
#endif
#if ENABLE_SCOPE
static Scope gScope;
#endif
#if ENABLE_MFCC
static std::vector<float> gMfcc;
#endif

// Shared state for frame-based processing
#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH
static std::vector<float> gRing;
static std::vector<float> gFrame;
static int gWriteIdx = 0;
static int gSamplesSinceHop = 0;
#endif

// Outputs (read from non-RT contexts if needed)
#if ENABLE_GIST_RMS
static std::atomic<float> gRms{0.f};
#endif
#if ENABLE_GIST_SPECTRAL
static std::atomic<float> gCentroid{0.f};
static std::atomic<float> gRolloff{0.f};
static std::atomic<float> gHfc{0.f};
#endif

bool setup(BelaContext* context, void* userData)
{
    const int fs = context->audioSampleRate;

    // Enable flush-to-zero and denormals-are-zero (avoid rare CPU spikes)
    {
        unsigned int fpscr = 0;
        asm volatile ("vmrs %0, fpscr" : "=r" (fpscr));
        fpscr |= (1u << 24) | (1u << 19);  // FZ (bit 24), DN (bit 19)
        asm volatile ("vmsr fpscr, %0" :: "r" (fpscr));
    }

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS
    gGist.reset(new Gist<float>(kFrameSize, fs, HanningWindow));
    rt_printf("Gist: spectral=%d rms=%d\n", ENABLE_GIST_SPECTRAL, ENABLE_GIST_RMS);
#endif

#if ENABLE_PITCH
    gGistPitch.reset(new Gist<float>(kFrameSize, fs, HanningWindow));
    gPitchFrame.assign(kFrameSize, 0.0f);
#if defined(USE_BITSTREAM_PITCH)
    applyPitchPreset(*gGistPitch);
#endif
    gPitchTask = Bela_createAuxiliaryTask([](void*) {
        if (!gPitchPending.load(std::memory_order_acquire))
            return;
        gGistPitch->processAudioFrame(gPitchFrame.data(), kFrameSize);
#if defined(USE_BITSTREAM_PITCH)
        gPitchHz.store(gGistPitch->pitchFast(), std::memory_order_relaxed);
#else
        gPitchHz.store(gGistPitch->pitch(), std::memory_order_relaxed);
#endif
#if ENABLE_MFCC
        const auto& mfcc = gGistPitch->getMelFrequencyCepstralCoefficients();
        const size_t n = std::min(gMfcc.size(), mfcc.size());
        if (n)
            std::memcpy(gMfcc.data(), mfcc.data(), n * sizeof(float));
#endif
        gPitchPending.store(false, std::memory_order_release);
    }, 50, "gist-pitch-task", nullptr);
    rt_printf("Pitch: enabled\n");
#endif

#if ENABLE_MFCC
    gMfcc.assign(13, 0.0f);
    rt_printf("MFCC: enabled\n");
#endif

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH
    gRing.assign(kFrameSize, 0.0f);
    gFrame.assign(kFrameSize, 0.0f);
    gWriteIdx = 0;
    gSamplesSinceHop = 0;
#endif

#if ENABLE_CARFAC
    gCarfac.reset(new CarfacFrontend());
    {
        CarfacFrontend::Params cp;
        cp.sample_rate = fs;
        cp.carfac_rate = CARFAC_RATE;  // 0 = full rate, else decimated
        cp.enable_agc = true;
        cp.publish_hop = kHopSize * 4;
        gCarfac->init(cp);
        rt_printf("CARFAC rate: %d Hz (input: %d Hz, decim: %dx)\n",
                  cp.carfac_rate ? cp.carfac_rate : fs, fs,
                  cp.carfac_rate ? (fs / cp.carfac_rate) : 1);
        const size_t bands = gCarfac->numBands();
        gEnvFast.assign(bands, 0.0f);
        gEnvSlow.assign(bands, 0.0f);
        gDelta.assign(bands, 0.0f);
        rt_printf("CARFAC: %zu bands\n", bands);
    }
#endif

#if ENABLE_SCOPE && ENABLE_CARFAC
    gNumScopeBands = gCarfac->numBands();
    gScopeBuffer.resize(gNumScopeBands);
    gScope.setup(gNumScopeBands, fs);
    rt_printf("Scope: %d CARFAC bands -> channels\n", gNumScopeBands);
#endif

#if ENABLE_LATENT && ENABLE_CARFAC
    {
        // Load latent model from project directory
        const char* model_path = "/root/Bela/projects/audio_features/latent_model.json";
        if (gLatentLayer.loadParams(model_path)) {
            // Set smoothing for control rate (~230 Hz)
            float control_rate = static_cast<float>(fs) / kHopSize / kLatentControlDecim;
            gLatentLayer.setSmoothingMs(30.0f, control_rate);
            rt_printf("Latent layer loaded: %d PCA dims, %d GMM components\n",
                      LatentLayer::kPcaDims, LatentLayer::kGmmComponents);
        } else {
            rt_printf("Warning: Latent model not found at %s\n", model_path);
        }
    }
#endif

    return true;
}

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH
static inline void extractFrameContiguous()
{
    const int tail = kFrameSize - gWriteIdx;
    std::memcpy(gFrame.data(),        gRing.data() + gWriteIdx, tail * sizeof(float));
    std::memcpy(gFrame.data() + tail, gRing.data(),             gWriteIdx * sizeof(float));
}
#endif

void render(BelaContext* context, void* userData)
{
    const unsigned int nFrames = context->audioFrames;

    for (unsigned int n = 0; n < nFrames; ++n)
    {
        // Mono input (ch 1 = right). Replace with whatever input you use.
        float x = audioRead(context, n, 1);

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH
        // Push to ring buffer for frame-based processing
        gRing[gWriteIdx] = x;
        gWriteIdx = (gWriteIdx + 1) % kFrameSize;
        gSamplesSinceHop++;
#endif

#if ENABLE_CARFAC
        // CARFAC cochlear processing per-sample
        gCarfac->processSample(x);
#endif

#if ENABLE_SCOPE && ENABLE_CARFAC
        // Log all CARFAC bands to scope (per-sample envelope values)
        for (int b = 0; b < gNumScopeBands; ++b) {
            gScopeBuffer[b] = gCarfac->getEnvFast(b);
        }
        gScope.log(gScopeBuffer.data());
#endif

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS || ENABLE_PITCH
        if (gSamplesSinceHop >= kHopSize)
        {
            gSamplesSinceHop -= kHopSize;
            extractFrameContiguous();

#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS
            gGist->processAudioFrame(gFrame.data(), kFrameSize);
#endif

#if ENABLE_GIST_SPECTRAL
            static int specDecim = 0;
            if ((specDecim++ & 0x3) == 0) {
                gCentroid.store(gGist->spectralCentroid(), std::memory_order_relaxed);
                gRolloff.store(gGist->spectralRolloff(), std::memory_order_relaxed);
                gHfc.store(gGist->highFrequencyContent(), std::memory_order_relaxed);
            }
#endif

#if ENABLE_GIST_RMS
            gRms.store(gGist->rootMeanSquare(), std::memory_order_relaxed);
#endif

#if ENABLE_PITCH
            if ((gPitchDecim++ & 0x3) == 0) {
                if (!gPitchPending.load(std::memory_order_acquire)) {
                    std::memcpy(gPitchFrame.data(), gFrame.data(), kFrameSize * sizeof(float));
                    gPitchPending.store(true, std::memory_order_release);
                    Bela_scheduleAuxiliaryTask(gPitchTask);
                }
            }
#endif
        }
#endif

#if ENABLE_CARFAC
        // Publish CARFAC envelopes and run latent layer at hop boundaries
        if ((gCarfacDecim++ & (kHopSize - 1)) == 0) {
            gCarfac->publish(gEnvFast, gEnvSlow, gDelta, nullptr);

#if ENABLE_LATENT
            // Run latent layer at control rate (~230 Hz)
            // Hop rate is ~689 Hz at 44.1k; decimate by 3 to reach ~230 Hz
            if ((gLatentDecim++ % kLatentControlDecim) == 0) {
                if (gLatentLayer.isLoaded()) {
                    LatentInput lin;
                    gCarfac->publishFeatures(lin);
                    gLatentLayer.process(lin, gLatentOutput);

                    // gLatentOutput.z[0..7] and gLatentOutput.r[0..13] now available
                    // for mapping to MIDI CC, CV, or other outputs
                }
            }
#endif  // ENABLE_LATENT
        }
#endif  // ENABLE_CARFAC
    }
}

void cleanup(BelaContext* context, void* userData)
{
#if ENABLE_GIST_SPECTRAL || ENABLE_GIST_RMS
    gGist.reset();
#endif
#if ENABLE_PITCH
    gGistPitch.reset();
#endif
#if ENABLE_CARFAC
    gCarfac.reset();
#endif
}
