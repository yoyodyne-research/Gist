#include <Bela.h>
#include <cmath>
#include <vector>
#include <cstring>
#include <atomic>
#include <memory>
#include "Gist.h"
#include "pffft.h" // for runtime SIMD check
#include "flenser.h" // multiband envelopes

// Auto-detect CARFAC headers staged in the project and enable frontend by default
#if !defined(HAVE_CARFAC)
#  if __has_include("carfac/upstream/cpp/carfac.h")
#    define HAVE_CARFAC 1
#  endif
#endif
#if !defined(ENABLE_CARFAC_FRONTEND)
#  if defined(HAVE_CARFAC)
#    define ENABLE_CARFAC_FRONTEND 1
#  else
#    define ENABLE_CARFAC_FRONTEND 0
#  endif
#endif

#if ENABLE_CARFAC_FRONTEND
#include "carfac_frontend.h"
#endif

#ifdef USE_BITSTREAM_PITCH
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
static constexpr int kFrameSize = 256;   // analysis window size
static constexpr int kHopSize   = 64;    // tighter responsiveness, keep CPU modest

// Analysis
static std::unique_ptr<Gist<float>> gGist;
static std::unique_ptr<Gist<float>> gGistPitch;  // separate instance for aux pitch
static std::unique_ptr<Flenser> gFlenser;        // multiband envelopes
#if ENABLE_CARFAC_FRONTEND
static std::unique_ptr<CarfacFrontend> gCarfac;
#endif
static std::vector<float> gRing;            // circular buffer of last kFrameSize samples
static std::vector<float> gFrame;           // contiguous copy (time-domain)
static std::vector<float> gPitchFrame;      // frame handed to aux pitch task
// Optionally compute MFCCs in aux task. Disable at compile time by defining
// ENABLE_MFCC=0 in settings.json (defines/cflags/cxxflags). When disabled,
// no MFCC work is done at runtime.
#if !defined(ENABLE_MFCC)
#define ENABLE_MFCC 0
#endif
#if ENABLE_MFCC
static std::vector<float> gMfcc;            // latest MFCCs (size ~13)
#endif
static std::vector<float> gEnvFast, gEnvSlow, gTransient; // multiband outputs
static int gFlenserDecim = 0;
static int gWriteIdx = 0;
static int gSamplesSinceHop = 0;
static AuxiliaryTask gPitchTask;
static std::atomic<bool> gPitchPending{false};

// Window configured inside Gist; no windowing needed here when using processAudioFrame()

// Outputs (read from non-RT contexts if needed)
static std::atomic<float> gRms{0.f};
static std::atomic<float> gCentroid{0.f};
static std::atomic<float> gRolloff{0.f};
static std::atomic<float> gHfc{0.f};
static std::atomic<float> gPitchHz{0.f};
static int gPitchDecim = 0;              // throttle expensive YIN calls

bool setup(BelaContext* context, void* userData)
{
    const int fs = context->audioSampleRate;
    gGist.reset(new Gist<float>(kFrameSize, fs, HanningWindow));
    gGistPitch.reset(new Gist<float>(kFrameSize, fs, HanningWindow));
    gFlenser.reset(new Flenser());
#if ENABLE_CARFAC_FRONTEND
    gCarfac.reset(new CarfacFrontend());
#endif
    #ifdef USE_BITSTREAM_PITCH
    // Apply preset or custom values via compile-time macros. Override in IDE:
    //  -DPITCH_PRESET=6 (GUITAR), 2 (BASS4), 1 (BASS6), etc.
    // Or set custom: -DPITCH_MIN_HZ=80.0f -DPITCH_HYST=0.02f
    applyPitchPreset(*gGistPitch);
    #endif

    gRing.assign(kFrameSize, 0.0f);
    gFrame.assign(kFrameSize, 0.0f);
    gPitchFrame.assign(kFrameSize, 0.0f);
#if ENABLE_MFCC
    gMfcc.assign(13, 0.0f);
#endif
#if !ENABLE_CARFAC_FRONTEND
    // Configure multiband envelopes (LR4) with 6 bands (SVF engine)
    {
        std::vector<float> edges;
        // Default edges (Hz). Adjust per preset if desired.
        #if defined(PITCH_PRESET) && (PITCH_PRESET==1)
            edges = { 40.f, 120.f, 300.f, 800.f, 2000.f, 5000.f };
        #elif defined(PITCH_PRESET) && (PITCH_PRESET==2)
            edges = { 50.f, 140.f, 320.f, 900.f, 2200.f, 5200.f };
        #elif defined(PITCH_PRESET) && (PITCH_PRESET==6)
            edges = { 80.f, 180.f, 400.f, 1000.f, 2500.f, 6000.f };
        #else
            edges = { 60.f, 150.f, 350.f, 900.f, 2300.f, 5500.f };
        #endif
        gFlenser->configure(fs, edges, 0.005f, 0.040f, 0.020f, 0.200f);
        const size_t bands = edges.size() + 1;
        gEnvFast.assign(bands, 0.0f);
        gEnvSlow.assign(bands, 0.0f);
        gTransient.assign(bands, 0.0f);
        rt_printf("Flenser bands=%zu\n", bands);
    }
#endif

#if ENABLE_CARFAC_FRONTEND
    // Configure CARFAC frontend (start with ~16 bands)
    {
        CarfacFrontend::Params cp;
        cp.sample_rate = fs;
        cp.num_channels = 16;
        cp.enable_agc = true;
        cp.publish_hop = kHopSize * 4; // align with our spec cadence (~256)
        gCarfac->init(cp);
        const size_t bands = gCarfac->numBands();
        gEnvFast.assign(bands, 0.0f);
        gEnvSlow.assign(bands, 0.0f);
        gTransient.assign(bands, 0.0f);
        rt_printf("CARFAC bands=%zu\n", bands);
    }
#endif
    gWriteIdx = 0;
    gSamplesSinceHop = 0;

    // Log frontend + SIMD backend (non-RT context)
#if ENABLE_CARFAC_FRONTEND
    rt_printf("Frontend: CARFAC\n");
#else
    rt_printf("Frontend: Flenser (SVF)\n");
#endif
    rt_printf("PFFFT SIMD: %s (size=%d)\n", pffft_simd_arch(), pffft_simd_size());

    // Enable flush-to-zero and denormals-are-zero (avoid rare CPU spikes)
    #if (defined(__arm__) || defined(__aarch64__) || defined(__arm64__)) && defined(__ARM_NEON)
    {
        unsigned int fpscr = 0;
        // Read FPSCR
        asm volatile ("vmrs %0, fpscr" : "=r" (fpscr));
        // Set FZ (bit 24) and DN (bit 19)
        fpscr |= (1u << 24) | (1u << 19);
        // Write FPSCR
        asm volatile ("vmsr fpscr, %0" :: "r" (fpscr));
    }
    #endif

    gPitchTask = Bela_createAuxiliaryTask([](void*) {
        if (!gPitchPending.load(std::memory_order_acquire))
            return;
        // Compute pitch on separate instance to avoid contention
        gGistPitch->processAudioFrame(gPitchFrame.data(), kFrameSize);
        #ifdef USE_BITSTREAM_PITCH
        gPitchHz.store(gGistPitch->pitchFast(), std::memory_order_relaxed);
        #else
        gPitchHz.store(gGistPitch->pitch(), std::memory_order_relaxed);
        #endif
        #if ENABLE_MFCC
        // Compute MFCCs on the aux instance and copy out
        const auto& mfcc = gGistPitch->getMelFrequencyCepstralCoefficients();
        const size_t n = std::min(gMfcc.size(), mfcc.size());
        if (n)
            std::memcpy(gMfcc.data(), mfcc.data(), n * sizeof(float));
        #endif
        gPitchPending.store(false, std::memory_order_release);
    }, 50, "gist-pitch-task", nullptr);
    return true;
}

static inline void extractFrameContiguous()
{
    // Copy ring -> contiguous frame in time order [oldest..newest)
    const int tail = kFrameSize - gWriteIdx;
    std::memcpy(gFrame.data(),            gRing.data() + gWriteIdx, tail * sizeof(float));
    std::memcpy(gFrame.data() + tail,     gRing.data(),             gWriteIdx * sizeof(float));
}

void render(BelaContext* context, void* userData)
{
    const unsigned int nFrames = context->audioFrames;

    for (unsigned int n = 0; n < nFrames; ++n)
    {
        // Mono input (ch 0). Replace with whatever input you use.
        float x = audioRead(context, n, 0);

        // Push to ring buffer
        gRing[gWriteIdx] = x;
        gWriteIdx = (gWriteIdx + 1) % kFrameSize;
        gSamplesSinceHop++;

        // Multiband envelopes per-sample
#if ENABLE_CARFAC_FRONTEND
        gCarfac->processSample(x);
#else
        gFlenser->processSample(x);
#endif

        if (gSamplesSinceHop >= kHopSize)
        {
            gSamplesSinceHop -= kHopSize;

            // Build a contiguous frame and process (Gist applies window internally)
            extractFrameContiguous();
            gGist->processAudioFrame(gFrame.data(), kFrameSize);

            // Frequency-domain features (lightweight; still, compute every other hop if needed)
            static int specDecim = 0;
            if ((specDecim++ & 0x3) == 0) { // every 4 hops
                gCentroid.store(gGist->spectralCentroid(), std::memory_order_relaxed);
                gRolloff.store(gGist->spectralRolloff(), std::memory_order_relaxed);
                gHfc.store(gGist->highFrequencyContent(), std::memory_order_relaxed);

                // MFCCs moved to aux task to reduce RT load
            }

            // Time-domain + pitch (pitch is heavy; offload to aux task)
            gRms.store(gGist->rootMeanSquare(), std::memory_order_relaxed);
            if ((gPitchDecim++ & 0x3) == 0) { // schedule every 4 hops
                if (!gPitchPending.load(std::memory_order_acquire)) {
                    // hand over a copy of the current frame to the aux task
                    std::memcpy(gPitchFrame.data(), gFrame.data(), kFrameSize * sizeof(float));
                    gPitchPending.store(true, std::memory_order_release);
                    Bela_scheduleAuxiliaryTask(gPitchTask);
                }
            }

            // Decimate and publish multiband envelopes (every 256 samples)
            if ((gFlenserDecim++ & 0x3) == 0) {
#if ENABLE_CARFAC_FRONTEND
                gCarfac->publish(gEnvFast, gEnvSlow, gTransient, nullptr);
#else
                const auto& ef = gFlenser->env_fast();
                const auto& es = gFlenser->env_slow();
                const auto& tr = gFlenser->transients();
                std::memcpy(gEnvFast.data(), ef.data(), ef.size()*sizeof(float));
                std::memcpy(gEnvSlow.data(), es.data(), es.size()*sizeof(float));
                std::memcpy(gTransient.data(), tr.data(), tr.size()*sizeof(float));
#endif
            }
        }
    }
}

void cleanup(BelaContext* context, void* userData)
{
    gGist.reset();
}
