// Minimal Bela-oriented example (not built by default).
// Demonstrates zero-copy, windowed fast path with PFFFT + NEON.

#include "Gist.h"
#include <vector>
#include <cmath>

int main() {
    const int fs = 44100;
    const int frameSize = 512; // power-of-two for PFFFT
    Gist<float> gist(frameSize, fs, HanningWindow);

    // Precompute window to apply upstream of the fast path
    std::vector<float> window = WindowFunctions<float>::createHanningWindow(frameSize);

    std::vector<float> input(frameSize, 0.0f);
    std::vector<float> windowed(frameSize, 0.0f);

    // Fill input[] from your Bela audio ring buffer here...

    // Apply window upstream, then call fast path (avoids extra copy into Gist)
    for (int i = 0; i < frameSize; ++i) windowed[i] = input[i] * window[i];
    gist.processWindowedFrame(windowed.data(), frameSize);

    // Read features immediately (frequency-domain examples)
    float centroid = gist.spectralCentroid();
    float hfc = gist.highFrequencyContent();

    // For pitch:
    float pitchHz = gist.pitch(); // requires time-domain path; use processAudioFrame() if needed

    // Use results...
    (void)centroid; (void)hfc; (void)pitchHz;
    return 0;
}

