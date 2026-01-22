// Force-enable NEON path when building PFFFT under the Bela IDE
// by compiling the C implementation as C++ with the right macros.

#ifndef PFFFT_ENABLE_NEON
#define PFFFT_ENABLE_NEON 1
#endif

// Silence verbose pragma messages from SIMD headers
#ifndef PFFFT_SILENCE_SIMD_MSG
#define PFFFT_SILENCE_SIMD_MSG 1
#endif

#if !defined(__arm__) && !defined(__aarch64__) && !defined(__arm64__)
#define __arm__ 1
#endif

extern "C" {
#include "pffft.c"
}
