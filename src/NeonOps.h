// Lightweight NEON helpers for hot loops (float-only)
#pragma once

#include <cstddef>

#if defined(USE_ARM_NEON) && defined(__ARM_NEON) && !defined(__ARM_NEON__)
#define __ARM_NEON__ 1
#endif

#if defined(USE_ARM_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>

namespace gist_neon {

// Compute magnitude spectrum: mag[i] = sqrt(r[i]^2 + i[i]^2)
// n must be the number of bins (e.g., frameSize/2)
inline void magnitude_f32(const float* r, const float* im, float* mag, std::size_t n)
{
    std::size_t i = 0;
    const std::size_t step = 4;
    for (; i + step <= n; i += step)
    {
        float32x4_t vr  = vld1q_f32(r + i);
        float32x4_t vi  = vld1q_f32(im + i);
        float32x4_t rr  = vmulq_f32(vr, vr);
        float32x4_t ii  = vmulq_f32(vi, vi);
        float32x4_t sum = vaddq_f32(rr, ii);
        float32x4_t ms  = vsqrtq_f32(sum);
        vst1q_f32(mag + i, ms);
    }
    for (; i < n; ++i)
    {
        float rr = r[i]*r[i];
        float ii = im[i]*im[i];
        mag[i] = std::sqrt(rr + ii);
    }
}

} // namespace gist_neon

#endif // USE_ARM_NEON && NEON

