// Lightweight NEON helpers for hot loops (float-only)
#pragma once

#include <cstddef>
#include <cmath>

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

// Vectorized sum of float array
inline float sum_f32(const float* data, std::size_t n)
{
    float32x4_t vsum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    const std::size_t step = 4;

    for (; i + step <= n; i += step)
    {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vaddq_f32(vsum, v);
    }

    // Horizontal sum of the 4 lanes
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float result = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);

    // Handle remainder
    for (; i < n; ++i)
    {
        result += data[i];
    }

    return result;
}

// Vectorized sum of squares: sum(data[i]^2)
inline float sum_squares_f32(const float* data, std::size_t n)
{
    float32x4_t vsum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    const std::size_t step = 4;

    for (; i + step <= n; i += step)
    {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vmlaq_f32(vsum, v, v);  // vsum += v * v
    }

    // Horizontal sum
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float result = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);

    // Handle remainder
    for (; i < n; ++i)
    {
        result += data[i] * data[i];
    }

    return result;
}

// Vectorized max absolute value
inline float max_abs_f32(const float* data, std::size_t n)
{
    float32x4_t vmax = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    const std::size_t step = 4;

    for (; i + step <= n; i += step)
    {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t vabs = vabsq_f32(v);
        vmax = vmaxq_f32(vmax, vabs);
    }

    // Horizontal max
    float32x2_t vmax2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float result = vget_lane_f32(vmax2, 0);

    // Handle remainder
    for (; i < n; ++i)
    {
        float absVal = data[i] < 0 ? -data[i] : data[i];
        if (absVal > result) result = absVal;
    }

    return result;
}

// Vectorized weighted sum: sum(data[i] * i) for spectral centroid
// Also returns sum(data[i]) in *outSum
inline float weighted_sum_f32(const float* data, std::size_t n, float* outSum)
{
    float32x4_t vsum = vdupq_n_f32(0.0f);
    float32x4_t vwsum = vdupq_n_f32(0.0f);

    // Index vector for weights [0, 1, 2, 3]
    float indices[4] = {0.0f, 1.0f, 2.0f, 3.0f};
    float32x4_t vidx = vld1q_f32(indices);
    float32x4_t vincr = vdupq_n_f32(4.0f);

    std::size_t i = 0;
    const std::size_t step = 4;

    for (; i + step <= n; i += step)
    {
        float32x4_t v = vld1q_f32(data + i);
        vsum = vaddq_f32(vsum, v);
        vwsum = vmlaq_f32(vwsum, v, vidx);  // vwsum += v * idx
        vidx = vaddq_f32(vidx, vincr);       // idx += 4
    }

    // Horizontal sums
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float sumResult = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);

    float32x2_t vwsum2 = vadd_f32(vget_low_f32(vwsum), vget_high_f32(vwsum));
    float wsumResult = vget_lane_f32(vpadd_f32(vwsum2, vwsum2), 0);

    // Handle remainder
    for (; i < n; ++i)
    {
        sumResult += data[i];
        wsumResult += data[i] * (float)i;
    }

    *outSum = sumResult;
    return wsumResult;
}

// Vectorized sum of squares and max of squares (for spectral crest)
// Returns sum of squared values, and *outMax receives max squared value
inline float sum_squares_and_max_f32(const float* data, std::size_t n, float* outMax)
{
    float32x4_t vsum = vdupq_n_f32(0.0f);
    float32x4_t vmax = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    const std::size_t step = 4;

    for (; i + step <= n; i += step)
    {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t vsq = vmulq_f32(v, v);
        vsum = vaddq_f32(vsum, vsq);
        vmax = vmaxq_f32(vmax, vsq);
    }

    // Horizontal sum
    float32x2_t vsum2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
    float sumResult = vget_lane_f32(vpadd_f32(vsum2, vsum2), 0);

    // Horizontal max
    float32x2_t vmax2 = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float maxResult = vget_lane_f32(vmax2, 0);

    // Handle remainder
    for (; i < n; ++i)
    {
        float sq = data[i] * data[i];
        sumResult += sq;
        if (sq > maxResult) maxResult = sq;
    }

    *outMax = maxResult;
    return sumResult;
}

} // namespace gist_neon

#else // Scalar fallbacks for non-ARM platforms

namespace gist_neon {

inline void magnitude_f32(const float* r, const float* im, float* mag, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        float rr = r[i]*r[i];
        float ii = im[i]*im[i];
        mag[i] = std::sqrt(rr + ii);
    }
}

inline float sum_f32(const float* data, std::size_t n)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        result += data[i];
    return result;
}

inline float sum_squares_f32(const float* data, std::size_t n)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        result += data[i] * data[i];
    return result;
}

inline float max_abs_f32(const float* data, std::size_t n)
{
    float result = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float absVal = data[i] < 0 ? -data[i] : data[i];
        if (absVal > result) result = absVal;
    }
    return result;
}

inline float weighted_sum_f32(const float* data, std::size_t n, float* outSum)
{
    float sumResult = 0.0f;
    float wsumResult = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        sumResult += data[i];
        wsumResult += data[i] * (float)i;
    }
    *outSum = sumResult;
    return wsumResult;
}

inline float sum_squares_and_max_f32(const float* data, std::size_t n, float* outMax)
{
    float sumResult = 0.0f;
    float maxResult = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        float sq = data[i] * data[i];
        sumResult += sq;
        if (sq > maxResult) maxResult = sq;
    }
    *outMax = maxResult;
    return sumResult;
}

} // namespace gist_neon

#endif // USE_ARM_NEON && NEON

