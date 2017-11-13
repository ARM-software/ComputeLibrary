/*
 * Copyright (c) 2016, 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/NEON/kernels/NEMagnitudePhaseKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
// Defines for computing atan2
constexpr float SCALE_FACTOR = 0.7111111111111111f;
constexpr float PI           = 3.141592653589793f;
constexpr float SCALE_180    = 180.0f / PI;
constexpr float SCALE_360    = SCALE_180 * SCALE_FACTOR;
constexpr float PI_4         = 0.7853981633974483f;
constexpr float COEFF1       = 0.0663f;
constexpr float COEFF2       = 0.2447f;
} // namespace

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace fp16
{
inline float16x8_t inv(float16x8_t x)
{
    const float16x8_t estimate = vrecpeq_f16(x);
    return vmulq_f16(estimate, vrecpsq_f16(x, estimate));
}

inline float16x8_t atan2_fast(float16x8_t gx, float16x8_t gy, float16x8_t scale)
{
    static const float16x8_t one     = vdupq_n_f16(1.0f);
    static const float16x8_t ninety  = vdupq_n_f16(90.f * SCALE_FACTOR);
    static const float16x8_t epsilon = vdupq_n_f16(1e-9f);
    static const float16x8_t piover4 = vdupq_n_f16(PI_4);
    static const float16x8_t coeff1  = vdupq_n_f16(COEFF1);
    static const float16x8_t coeff2  = vdupq_n_f16(COEFF2);

    const float16x8_t abs_gx = vabsq_f16(gx);
    const float16x8_t abs_gy = vabsq_f16(gy);
    const float16x8_t tmin   = vminq_f16(abs_gx, abs_gy);
    const float16x8_t tmax   = vmaxq_f16(abs_gx, abs_gy);

    // z = min(x, y) / max(x, y)
    const float16x8_t z    = vmulq_f16(tmin, inv(vaddq_f16(tmax, epsilon)));
    const float16x8_t absz = vabsq_f16(z);

    //                   = x * [pi/4 + (1 - |x|) * (0.2447 + 0.0663 * |x|)]
    float16x8_t arctan = vmulq_f16(z, vfmaq_f16(piover4,
                                                vsubq_f16(one, absz),
                                                vfmaq_f16(coeff2, coeff1, absz)));

    // Radians to degrees conversion with applied a scale factor in order to have the result [0, 255]
    arctan = vmulq_f16(arctan, scale);

    /* If z > 1, result = 90 - result */
    return vbslq_f16(vcgeq_f16(abs_gx, abs_gy), arctan, vsubq_f16(ninety, arctan));
}

inline float16x8_t atan2_0_360(float16x8_t gx, float16x8_t gy)
{
    static const float16x8_t scale      = vdupq_n_f16(SCALE_360);
    static const float16x8_t threesixty = vdupq_n_f16(360.0f * SCALE_FACTOR);
    static const float16x8_t zero       = vdupq_n_f16(0.0f);
    static const float16x8_t oneeighty  = vdupq_n_f16(180.0f * SCALE_FACTOR);

    float16x8_t arctan = atan2_fast(gx, gy, scale);

    // Choose correct quadrant
    arctan = vbslq_f16(vcltq_f16(gx, zero), vsubq_f16(oneeighty, arctan), arctan);
    arctan = vbslq_f16(vcltq_f16(gy, zero), vsubq_f16(threesixty, arctan), arctan);

    return arctan;
}

inline float16x8_t atan2_0_180(float16x8_t gx, float16x8_t gy)
{
    static const float16x8_t scale      = vdupq_n_f16(SCALE_180);
    static const float16x8_t threesixty = vdupq_n_f16(360.0f * SCALE_FACTOR);
    static const float16x8_t oneeighty  = vdupq_n_f16(180.0f * SCALE_FACTOR);
    static const float16x8_t zero       = vdupq_n_f16(0.0f);

    float16x8_t arctan = atan2_fast(gx, gy, scale);

    // Choose correct quadrant
    arctan = vbslq_f16(vcltq_f16(gx, zero), vsubq_f16(oneeighty, arctan), arctan);
    arctan = vbslq_f16(vcltq_f16(gy, zero), vsubq_f16(threesixty, arctan), arctan);
    arctan = vbslq_f16(vcgtq_f16(arctan, oneeighty), vsubq_f16(arctan, oneeighty), arctan);

    return arctan;
}

inline float32x4_t invsqrtv(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);

    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                                sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                                sqrt_reciprocal);

    return sqrt_reciprocal;
}

inline float32x4_t sqrtv(float32x4_t x)
{
    float32x4_t res = vdupq_n_f32(0.5f);
    return vmlaq_f32(res, x, invsqrtv(x));
}

inline int16x8_t magnitude_l1(int16x8_t input1, int16x8_t input2)
{
    return vqaddq_s16(vqabsq_s16(input1), vqabsq_s16(input2));
}

inline int16x8_t magnitude_l2(int16x8_t input1, int16x8_t input2)
{
    const int32x4x2_t square_x =
    {
        vmull_s16(vget_low_s16(input1), vget_low_s16(input1)),
        vmull_s16(vget_high_s16(input1), vget_high_s16(input1))
    };

    const int32x4x2_t square_y =
    {
        vmull_s16(vget_low_s16(input2), vget_low_s16(input2)),
        vmull_s16(vget_high_s16(input2), vget_high_s16(input2))
    };

    const uint32x4x2_t sum =
    {
        vaddq_u32(vreinterpretq_u32_s32(square_x.val[0]),
        vreinterpretq_u32_s32(square_y.val[0])),
        vaddq_u32(vreinterpretq_u32_s32(square_x.val[1]),
        vreinterpretq_u32_s32(square_y.val[1]))
    };

    const float32x4x2_t res =
    {
        sqrtv(vcvtq_f32_u32(sum.val[0])),
        sqrtv(vcvtq_f32_u32(sum.val[1]))
    };

    return vcombine_s16(vqmovn_s32(vcvtq_s32_f32(res.val[0])),
                        vqmovn_s32(vcvtq_s32_f32(res.val[1])));
}

inline uint8x8_t phase_signed(int16x8_t input1, int16x8_t input2)
{
    static const float16x8_t zeropointfive = vdupq_n_f16(0.5f);

    const float16x8_t inputx_f16 = vcvtq_f16_s16(input1);
    const float16x8_t inputy_f16 = vcvtq_f16_s16(input2);

    // Compute fast atan2
    const float16x8_t angle = atan2_0_360(inputx_f16, inputy_f16);

    return vqmovun_s16(vcvtq_s16_f16(vaddq_f16(angle, zeropointfive)));
}

inline uint8x8_t phase_unsigned(int16x8_t input1, int16x8_t input2)
{
    static const float16x8_t zeropointfive = vdupq_n_f16(0.5f);

    const float16x8_t inputx_f16 = vcvtq_f16_s16(input1);
    const float16x8_t inputy_f16 = vcvtq_f16_s16(input2);

    // Compute fast atan2
    const float16x8_t angle = atan2_0_180(inputx_f16, inputy_f16);

    return vqmovun_s16(vcvtq_s16_f16(vaddq_f16(angle, zeropointfive)));
}

template <MagnitudeType mag_type>
inline int16x8x2_t compute_magnitude(const int16x8x2_t &in0, const int16x8x2_t &gx);

template <>
inline int16x8x2_t compute_magnitude<MagnitudeType::L2NORM>(const int16x8x2_t &in0, const int16x8x2_t &gx)
{
    const int16x8x2_t mag =
    {
        magnitude_l2(in0.val[0], gx.val[0]),
        magnitude_l2(in0.val[1], gx.val[1])
    };

    return mag;
}

template <>
inline int16x8x2_t compute_magnitude<MagnitudeType::L1NORM>(const int16x8x2_t &in0, const int16x8x2_t &gx)
{
    const int16x8x2_t mag =
    {
        magnitude_l1(in0.val[0], gx.val[0]),
        magnitude_l1(in0.val[1], gx.val[1])
    };

    return mag;
}

template <PhaseType phase_type>
inline uint8x16_t compute_phase(const int16x8x2_t &in0, const int16x8x2_t &gx);

template <>
inline uint8x16_t compute_phase<PhaseType::SIGNED>(const int16x8x2_t &in0, const int16x8x2_t &gx)
{
    return vcombine_u8(phase_signed(in0.val[0], gx.val[0]),
                       phase_signed(in0.val[1], gx.val[1]));
}

template <>
inline uint8x16_t compute_phase<PhaseType::UNSIGNED>(const int16x8x2_t &in0, const int16x8x2_t &gx)
{
    return vcombine_u8(phase_unsigned(in0.val[0], gx.val[0]),
                       phase_unsigned(in0.val[1], gx.val[1]));
}
} // namespace fp16

template <MagnitudeType mag_type, PhaseType phase_type>
NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::NEMagnitudePhaseFP16Kernel()
    : _func(nullptr), _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr)
{
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::configure(const ITensor *gx, const ITensor *gy, ITensor *magnitude, ITensor *phase)
{
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(gx, Format::S16);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(gy, Format::S16);
    ARM_COMPUTE_ERROR_ON((nullptr == magnitude) && (nullptr == phase));

    const bool run_mag   = magnitude != nullptr;
    const bool run_phase = phase != nullptr;

    if(run_mag)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(magnitude, Format::S16);
    }

    if(run_phase)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(phase, Format::U8);
    }

    _gx        = gx;
    _gy        = gy;
    _magnitude = magnitude;
    _phase     = phase;

    if(run_mag && run_phase)
    {
        /* Run magnitude and phase */
        _func = &NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::magnitude_phase;
    }
    else if(run_mag)
    {
        /* Run magnitude */
        _func = &NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::magnitude;
    }
    else if(run_phase)
    {
        /* Run phase */
        _func = &NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::phase;
    }
    else
    {
        ARM_COMPUTE_ERROR("At least one output must be NOT NULL");
    }

    const unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window                 win = calculate_max_window(*gx->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal magnitude_access(magnitude == nullptr ? nullptr : magnitude->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal phase_access(phase == nullptr ? nullptr : phase->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(gx->info(), 0, num_elems_processed_per_iteration),
                              AccessWindowHorizontal(gy->info(), 0, num_elems_processed_per_iteration),
                              magnitude_access,
                              phase_access);

    ValidRegion valid_region = intersect_valid_regions(gx->info()->valid_region(),
                                                       gy->info()->valid_region());

    magnitude_access.set_valid_region(win, valid_region);
    phase_access.set_valid_region(win, valid_region);

    INEKernel::configure(win);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::magnitude(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator magnitude(_magnitude, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
        };

        const int16x8x2_t input2 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
        };

        // Compute and store magnitude
        const int16x8x2_t mag = fp16::compute_magnitude<mag_type>(input1, input2);

        /* Store magnitude */
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()), mag.val[0]);
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()) + 8, mag.val[1]);
    },
    gx, gy, magnitude);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::phase(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator phase(_phase, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
        };

        const int16x8x2_t input2 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
        };

        // Compute and store phase
        vst1q_u8(phase.ptr(), fp16::compute_phase<phase_type>(input1, input2));
    },
    gx, gy, phase);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::magnitude_phase(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator magnitude(_magnitude, window);
    Iterator phase(_phase, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
        };

        const int16x8x2_t input2 =
        {
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
            vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
        };

        // Compute and store magnitude
        const int16x8x2_t mag = fp16::compute_magnitude<mag_type>(input1, input2);

        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()), mag.val[0]);
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()) + 8, mag.val[1]);

        // Compute and store phase
        vst1q_u8(phase.ptr(), fp16::compute_phase<phase_type>(input1, input2));
    },
    gx, gy, magnitude, phase);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseFP16Kernel<mag_type, phase_type>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

template class arm_compute::NEMagnitudePhaseFP16Kernel<MagnitudeType::L1NORM, PhaseType::SIGNED>;
template class arm_compute::NEMagnitudePhaseFP16Kernel<MagnitudeType::L2NORM, PhaseType::SIGNED>;
template class arm_compute::NEMagnitudePhaseFP16Kernel<MagnitudeType::L1NORM, PhaseType::UNSIGNED>;
template class arm_compute::NEMagnitudePhaseFP16Kernel<MagnitudeType::L2NORM, PhaseType::UNSIGNED>;
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

namespace
{
inline float32x4_t inv(float32x4_t x)
{
    float32x4_t result = vrecpeq_f32(x);
    result             = vmulq_f32(vrecpsq_f32(x, result), result);
    return result;
}

inline float32x4_t atan2_0_360(float32x4_t gx, float32x4_t gy)
{
    const float32x4_t zero       = vdupq_n_f32(0.0f);
    const float32x4_t epsilon    = vdupq_n_f32(1e-9f);
    const float32x4_t piover4    = vdupq_n_f32(PI_4);
    const float32x4_t coeff1     = vdupq_n_f32(COEFF1);
    const float32x4_t coeff2     = vdupq_n_f32(COEFF2);
    const float32x4_t ninety     = vdupq_n_f32(90.0f * SCALE_FACTOR);
    const float32x4_t oneeighty  = vdupq_n_f32(180.0f * SCALE_FACTOR);
    const float32x4_t threesixty = vdupq_n_f32(360.0f * SCALE_FACTOR);
    const float32x4_t scale      = vdupq_n_f32(SCALE_360);

    float32x4_t abs_gx = vabsq_f32(gx);
    float32x4_t abs_gy = vabsq_f32(gy);
    float32x4_t tmin   = vminq_f32(abs_gx, abs_gy);
    float32x4_t tmax   = vmaxq_f32(abs_gx, abs_gy);
    float32x4_t z      = vmulq_f32(tmin, inv(vaddq_f32(tmax, epsilon)));
    float32x4_t absz   = vabsq_f32(z);
    float32x4_t term   = vmulq_f32(z, vsubq_f32(vdupq_n_f32(1.0f), absz));

    /* Compute y = pi/4 * x - x*(abs(x)-1)*(0.2447+0.0663 * abs(x) */
    float32x4_t result = vaddq_f32(coeff2, vmulq_f32(absz, coeff1));
    result             = vmulq_f32(result, term);
    result             = vmlaq_f32(result, piover4, z);

    /* Radians to degrees conversion with applied a scale factor in order to have the result [0, 255]  */
    result = vmulq_f32(result, scale);

    /* If z > 1, result = 90 - result */
    result = vbslq_f32(vcgeq_f32(abs_gx, abs_gy), result, vsubq_f32(ninety, result));

    /* Choose correct quadrant */
    result = vbslq_f32(vcltq_f32(gx, zero), vsubq_f32(oneeighty, result), result);
    result = vbslq_f32(vcltq_f32(gy, zero), vsubq_f32(threesixty, result), result);

    return result;
}

inline float32x4_t atan2_0_180(float32x4_t gx, float32x4_t gy)
{
    const float32x4_t zero       = vdupq_n_f32(0.0f);
    const float32x4_t epsilon    = vdupq_n_f32(1e-9f); // epsilon used to avoiding division by 0
    const float32x4_t piover4    = vdupq_n_f32(PI_4);
    const float32x4_t coeff1     = vdupq_n_f32(COEFF1);
    const float32x4_t coeff2     = vdupq_n_f32(COEFF2);
    const float32x4_t ninety     = vdupq_n_f32(90.0f);
    const float32x4_t oneeighty  = vdupq_n_f32(180.0f);
    const float32x4_t threesixty = vdupq_n_f32(360.0f);
    const float32x4_t scale      = vdupq_n_f32(SCALE_180);

    float32x4_t abs_gx = vabsq_f32(gx);
    float32x4_t abs_gy = vabsq_f32(gy);
    float32x4_t tmin   = vminq_f32(abs_gx, abs_gy);
    float32x4_t tmax   = vmaxq_f32(abs_gx, abs_gy);
    float32x4_t z      = vmulq_f32(tmin, inv(vaddq_f32(tmax, epsilon)));
    float32x4_t absz   = vabsq_f32(z);

    /* Compute y = pi/4 * z - z*(abs(z)-1)*(0.2447+0.0663 * abs(z) */
    float32x4_t term   = vmulq_f32(z, vsubq_f32(vdupq_n_f32(1.0f), absz));
    float32x4_t result = vaddq_f32(coeff2, vmulq_f32(absz, coeff1));
    result             = vmulq_f32(result, term);
    result             = vmlaq_f32(result, piover4, z);

    /* Radians to degrees conversion */
    result = vmulq_f32(result, scale);

    /* If z > 1, result = 90 - result */
    result = vbslq_f32(vcgeq_f32(abs_gx, abs_gy), result, vsubq_f32(ninety, result));

    /* Choose correct quadrant */
    result = vbslq_f32(vcltq_f32(gx, zero), vsubq_f32(oneeighty, result), result);
    result = vbslq_f32(vcltq_f32(gy, zero), vsubq_f32(threesixty, result), result);
    result = vbslq_f32(vcgtq_f32(result, oneeighty), vsubq_f32(result, oneeighty), result);

    return result;
}

inline float32x4_t invsqrtv(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);

    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                                sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal),
                                sqrt_reciprocal);

    return sqrt_reciprocal;
}

inline float32x4_t sqrtv(float32x4_t x)
{
    float32x4_t res = vdupq_n_f32(0.5f);
    return vmlaq_f32(res, x, invsqrtv(x));
}

inline int16x8_t magnitude_l2(int16x8_t input1, int16x8_t input2)
{
    const int32x4x2_t square_x =
    {
        {
            vmull_s16(vget_low_s16(input1), vget_low_s16(input1)),
            vmull_s16(vget_high_s16(input1), vget_high_s16(input1))
        }
    };

    const int32x4x2_t square_y =
    {
        {
            vmull_s16(vget_low_s16(input2), vget_low_s16(input2)),
            vmull_s16(vget_high_s16(input2), vget_high_s16(input2))
        }
    };

    const uint32x4x2_t sum =
    {
        {
            vaddq_u32(vreinterpretq_u32_s32(square_x.val[0]), vreinterpretq_u32_s32(square_y.val[0])),
            vaddq_u32(vreinterpretq_u32_s32(square_x.val[1]), vreinterpretq_u32_s32(square_y.val[1]))
        }
    };

    const float32x4x2_t res =
    {
        {
            sqrtv(vcvtq_f32_u32(sum.val[0])),
            sqrtv(vcvtq_f32_u32(sum.val[1]))
        }
    };

    return vcombine_s16(vqmovn_s32(vcvtq_s32_f32(res.val[0])),
                        vqmovn_s32(vcvtq_s32_f32(res.val[1])));
}

inline int16x8_t magnitude_l1(int16x8_t input1, int16x8_t input2)
{
    /* Saturating add */
    return vqaddq_s16(vqabsq_s16(input1), vqabsq_s16(input2));
}

inline uint8x8_t phase_signed(int16x8_t input1, int16x8_t input2)
{
    const float32x4_t zeropointfive = vdupq_n_f32(0.5f);

    float32x4_t inputx_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input1)));
    float32x4_t inputx_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input1)));
    float32x4_t inputy_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input2)));
    float32x4_t inputy_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input2)));

    /* Compute fast atan2 */
    float32x4_t angle_high = atan2_0_360(inputx_f32_high, inputy_f32_high);
    float32x4_t angle_low  = atan2_0_360(inputx_f32_low, inputy_f32_low);

    angle_high = vaddq_f32(angle_high, zeropointfive);
    angle_low  = vaddq_f32(angle_low, zeropointfive);

    return vmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(angle_low)),
                                  vqmovun_s32(vcvtq_s32_f32(angle_high))));
}

inline uint8x8_t phase_unsigned(int16x8_t input1, int16x8_t input2)
{
    const float32x4_t zeropointfive = vdupq_n_f32(0.5f);

    float32x4_t inputx_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input1)));
    float32x4_t inputx_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input1)));
    float32x4_t inputy_f32_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input2)));
    float32x4_t inputy_f32_low  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input2)));

    /* Compute fast atan2 */
    float32x4_t angle_high = atan2_0_180(inputx_f32_high, inputy_f32_high);
    float32x4_t angle_low  = atan2_0_180(inputx_f32_low, inputy_f32_low);

    angle_high = vaddq_f32(angle_high, zeropointfive);
    angle_low  = vaddq_f32(angle_low, zeropointfive);

    return vmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(angle_low)),
                                  vqmovun_s32(vcvtq_s32_f32(angle_high))));
}
} // namespace

template <MagnitudeType mag_type, PhaseType phase_type>
NEMagnitudePhaseKernel<mag_type, phase_type>::NEMagnitudePhaseKernel()
    : _func(nullptr), _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr)
{
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseKernel<mag_type, phase_type>::configure(const ITensor *gx, const ITensor *gy, ITensor *magnitude, ITensor *phase)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gx, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gy, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON((nullptr == magnitude) && (nullptr == phase));

    const bool run_mag   = magnitude != nullptr;
    const bool run_phase = phase != nullptr;

    if(run_mag)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::S16);
    }

    if(run_phase)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    }

    _gx        = gx;
    _gy        = gy;
    _magnitude = magnitude;
    _phase     = phase;

    if(run_mag && run_phase)
    {
        /* Run magnitude and phase */
        _func = &NEMagnitudePhaseKernel<mag_type, phase_type>::magnitude_phase;
    }
    else
    {
        if(run_mag)
        {
            /* Run magnitude */
            _func = &NEMagnitudePhaseKernel<mag_type, phase_type>::magnitude;
        }
        else if(run_phase)
        {
            /* Run phase */
            _func = &NEMagnitudePhaseKernel<mag_type, phase_type>::phase;
        }
        else
        {
            ARM_COMPUTE_ERROR("At least one output must be NOT NULL");
        }
    }

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window                 win = calculate_max_window(*gx->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal magnitude_access(magnitude == nullptr ? nullptr : magnitude->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal phase_access(phase == nullptr ? nullptr : phase->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(gx->info(), 0, num_elems_processed_per_iteration),
                              AccessWindowHorizontal(gy->info(), 0, num_elems_processed_per_iteration),
                              magnitude_access,
                              phase_access);

    ValidRegion valid_region = intersect_valid_regions(gx->info()->valid_region(),
                                                       gy->info()->valid_region());

    magnitude_access.set_valid_region(win, valid_region);
    phase_access.set_valid_region(win, valid_region);

    INEKernel::configure(win);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseKernel<mag_type, phase_type>::magnitude(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator magnitude(_magnitude, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
            }
        };

        const int16x8x2_t input2 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
            }
        };

        /* Compute magnitude */
        int16x8x2_t mag{ {} };

        if(MagnitudeType::L2NORM == mag_type)
        {
            mag.val[0] = magnitude_l2(input1.val[0], input2.val[0]);
            mag.val[1] = magnitude_l2(input1.val[1], input2.val[1]);
        }
        else
        {
            mag.val[0] = magnitude_l1(input1.val[0], input2.val[0]);
            mag.val[1] = magnitude_l1(input1.val[1], input2.val[1]);
        }

        /* Store magnitude */
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()), mag.val[0]);
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()) + 8, mag.val[1]);
    },
    gx, gy, magnitude);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseKernel<mag_type, phase_type>::phase(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator phase(_phase, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
            }
        };

        const int16x8x2_t input2 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
            }
        };

        /* Compute phase */
        uint8x8x2_t vphase{ {} };

        if(PhaseType::SIGNED == phase_type)
        {
            vphase.val[0] = phase_signed(input1.val[0], input2.val[0]);
            vphase.val[1] = phase_signed(input1.val[1], input2.val[1]);
        }
        else
        {
            vphase.val[0] = phase_unsigned(input1.val[0], input2.val[0]);
            vphase.val[1] = phase_unsigned(input1.val[1], input2.val[1]);
        }

        /* Store phase */
        vst1q_u8(phase.ptr(), vcombine_u8(vphase.val[0], vphase.val[1]));
    },
    gx, gy, phase);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseKernel<mag_type, phase_type>::magnitude_phase(const Window &window)
{
    Iterator gx(_gx, window);
    Iterator gy(_gy, window);
    Iterator magnitude(_magnitude, window);
    Iterator phase(_phase, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int16x8x2_t input1 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gx.ptr()) + 8)
            }
        };

        const int16x8x2_t input2 =
        {
            {
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr())),
                vld1q_s16(reinterpret_cast<int16_t *>(gy.ptr()) + 8)
            }
        };

        /* Compute magnitude */
        int16x8x2_t mag{ {} };

        if(MagnitudeType::L2NORM == mag_type)
        {
            mag.val[0] = magnitude_l2(input1.val[0], input2.val[0]);
            mag.val[1] = magnitude_l2(input1.val[1], input2.val[1]);
        }
        else
        {
            mag.val[0] = magnitude_l1(input1.val[0], input2.val[0]);
            mag.val[1] = magnitude_l1(input1.val[1], input2.val[1]);
        }

        /* Store magnitude */
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()), mag.val[0]);
        vst1q_s16(reinterpret_cast<int16_t *>(magnitude.ptr()) + 8, mag.val[1]);

        /* Compute phase */
        uint8x8x2_t vphase{ {} };

        if(PhaseType::SIGNED == phase_type)
        {
            vphase.val[0] = phase_signed(input1.val[0], input2.val[0]);
            vphase.val[1] = phase_signed(input1.val[1], input2.val[1]);
        }
        else
        {
            vphase.val[0] = phase_unsigned(input1.val[0], input2.val[0]);
            vphase.val[1] = phase_unsigned(input1.val[1], input2.val[1]);
        }

        /* Store phase */
        vst1q_u8(phase.ptr(), vcombine_u8(vphase.val[0], vphase.val[1]));
    },
    gx, gy, magnitude, phase);
}

template <MagnitudeType mag_type, PhaseType phase_type>
void NEMagnitudePhaseKernel<mag_type, phase_type>::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

template class arm_compute::NEMagnitudePhaseKernel<MagnitudeType::L1NORM, PhaseType::SIGNED>;
template class arm_compute::NEMagnitudePhaseKernel<MagnitudeType::L2NORM, PhaseType::SIGNED>;
template class arm_compute::NEMagnitudePhaseKernel<MagnitudeType::L1NORM, PhaseType::UNSIGNED>;
template class arm_compute::NEMagnitudePhaseKernel<MagnitudeType::L2NORM, PhaseType::UNSIGNED>;
