/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NESYMM_H__
#define __ARM_COMPUTE_NESYMM_H__

#include "arm_compute/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
using qsymm8_t  = int8_t;  /**< 8 bit quantized symmetric scalar value */
using qsymm16_t = int16_t; /**< 16 bit quantized symmetric scalar value */

using qsymm16x8_t   = int16x8_t;   /**< 16 bit quantized symmetric vector with 8 elements */
using qsymm16x8x2_t = int16x8x2_t; /**< 16 bit quantized symmetric vector with 16 elements */

/** Performs final quantization step on 8 signed 16-bit elements
 *
 * @tparam is_bounded_relu Specified if a fused bounded relu should be applied
 *
 * @param[in] in_s32                       Input to be quantized.
 * @param[in] result_fixedpoint_multiplier Result multiplier parameter
 * @param[in] result_shift                 Result shift parameter
 * @param[in] min_s16                      Relu lower bound
 * @param[in] max_s16                      Relu upper bound
 *
 * @return Quantized values
 */
template <bool is_bounded_relu>
int16x8_t finalize_quantization_int16(int32x4x2_t &in_s32,
                                      int          result_fixedpoint_multiplier,
                                      int32_t      result_shift,
                                      int16x8_t    min_s16,
                                      int16x8_t    max_s16)
{
    // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
    in_s32.val[0] = vqrdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
    in_s32.val[1] = vqrdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);

    // Round to the nearest division by a power-of-two using result_shift_s32
    in_s32.val[0] = rounding_divide_by_pow2(in_s32.val[0], result_shift);
    in_s32.val[1] = rounding_divide_by_pow2(in_s32.val[1], result_shift);

    // Convert S32 to S16
    int16x8_t out_s16 = vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1]));

    if(is_bounded_relu)
    {
        out_s16 = vmaxq_s16(out_s16, min_s16);
        out_s16 = vminq_s16(out_s16, max_s16);
    }

    return out_s16;
}

/** Performs final quantization step on single signed 16-bit element
 *
 * @tparam is_bounded_relu Specified if a fused bounded relu should be applied
 *
 * @param[in] in_value                     Input to be quantized.
 * @param[in] result_fixedpoint_multiplier Result multiplier parameter
 * @param[in] result_shift                 Result shift parameter
 * @param[in] min_s16                      Relu lower bound
 * @param[in] max_s16                      Relu upper bound
 *
 * @return Quantized values
 */
template <bool is_bounded_relu>
inline int16_t finalize_quantization_int16(int32_t in_value, int result_fixedpoint_multiplier,
                                           int32_t result_shift, int16_t min_s16, int16_t max_s16)
{
    int32x4_t in_s32 = vdupq_n_s32(in_value);

    // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
    in_value = vgetq_lane_s32(vqrdmulhq_n_s32(in_s32, result_fixedpoint_multiplier), 0);

    // Shift value by result_shift_s32
    in_value = rounding_divide_by_pow2(in_value, result_shift);

    // Bound the result
    int16_t out_s16 = static_cast<int16_t>(std::max<int32_t>(-32768, std::min<int32_t>(32767, in_value)));

    if(is_bounded_relu)
    {
        out_s16 = static_cast<int16_t>(std::max(min_s16, std::min(max_s16, out_s16)));
    }

    return out_s16;
}

/** Dequantize a neon vector holding 8 16-bit quantized values.
 *
 * @param[in] qv    Input values to be dequantized.
 * @param[in] scale Quantization scale
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x2_t vdequantize_int16(const int16x8_t &qv, float scale)
{
    const float32x4_t   vscale = vdupq_n_f32(scale);
    const float32x4x2_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qv))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qv))), vscale)
        }
    };
    return vdequantized_input;
}

/** Quantize a neon vector holding 8 floating point values.
 *
 * @param[in] qv    Input values to be quantized.
 * @param[in] scale Quantization scale
 *
 * @return A neon vector holding the quantized values
 */
inline int16x8_t vquantize_int16(const float32x4x2_t &qv, float scale)
{
    const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);

    const int32x4x2_t rf =
    {
        {
#ifdef __aarch64__
            vcvtnq_s32_f32(vmulq_f32(qv.val[0], vinvscale)),
            vcvtnq_s32_f32(vmulq_f32(qv.val[1], vinvscale))
#else  //__aarch64__
            vcvtq_s32_f32(vmulq_f32(qv.val[0], vinvscale)),
            vcvtq_s32_f32(vmulq_f32(qv.val[1], vinvscale))
#endif //__aarch64__
        }
    };
    return vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1]));
}

/** Dequantize a neon vector holding 16 16-bit quantized values.
 *
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x4_t vdequantize(const int16x8x2_t &qv, const UniformQuantizationInfo &qi)
{
    const float         scale  = qi.scale;
    const float32x4_t   vscale = vdupq_n_f32(scale);
    const float32x4x4_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qv.val[0]))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qv.val[0]))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qv.val[1]))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qv.val[1]))), vscale),
        }
    };
    return vdequantized_input;
}

/** Quantize a neon vector holding 16 floating point values.
 *
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return A neon vector holding the quantized values
 */
inline qsymm16x8x2_t vquantize_qsymm16(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    const float scale = qi.scale;
    ARM_COMPUTE_ERROR_ON(scale == 0.f);
    const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
    const int32x4x4_t rf =
    {
        {
#ifdef __aarch64__
            vcvtnq_s32_f32(vmulq_f32(qv.val[0], vinvscale)),
            vcvtnq_s32_f32(vmulq_f32(qv.val[1], vinvscale)),
            vcvtnq_s32_f32(vmulq_f32(qv.val[2], vinvscale)),
            vcvtnq_s32_f32(vmulq_f32(qv.val[3], vinvscale)),
#else  //__aarch64__
            vcvtq_s32_f32(vmulq_f32(qv.val[0], vinvscale)),
            vcvtq_s32_f32(vmulq_f32(qv.val[1], vinvscale)),
            vcvtq_s32_f32(vmulq_f32(qv.val[2], vinvscale)),
            vcvtq_s32_f32(vmulq_f32(qv.val[3], vinvscale)),
#endif //__aarch64__
        }
    };
    const qsymm16x8x2_t res =
    {
        vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])),
        vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])),
    };

    return res;
}

} // namespace arm_compute
#endif // __ARM_COMPUTE_NESYMM_H__
