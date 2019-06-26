/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEASYMM_H__
#define __ARM_COMPUTE_NEASYMM_H__

#include "arm_compute/core/NEON/NEMath.h"
#include <arm_neon.h>

namespace arm_compute
{
using qasymm8x8_t   = uint8x8_t;   /**< 8 bit quantized asymmetric vector with 8 elements */
using qasymm8x8x2_t = uint8x8x2_t; /**< 8 bit quantized asymmetric vector with 16 elements */
using qasymm8x8x3_t = uint8x8x3_t; /**< 8 bit quantized asymmetric vector with 24 elements */
using qasymm8x8x4_t = uint8x8x4_t; /**< 8 bit quantized asymmetric vector with 32 elements */
using qasymm8x16_t  = uint8x16_t;  /**< 8 bit quantized asymmetric vector with 16 elements */

/** Perform a multiply-accumulate on all 16 components of a QASYMM8 vector
 *
 * vd*vs + vo
 *
 * @param[in] vd Input vector value in QASYMM8 format
 * @param[in] vs Vector multiplier in F32 format. The multiplier value must be duplicated across all four lanes.
 * @param[in] vo Vector addend in F32 format. The addend value must be duplicated across all four lanes.
 *
 * @return A 16-component vector in QASYMM8 format, saturated to fit
 */
uint8x16_t vmlaq_qasymm8(qasymm8x16_t vd, float32x4_t vs, float32x4_t vo);

/** Performs final quantization step on 16 elements
 *
 * @tparam is_bounded_relu Specified if a fused bounded relu should be applied
 *
 * @param in_s32                        Input to be quantized.
 * @param result_fixedpoint_multiplier  Result multiplier parameter
 * @param result_shift                  Result shift parameter
 * @param result_offset_after_shift_s32 Result offset parameter
 * @param min_u8                        Relu lower bound
 * @param max_u8                        Relu upper bound
 *
 * @return Quantized values
 */
template <bool is_bounded_relu>
uint8x16_t finalize_quantization(int32x4x4_t &in_s32,
                                 int          result_fixedpoint_multiplier,
                                 int32_t      result_shift,
                                 int32x4_t    result_offset_after_shift_s32,
                                 uint8x16_t   min_u8,
                                 uint8x16_t   max_u8)
{
    const static int32x4_t zero_s32 = vdupq_n_s32(0);

    // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
    in_s32.val[0] = vqrdmulhq_n_s32(in_s32.val[0], result_fixedpoint_multiplier);
    in_s32.val[1] = vqrdmulhq_n_s32(in_s32.val[1], result_fixedpoint_multiplier);
    in_s32.val[2] = vqrdmulhq_n_s32(in_s32.val[2], result_fixedpoint_multiplier);
    in_s32.val[3] = vqrdmulhq_n_s32(in_s32.val[3], result_fixedpoint_multiplier);

    // Round to the nearest division by a power-of-two using result_shift_s32
    in_s32.val[0] = rounding_divide_by_pow2(in_s32.val[0], result_shift);
    in_s32.val[1] = rounding_divide_by_pow2(in_s32.val[1], result_shift);
    in_s32.val[2] = rounding_divide_by_pow2(in_s32.val[2], result_shift);
    in_s32.val[3] = rounding_divide_by_pow2(in_s32.val[3], result_shift);

    // Add the offset terms
    in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_after_shift_s32);
    in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_after_shift_s32);
    in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_after_shift_s32);
    in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_after_shift_s32);

    // Saturate negative values
    in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
    in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
    in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
    in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

    // Convert S32 to S16
    const int16x8x2_t in_s16 =
    {
        {
            vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
            vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
        }
    };

    // Convert S16 to U8
    uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

    if(is_bounded_relu)
    {
        out_u8 = vmaxq_u8(out_u8, min_u8);
        out_u8 = vminq_u8(out_u8, max_u8);
    }

    return out_u8;
}

/** Performs final quantization step on single element
 *
 * @tparam is_bounded_relu Specified if a fused bounded relu should be applied
 *
 * @param[in] in_value                      Input to be quantized.
 * @param[in] result_fixedpoint_multiplier  Result multiplier parameter
 * @param[in] result_shift                  Result shift parameter
 * @param[in] result_offset_after_shift_s32 Result offset parameter
 * @param[in] min_u8                        Relu lower bound
 * @param[in] max_u8                        Relu upper bound
 *
 * @return Quantized value
 */
template <bool is_bounded_relu>
inline uint8_t finalize_quantization(int32_t in_value, int result_fixedpoint_multiplier,
                                     int32_t result_shift, int32_t result_offset_after_shift_s32,
                                     uint8_t min_u8, uint8_t max_u8)
{
    int32x4_t in_s32 = vdupq_n_s32(in_value);

    // Fixed point multiplication with vector saturating rounding doubling multiply high with scalar
    in_value = vgetq_lane_s32(vqrdmulhq_n_s32(in_s32, result_fixedpoint_multiplier), 0);

    // Shift value by result_shift_s32
    in_value = rounding_divide_by_pow2(in_value, result_shift);

    // Add the offset term
    in_value += result_offset_after_shift_s32;

    // Bound the result
    uint8_t out_u8 = static_cast<uint8_t>(std::max<int32_t>(0, std::min<int32_t>(255, in_value)));
    if(is_bounded_relu)
    {
        out_u8 = static_cast<uint8_t>(std::max(min_u8, std::min(max_u8, out_u8)));
    }

    return out_u8;
}

/** Dequantize a neon vector holding 8 quantized values.
 *
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x2_t vdequantize(const uint8x8_t &qv, const UniformQuantizationInfo &qi)
{
    const float         scale   = qi.scale;
    const int           offset  = qi.offset;
    const int32x4_t     voffset = vdupq_n_s32(offset);
    const float32x4_t   vscale  = vdupq_n_f32(scale);
    const float32x4x2_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(qv)))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(qv)))), voffset)), vscale),
        }
    };
    return vdequantized_input;
}

/** Dequantize a neon vector holding 16 quantized values.
 *
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x4_t vdequantize(const uint8x16_t &qv, const UniformQuantizationInfo &qi)
{
    const float         scale   = qi.scale;
    const int           offset  = qi.offset;
    const int32x4_t     voffset = vdupq_n_s32(offset);
    const float32x4_t   vscale  = vdupq_n_f32(scale);
    const float32x4x4_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(qv))))), voffset)), vscale),
        }
    };
    return vdequantized_input;
}

/** Dequantize following an asymmetric quantization scheme a neon vector holding 16 quantized values.
 *
 * @param[in] qv     Input values to be dequantized.
 * @param[in] scale  Quantization scaling factor.
 * @param[in] offset Zero quantization offset.
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x4_t vdequantize(const uint8x16_t &qv, float scale, int32_t offset)
{
    const int32x4_t     voffset = vdupq_n_s32(offset);
    const float32x4_t   vscale  = vdupq_n_f32(scale);
    const float32x4x4_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(qv))))), voffset)), vscale),
            vmulq_f32(vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(qv))))), voffset)), vscale),
        }
    };
    return vdequantized_input;
}

/** Dequantize following a symmetric quantization scheme a neon vector holding 16 quantized values.
 *
 * @param[in] qv    Input values to be dequantized.
 * @param[in] scale Quantization scaling factor.
 *
 * @return Dequantized values in a neon vector
 */
inline float32x4x4_t vdequantize(const int8x16_t &qv, float scale)
{
    const float32x4_t   vscale = vdupq_n_f32(scale);
    const float32x4x4_t vdequantized_input =
    {
        {
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(qv))))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(qv))))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(qv))))), vscale),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(qv))))), vscale),
        }
    };
    return vdequantized_input;
}

/** Quantize a neon vector holding 8 floating point values.
 *
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return A neon vector holding the quantized values
 */
inline uint8x8_t vquantize(const float32x4x2_t &qv, const UniformQuantizationInfo &qi)
{
    const float       scale     = qi.scale;
    const int         offset    = qi.offset;
    const float32x4_t voffset   = vdupq_n_f32(offset);
    const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
    const int32x4x4_t rf =
    {
        {
#ifdef __aarch64__
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#else  //__aarch64__
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
#endif //__aarch64__
        }
    };
    return vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
}

/** Quantize a neon vector holding 16 floating point values.
 *
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return A neon vector holding the quantized values
 */
inline uint8x16_t vquantize(const float32x4x4_t &qv, const UniformQuantizationInfo &qi)
{
    const float       scale     = qi.scale;
    const int         offset    = qi.offset;
    const float32x4_t voffset   = vdupq_n_f32(offset);
    const float32x4_t vinvscale = vdupq_n_f32(1.f / scale);
    const int32x4x4_t rf =
    {
        {
#ifdef __aarch64__
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[2], vinvscale)),
            vcvtnq_s32_f32(vmlaq_f32(voffset, qv.val[3], vinvscale)),
#else  //__aarch64__
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[0], vinvscale)),
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[1], vinvscale)),
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[2], vinvscale)),
            vcvtq_s32_f32(vmlaq_f32(voffset, qv.val[3], vinvscale)),
#endif //__aarch64__
        }
    };
    const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1])));
    const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf.val[2]), vqmovn_s32(rf.val[3])));
    return vcombine_u8(pa, pb);
}
} // namespace arm_compute
#include "arm_compute/core/NEON/NEAsymm.inl"
#endif // __ARM_COMPUTE_NEASYMM_H__
