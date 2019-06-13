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

#include "NEAsymm.h"
#include <arm_neon.h>

namespace arm_compute
{
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
} // namespace arm_compute
#endif // __ARM_COMPUTE_NESYMM_H__
