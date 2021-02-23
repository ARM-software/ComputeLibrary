/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_VALIDATION_UTILS_QUANTIZED_ASYMM_H
#define ARM_COMPUTE_TEST_VALIDATION_UTILS_QUANTIZED_ASYMM_H

#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Rounded to nearest division by a power-of-two. */
inline int32_t asymm_rounding_divide_by_pow2(int32_t x, int exponent)
{
    const int32_t mask      = (1 << exponent) - 1;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + ((x & mask) > threshold ? 1 : 0);
}

/** Multiplication of two integers. The same as ARMv7 Neon VQRDMULH instruction. */
inline int32_t asymm_int_mult(int32_t a, int32_t b)
{
    bool    overflow = a == b && a == std::numeric_limits<int32_t>::min();
    int64_t a_64(a);
    int64_t b_64(b);
    int64_t ab_64        = a_64 * b_64;
    int32_t nudge        = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

/** Quantize down the input value in range [min, max]. */
inline int32_t quantize_down_scale_by_fixedpoint(int32_t val, int32_t result_mult_int, int32_t result_shift,
                                                 int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    int32_t res = 0;
    if(result_shift < 0)
    {
        res = asymm_int_mult(val * (1 << (-result_shift)), result_mult_int);
    }
    else
    {
        res = asymm_rounding_divide_by_pow2(asymm_int_mult(val, result_mult_int), result_shift);
    }
    res += result_offset_after_shift;
    res = utility::clamp<int32_t>(res, min, max);
    return res;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_VALIDATION_UTILS_QUANTIZED_ASYMM_H */
