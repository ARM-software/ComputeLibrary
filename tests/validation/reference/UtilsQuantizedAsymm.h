/*
 * Copyright (c) 2017-2021, 2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_REFERENCE_UTILSQUANTIZEDASYMM_H
#define ACL_TESTS_VALIDATION_REFERENCE_UTILSQUANTIZEDASYMM_H

#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
#if __clang__
// This has been tested on clang 7.0.2 (__clang_major__ == 7 && __clang_minor__ == 0 && __clang_patchlevel__ == 2)
inline int64_t to_int64(int32_t val)
{
    return static_cast<int64_t>(val) | ((val < 0) ? (((1ll << 32) - 1) << 32) : 0);
}
#else  // __clang__
inline int64_t to_int64(int32_t val)
{
    return static_cast<int64_t>(val);
}
#endif // __clang__
} // namespace

/** Rounded to nearest division by a power-of-two.
 * This implements the documented behaviour of SRSHL with a negative shift. */
inline int32_t asymm_rounding_divide_by_pow2(int32_t x, int exponent)
{
    return (exponent == 0) ? x : ((x + (1 << (exponent-1))) >> exponent);
}

/** Doubling multiplication of two integers, returning high half.
 * This implements the documented behaviour of SQDMULH */
inline int32_t asymm_int_mult(int32_t a, int32_t b)
{
    const bool    overflow     = a == b && a == std::numeric_limits<int32_t>::min();
    const int64_t a_64         = to_int64(a);
    const int64_t b_64         = to_int64(b);
    const int64_t ab_x2_64     = a_64 * b_64 * 2;
    return overflow ? std::numeric_limits<int32_t>::max() : (ab_x2_64 >> 32);
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
#endif // ACL_TESTS_VALIDATION_REFERENCE_UTILSQUANTIZEDASYMM_H
