/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_HELPERS_ASYMM_H
#define ARM_COMPUTE_HELPERS_ASYMM_H

#include "helpers.h"

/** Correctly-rounded-to-nearest division by a power-of-two.
 *
 * @param[in] size Size of vector.
 *
 * @return Correctly-rounded-to-nearest division by a power-of-two.
 */
#define ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(size)                                                                   \
    inline VEC_DATA_TYPE(int, size) asymm_rounding_divide_by_POW2_##size(VEC_DATA_TYPE(int, size) x, int exponent) \
    {                                                                                                              \
        VEC_DATA_TYPE(int, size)                                                                                   \
        mask = (1 << exponent) - 1;                                                                                \
        const VEC_DATA_TYPE(int, size) zero = 0;                                                                   \
        const VEC_DATA_TYPE(int, size) one  = 1;                                                                   \
        VEC_DATA_TYPE(int, size)                                                                                   \
        threshold = (mask >> 1) + select(zero, one, x < 0);                                                        \
        return (x >> exponent) + select(zero, one, (x & mask) > threshold);                                        \
    }

ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(2)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(8)
ASYMM_ROUNDING_DIVIDE_BY_POW2_IMPL(16)

#define ASYMM_ROUNDING_DIVIDE_BY_POW2(x, exponent, size) asymm_rounding_divide_by_POW2_##size(x, exponent)

/** Product of two numbers, interpreting them as fixed-point values in the interval [-1, 1),
 * rounding to the nearest value, and saturating -1 * -1 to the maximum value.
 *
 * @param[in] size Size of vector.
 *
 * @return Product of two fixed-point numbers.
 */
#define ASYMM_MULT_IMP(size)                                                                                 \
    inline VEC_DATA_TYPE(int, size) asymm_mult##size(VEC_DATA_TYPE(int, size) a, VEC_DATA_TYPE(int, size) b) \
    {                                                                                                        \
        VEC_DATA_TYPE(int, size)                                                                             \
        overflow = a == b && a == INT_MIN;                                                                   \
        VEC_DATA_TYPE(long, size)                                                                            \
        a_64 = convert_long##size(a);                                                                        \
        VEC_DATA_TYPE(long, size)                                                                            \
        b_64 = convert_long##size(b);                                                                        \
        VEC_DATA_TYPE(long, size)                                                                            \
        ab_64 = a_64 * b_64;                                                                                 \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask1 = 1 << 30;                                                                                     \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask2 = 1 - (1 << 30);                                                                               \
        VEC_DATA_TYPE(long, size)                                                                            \
        nudge = select(mask2, mask1, ab_64 >= 0);                                                            \
        VEC_DATA_TYPE(long, size)                                                                            \
        mask = 1ll << 31;                                                                                    \
        VEC_DATA_TYPE(int, size)                                                                             \
        ab_x2_high32 = convert_int##size((ab_64 + nudge) / mask);                                            \
        return select(ab_x2_high32, INT_MAX, overflow);                                                      \
    }

ASYMM_MULT_IMP(2)
ASYMM_MULT_IMP(8)
ASYMM_MULT_IMP(16)

#define ASYMM_MULT(a, b, size) asymm_mult##size(a, b)

#define ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(x, quantized_multiplier, right_shift, size) \
    ASYMM_ROUNDING_DIVIDE_BY_POW2(ASYMM_MULT(x, quantized_multiplier, size), right_shift, size)

#endif // ARM_COMPUTE_HELPERS_ASYMM_H
