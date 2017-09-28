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
#include "arm_compute/core/Error.h"

#include <cmath>
#include <limits>

namespace
{
template <typename TpIn, typename TpSat>
inline TpSat saturate_convert(TpIn a)
{
    if(a > std::numeric_limits<TpSat>::max())
    {
        a = std::numeric_limits<TpSat>::max();
    }
    if(a < std::numeric_limits<TpSat>::min())
    {
        a = std::numeric_limits<TpSat>::min();
    }
    return static_cast<TpSat>(a);
}
} // namespace

namespace arm_compute
{
inline qint8_t sqshl_qs8(qint8_t a, int shift)
{
    qint16_t tmp = static_cast<qint16_t>(a) << shift;

    // Saturate the result in case of overflow and cast to qint8_t
    return saturate_convert<qint16_t, qint8_t>(tmp);
}

inline qint16_t sqshl_qs16(qint16_t a, int shift)
{
    qint32_t tmp = static_cast<qint32_t>(a) << shift;

    // Saturate the result in case of overflow and cast to qint16_t
    return saturate_convert<qint32_t, qint16_t>(tmp);
}

inline qint8_t sshr_qs8(qint8_t a, int shift)
{
    ARM_COMPUTE_ERROR_ON_MSG(shift == 0, "Shift should not be zero");
    const qint8_t round_val = 1 << (shift - 1);
    return sqadd_qs8(a, round_val) >> shift;
}

inline qint16_t sshr_qs16(qint16_t a, int shift)
{
    ARM_COMPUTE_ERROR_ON_MSG(shift == 0, "Shift should not be zero");
    const qint16_t round_val = 1 << (shift - 1);
    return sqadd_qs16(a, round_val) >> shift;
}

inline qint8_t sabs_qs8(qint8_t a)
{
    return (a < 0) ? (a == std::numeric_limits<int8_t>::min()) ? std::numeric_limits<int8_t>::max() : -a : a;
}

inline qint16_t sabs_qs16(qint16_t a)
{
    return (a < 0) ? (a == std::numeric_limits<int16_t>::min()) ? std::numeric_limits<int16_t>::max() : -a : a;
}

inline qint8_t sadd_qs8(qint8_t a, qint8_t b)
{
    return a + b;
}

inline qint16_t sadd_qs16(qint16_t a, qint16_t b)
{
    return a + b;
}

inline qint8_t sqadd_qs8(qint8_t a, qint8_t b)
{
    // We need to store the temporary result in qint16_t otherwise we cannot evaluate the overflow
    qint16_t tmp = (static_cast<qint16_t>(a) + static_cast<qint16_t>(b));

    // Saturate the result in case of overflow and cast to qint8_t
    return saturate_convert<qint16_t, qint8_t>(tmp);
}

inline qint16_t sqadd_qs16(qint16_t a, qint16_t b)
{
    // We need to store the temporary result in qint32_t otherwise we cannot evaluate the overflow
    qint32_t tmp = (static_cast<qint32_t>(a) + static_cast<qint32_t>(b));

    // Saturate the result in case of overflow and cast to qint16_t
    return saturate_convert<qint32_t, qint16_t>(tmp);
}

inline qint32_t sqadd_qs32(qint32_t a, qint32_t b)
{
    // We need to store the temporary result in qint64_t otherwise we cannot evaluate the overflow
    qint64_t tmp = (static_cast<qint64_t>(a) + static_cast<qint64_t>(b));

    // Saturate the result in case of overflow and cast to qint32_t
    return saturate_convert<qint64_t, qint32_t>(tmp);
}

inline qint8_t ssub_qs8(qint8_t a, qint8_t b)
{
    return a - b;
}

inline qint16_t ssub_qs16(qint16_t a, qint16_t b)
{
    return a - b;
}

inline qint8_t sqsub_qs8(qint8_t a, qint8_t b)
{
    // We need to store the temporary result in uint16_t otherwise we cannot evaluate the overflow
    qint16_t tmp = static_cast<qint16_t>(a) - static_cast<qint16_t>(b);

    // Saturate the result in case of overflow and cast to qint8_t
    return saturate_convert<qint16_t, qint8_t>(tmp);
}

inline qint16_t sqsub_qs16(qint16_t a, qint16_t b)
{
    // We need to store the temporary result in qint32_t otherwise we cannot evaluate the overflow
    qint32_t tmp = static_cast<qint32_t>(a) - static_cast<qint32_t>(b);

    // Saturate the result in case of overflow and cast to qint16_t
    return saturate_convert<qint32_t, qint16_t>(tmp);
}

inline qint8_t smul_qs8(qint8_t a, qint8_t b, int fixed_point_position)
{
    const qint16_t round_up_const = (1 << (fixed_point_position - 1));

    qint16_t tmp = static_cast<qint16_t>(a) * static_cast<qint16_t>(b);

    // Rounding up
    tmp += round_up_const;

    return static_cast<qint8_t>(tmp >> fixed_point_position);
}

inline qint16_t smul_qs16(qint16_t a, qint16_t b, int fixed_point_position)
{
    const qint32_t round_up_const = (1 << (fixed_point_position - 1));

    qint32_t tmp = static_cast<qint32_t>(a) * static_cast<qint32_t>(b);

    // Rounding up
    tmp += round_up_const;

    return static_cast<qint16_t>(tmp >> fixed_point_position);
}

inline qint8_t sqmul_qs8(qint8_t a, qint8_t b, int fixed_point_position)
{
    const qint16_t round_up_const = (1 << (fixed_point_position - 1));

    qint16_t tmp = static_cast<qint16_t>(a) * static_cast<qint16_t>(b);

    // Rounding up
    tmp += round_up_const;

    return saturate_convert<qint16_t, qint8_t>(tmp >> fixed_point_position);
}

inline qint16_t sqmul_qs16(qint16_t a, qint16_t b, int fixed_point_position)
{
    const qint32_t round_up_const = (1 << (fixed_point_position - 1));

    qint32_t tmp = static_cast<qint32_t>(a) * static_cast<qint32_t>(b);

    // Rounding up
    tmp += round_up_const;

    return saturate_convert<qint32_t, qint16_t>(tmp >> fixed_point_position);
}

inline qint16_t sqmull_qs8(qint8_t a, qint8_t b, int fixed_point_position)
{
    const qint16_t round_up_const = (1 << (fixed_point_position - 1));

    qint16_t tmp = static_cast<qint16_t>(a) * static_cast<qint16_t>(b);

    // Rounding up
    tmp += round_up_const;

    return tmp >> fixed_point_position;
}

inline qint32_t sqmull_qs16(qint16_t a, qint16_t b, int fixed_point_position)
{
    const qint32_t round_up_const = (1 << (fixed_point_position - 1));

    qint32_t tmp = static_cast<qint32_t>(a) * static_cast<qint32_t>(b);

    // Rounding up
    tmp += round_up_const;

    return tmp >> fixed_point_position;
}

inline qint8_t sinvsqrt_qs8(qint8_t a, int fixed_point_position)
{
    const qint8_t shift = 8 - (fixed_point_position + (__builtin_clz(a) - 24));

    const qint8_t const_three = (3 << fixed_point_position);
    qint8_t       temp        = shift < 0 ? (a << -shift) : (a >> shift);
    qint8_t       x2          = temp;

    // We need three iterations to find the result
    for(int i = 0; i < 3; ++i)
    {
        qint8_t three_minus_dx = ssub_qs8(const_three, smul_qs8(temp, smul_qs8(x2, x2, fixed_point_position), fixed_point_position));
        x2                     = (smul_qs8(x2, three_minus_dx, fixed_point_position) >> 1);
    }

    temp = shift < 0 ? (x2 << (-shift >> 1)) : (x2 >> (shift >> 1));

    return temp;
}

inline qint16_t sinvsqrt_qs16(qint16_t a, int fixed_point_position)
{
    const qint16_t shift = 16 - (fixed_point_position + (__builtin_clz(a) - 16));

    const qint16_t const_three = (3 << fixed_point_position);
    qint16_t       temp        = shift < 0 ? (a << -shift) : (a >> shift);
    qint16_t       x2          = temp;

    // We need three iterations to find the result
    for(int i = 0; i < 3; ++i)
    {
        qint16_t three_minus_dx = ssub_qs16(const_three, smul_qs16(temp, smul_qs16(x2, x2, fixed_point_position), fixed_point_position));
        x2                      = smul_qs16(x2, three_minus_dx, fixed_point_position) >> 1;
    }

    temp = shift < 0 ? (x2 << ((-shift) >> 1)) : (x2 >> (shift >> 1));

    return temp;
}

inline qint8_t sdiv_qs8(qint8_t a, qint8_t b, int fixed_point_position)
{
    const qint16_t temp = a << fixed_point_position;
    return static_cast<qint8_t>(temp / b);
}

inline qint16_t sdiv_qs16(qint16_t a, qint16_t b, int fixed_point_position)
{
    const qint32_t temp = a << fixed_point_position;
    return static_cast<qint16_t>(temp / b);
}

inline qint8_t sqexp_qs8(qint8_t a, int fixed_point_position)
{
    // Constants
    const qint8_t const_one = (1 << fixed_point_position);
    const qint8_t ln2       = ((0x58 >> (6 - fixed_point_position)) + 1) >> 1;
    const qint8_t inv_ln2   = (((0x38 >> (6 - fixed_point_position)) + 1) >> 1) | const_one;
    const qint8_t A         = ((0x7F >> (6 - fixed_point_position)) + 1) >> 1;
    const qint8_t B         = ((0x3F >> (6 - fixed_point_position)) + 1) >> 1;
    const qint8_t C         = ((0x16 >> (6 - fixed_point_position)) + 1) >> 1;
    const qint8_t D         = ((0x05 >> (6 - fixed_point_position)) + 1) >> 1;

    // Polynomial expansion
    const int     dec_a = (sqmul_qs8(a, inv_ln2, fixed_point_position) >> fixed_point_position);
    const qint8_t alpha = sabs_qs8(sqsub_qs8(a, sqmul_qs8(ln2, sqshl_qs8(dec_a, fixed_point_position), fixed_point_position)));
    qint8_t       sum   = sqadd_qs8(sqmul_qs8(alpha, D, fixed_point_position), C);
    sum                 = sqadd_qs8(sqmul_qs8(alpha, sum, fixed_point_position), B);
    sum                 = sqadd_qs8(sqmul_qs8(alpha, sum, fixed_point_position), A);
    sum                 = sqmul_qs8(alpha, sum, fixed_point_position);
    sum                 = sqadd_qs8(sum, const_one);

    return (dec_a < 0) ? (sum >> -dec_a) : sqshl_qs8(sum, dec_a);
}

inline qint16_t sqexp_qs16(qint16_t a, int fixed_point_position)
{
    // Constants
    const qint16_t const_one = (1 << fixed_point_position);
    const qint16_t ln2       = ((0x58B9 >> (14 - fixed_point_position)) + 1) >> 1;
    const qint16_t inv_ln2   = (((0x38AA >> (14 - fixed_point_position)) + 1) >> 1) | const_one;
    const qint16_t A         = ((0x7FBA >> (14 - fixed_point_position)) + 1) >> 1;
    const qint16_t B         = ((0x3FE9 >> (14 - fixed_point_position)) + 1) >> 1;
    const qint16_t C         = ((0x1693 >> (14 - fixed_point_position)) + 1) >> 1;
    const qint16_t D         = ((0x0592 >> (14 - fixed_point_position)) + 1) >> 1;

    // Polynomial expansion
    const int      dec_a = (sqmul_qs16(a, inv_ln2, fixed_point_position) >> fixed_point_position);
    const qint16_t alpha = sabs_qs16(sqsub_qs16(a, sqmul_qs16(ln2, sqshl_qs16(dec_a, fixed_point_position), fixed_point_position)));
    qint16_t       sum   = sqadd_qs16(sqmul_qs16(alpha, D, fixed_point_position), C);
    sum                  = sqadd_qs16(sqmul_qs16(alpha, sum, fixed_point_position), B);
    sum                  = sqadd_qs16(sqmul_qs16(alpha, sum, fixed_point_position), A);
    sum                  = sqmul_qs16(alpha, sum, fixed_point_position);
    sum                  = sqadd_qs16(sum, const_one);

    return (dec_a < 0) ? (sum >> -dec_a) : sqshl_qs16(sum, dec_a);
}

inline qint8_t slog_qs8(qint8_t a, int fixed_point_position)
{
    // Constants
    qint8_t const_one = (1 << fixed_point_position);
    qint8_t ln2       = (0x58 >> (7 - fixed_point_position));
    qint8_t A         = (0x5C >> (7 - fixed_point_position - 1));
    qint8_t B         = -(0x56 >> (7 - fixed_point_position));
    qint8_t C         = (0x29 >> (7 - fixed_point_position));
    qint8_t D         = -(0x0A >> (7 - fixed_point_position));

    if((const_one == a) || (a < 0))
    {
        return 0;
    }
    else if(a < const_one)
    {
        return -slog_qs8(sdiv_qs8(const_one, a, fixed_point_position), fixed_point_position);
    }

    // Remove even powers of 2
    qint8_t shift_val = 31 - __builtin_clz(a >> fixed_point_position);
    a >>= shift_val;
    a = ssub_qs8(a, const_one);

    // Polynomial expansion
    qint8_t sum = sqadd_qs8(sqmul_qs8(a, D, fixed_point_position), C);
    sum         = sqadd_qs8(sqmul_qs8(a, sum, fixed_point_position), B);
    sum         = sqadd_qs8(sqmul_qs8(a, sum, fixed_point_position), A);
    sum         = sqmul_qs8(a, sum, fixed_point_position);

    return smul_qs8(sadd_qs8(sum, shift_val << fixed_point_position), ln2, fixed_point_position);
}

inline qint16_t slog_qs16(qint16_t a, int fixed_point_position)
{
    // Constants
    qint16_t const_one = (1 << fixed_point_position);
    qint16_t ln2       = (0x58B9 >> (7 - fixed_point_position));
    qint16_t A         = (0x5C0F >> (7 - fixed_point_position - 1));
    qint16_t B         = -(0x56AE >> (7 - fixed_point_position));
    qint16_t C         = (0x2933 >> (7 - fixed_point_position));
    qint16_t D         = -(0x0AA7 >> (7 - fixed_point_position));

    if((const_one == a) || (a < 0))
    {
        return 0;
    }
    else if(a < const_one)
    {
        return -slog_qs16(sdiv_qs16(const_one, a, fixed_point_position), fixed_point_position);
    }

    // Remove even powers of 2
    qint16_t shift_val = 31 - __builtin_clz(a >> fixed_point_position);
    a >>= shift_val;
    a = ssub_qs16(a, const_one);

    // Polynomial expansion
    qint16_t sum = sqadd_qs16(sqmul_qs16(a, D, fixed_point_position), C);
    sum          = sqadd_qs16(sqmul_qs16(a, sum, fixed_point_position), B);
    sum          = sqadd_qs16(sqmul_qs16(a, sum, fixed_point_position), A);
    sum          = sqmul_qs16(a, sum, fixed_point_position);

    return smul_qs16(sadd_qs16(sum, shift_val << fixed_point_position), ln2, fixed_point_position);
}

inline float scvt_f32_qs8(qint8_t a, int fixed_point_position)
{
    return static_cast<float>(a) / (1 << fixed_point_position);
}

inline qint8_t sqcvt_qs8_f32(float a, int fixed_point_position)
{
    // round_nearest_integer(a * 2^(fixed_point_position))
    return saturate_convert<float, qint8_t>(a * (1 << fixed_point_position) + ((a >= 0) ? 0.5 : -0.5));
}

inline float scvt_f32_qs16(qint16_t a, int fixed_point_position)
{
    return static_cast<float>(a) / (1 << fixed_point_position);
}

inline qint16_t sqcvt_qs16_f32(float a, int fixed_point_position)
{
    // round_nearest_integer(a * 2^(fixed_point_position))
    return saturate_convert<float, qint16_t>(a * (1 << fixed_point_position) + ((a >= 0) ? 0.5 : -0.5));
}

inline qint8_t sqmovn_qs16(qint16_t a)
{
    // Saturate the result in case of overflow and cast to qint8_t
    return saturate_convert<qint16_t, qint8_t>(a);
}

inline qint16_t sqmovn_qs32(qint32_t a)
{
    // Saturate the result in case of overflow and cast to qint16_t
    return saturate_convert<qint32_t, qint16_t>(a);
}
}
