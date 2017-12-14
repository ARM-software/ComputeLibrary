/*
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
#ifndef __ARM_COMPUTE_FIXEDPOINT_H__
#define __ARM_COMPUTE_FIXEDPOINT_H__

#include <cstdint>

namespace arm_compute
{
using qint8_t  = int8_t;  /**< 8 bit fixed point scalar value */
using qint16_t = int16_t; /**< 16 bit fixed point scalar value */
using qint32_t = int32_t; /**< 32 bit fixed point scalar value */
using qint64_t = int64_t; /**< 64 bit fixed point scalar value */

/** 8 bit fixed point scalar saturating shift left
 *
 * @param[in] a     First 8 bit fixed point input
 * @param[in] shift Shift amount (positive only values)
 *
 * @return The result of the 8 bit fixed point shift. The result is saturated in case of overflow
 */
qint8_t sqshl_qs8(qint8_t a, int shift);

/** 8 bit fixed point scalar shift right
 *
 * @param[in] a     First 8 bit fixed point input
 * @param[in] shift Shift amount (positive only values)
 *
 * @return The result of the 8 bit fixed point shift
 */
qint8_t sshr_qs8(qint8_t a, int shift);

/** 16 bit fixed point scalar shift right
 *
 * @param[in] a     First 16 bit fixed point input
 * @param[in] shift Shift amount (positive only values)
 *
 * @return The result of the 16 bit fixed point shift
 */
qint16_t sshr_qs16(qint16_t a, int shift);

/** 16 bit fixed point scalar saturating shift left
 *
 * @param[in] a     First 16 bit fixed point input
 * @param[in] shift Shift amount (positive only values)
 *
 * @return The result of the 16 bit fixed point shift. The result is saturated in case of overflow
 */
qint16_t sqshl_qs16(qint16_t a, int shift);

/** 8 bit fixed point scalar absolute value
 *
 * @param[in] a 8 bit fixed point input
 *
 * @return The result of the 8 bit fixed point absolute value
 */
qint8_t sabs_qs8(qint8_t a);

/** 16 bit fixed point scalar absolute value
 *
 * @param[in] a 16 bit fixed point input
 *
 * @return The result of the 16 bit fixed point absolute value
 */
qint16_t sabs_qs16(qint16_t a);

/** 8 bit fixed point scalar add
 *
 * @param[in] a First 8 bit fixed point input
 * @param[in] b Second 8 bit fixed point input
 *
 * @return The result of the 8 bit fixed point addition
 */
qint8_t sadd_qs8(qint8_t a, qint8_t b);

/** 16 bit fixed point scalar add
 *
 * @param[in] a First 16 bit fixed point input
 * @param[in] b Second 16 bit fixed point input
 *
 * @return The result of the 16 bit fixed point addition
 */
qint16_t sadd_qs16(qint16_t a, qint16_t b);

/** 8 bit fixed point scalar saturating add
 *
 * @param[in] a First 8 bit fixed point input
 * @param[in] b Second 8 bit fixed point input
 *
 * @return The result of the 8 bit fixed point addition. The result is saturated in case of overflow
 */
qint8_t sqadd_qs8(qint8_t a, qint8_t b);

/** 16 bit fixed point scalar saturating add
 *
 * @param[in] a First 16 bit fixed point input
 * @param[in] b Second 16 bit fixed point input
 *
 * @return The result of the 16 bit fixed point addition. The result is saturated in case of overflow
 */
qint16_t sqadd_qs16(qint16_t a, qint16_t b);

/** 32 bit fixed point scalar saturating add
 *
 * @param[in] a First 32 bit fixed point input
 * @param[in] b Second 32 bit fixed point input
 *
 * @return The result of the 32 bit fixed point addition. The result is saturated in case of overflow
 */
qint32_t sqadd_qs32(qint32_t a, qint32_t b);

/** 8 bit fixed point scalar subtraction
 *
 * @param[in] a First 8 bit fixed point input
 * @param[in] b Second 8 bit fixed point input
 *
 * @return The result of the 8 bit fixed point subtraction
 */
qint8_t ssub_qs8(qint8_t a, qint8_t b);

/** 16 bit fixed point scalar subtraction
 *
 * @param[in] a First 16 bit fixed point input
 * @param[in] b Second 16 bit fixed point input
 *
 * @return The result of the 16 bit fixed point subtraction
 */
qint16_t ssub_qs16(qint16_t a, qint16_t b);

/** 8 bit fixed point scalar saturating subtraction
 *
 * @param[in] a First 8 bit fixed point input
 * @param[in] b Second 8 bit fixed point input
 *
 * @return The result of the 8 bit fixed point subtraction. The result is saturated in case of overflow
 */
qint8_t sqsub_qs8(qint8_t a, qint8_t b);

/** 16 bit fixed point scalar saturating subtraction
 *
 * @param[in] a First 16 bit fixed point input
 * @param[in] b Second 16 bit fixed point input
 *
 * @return The result of the 16 bit fixed point subtraction. The result is saturated in case of overflow
 */
qint16_t sqsub_qs16(qint16_t a, qint16_t b);

/** 8 bit fixed point scalar multiply
 *
 * @param[in] a                    First 8 bit fixed point input
 * @param[in] b                    Second 8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point multiplication.
 */
qint8_t smul_qs8(qint8_t a, qint8_t b, int fixed_point_position);

/** 16 bit fixed point scalar multiply
 *
 * @param[in] a                    First 16 bit fixed point input
 * @param[in] b                    Second 16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point multiplication.
 */
qint16_t smul_qs16(qint16_t a, qint16_t b, int fixed_point_position);

/** 8 bit fixed point scalar saturating multiply
 *
 * @param[in] a                    First 8 bit fixed point input
 * @param[in] b                    Second 8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point multiplication. The result is saturated in case of overflow
 */
qint8_t sqmul_qs8(qint8_t a, qint8_t b, int fixed_point_position);

/** 16 bit fixed point scalar saturating multiply
 *
 * @param[in] a                    First 16 bit fixed point input
 * @param[in] b                    Second 16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point multiplication. The result is saturated in case of overflow
 */
qint16_t sqmul_qs16(qint16_t a, qint16_t b, int fixed_point_position);

/** 8 bit fixed point scalar multiply long
 *
 * @param[in] a                    First 8 bit fixed point input
 * @param[in] b                    Second 8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point multiplication long. The result is saturated in case of overflow
 */
qint16_t sqmull_qs8(qint8_t a, qint8_t b, int fixed_point_position);

/** 16 bit fixed point scalar multiply long
 *
 * @param[in] a                    First 16 bit fixed point input
 * @param[in] b                    Second 16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point multiplication long. The result is saturated in case of overflow
 */
qint32_t sqmull_qs16(qint16_t a, qint16_t b, int fixed_point_position);

/** 16 bit fixed point scalar saturating multiply
 *
 * @param[in] a                    First 16 bit fixed point input
 * @param[in] b                    Second 16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point multiplication. The result is saturated in case of overflow
 */
qint16_t sqmul_qs16(qint16_t a, qint16_t b, int fixed_point_position);

/** 8 bit fixed point scalar inverse square root
 *
 * @param[in] a                    8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point inverse square root.
 */
qint8_t sinvsqrt_qs8(qint8_t a, int fixed_point_position);

/** 16 bit fixed point scalar inverse square root
 *
 * @param[in] a                    16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point inverse square root.
 */
qint16_t sinvsqrt_qs16(qint16_t a, int fixed_point_position);

/** 8 bit fixed point scalar division
 *
 * @param[in] a                    First 8 bit fixed point input
 * @param[in] b                    Second 8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point division.
 */
qint8_t sdiv_qs8(qint8_t a, qint8_t b, int fixed_point_position);

/** 16 bit fixed point scalar division
 *
 * @param[in] a                    First 16 bit fixed point input
 * @param[in] b                    Second 16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point division.
 */
qint16_t sdiv_qs16(qint16_t a, qint16_t b, int fixed_point_position);

/** 8 bit fixed point scalar exponential
 *
 * @param[in] a                    8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point exponential.
 */
qint8_t sqexp_qs8(qint8_t a, int fixed_point_position);

/** 16 bit fixed point scalar exponential
 *
 * @param[in] a                    16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point exponential.
 */
qint16_t sqexp_qs16(qint16_t a, int fixed_point_position);

/** 16 bit fixed point scalar exponential
 *
 * @param[in] a                    16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point exponential.
 */
qint16_t sexp_qs16(qint16_t a, int fixed_point_position);

/** 8 bit fixed point scalar logarithm
 *
 * @param[in] a                    8 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 8 bit fixed point logarithm.
 */
qint8_t slog_qs8(qint8_t a, int fixed_point_position);

/** 16 bit fixed point scalar logarithm
 *
 * @param[in] a                    16 bit fixed point input
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the 16 bit fixed point logarithm.
 */
qint16_t slog_qs16(qint16_t a, int fixed_point_position);

/** Convert an 8 bit fixed point to float
 *
 * @param[in] a                    Input to convert
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 8 bit fixed point -> float
 */
float scvt_f32_qs8(qint8_t a, int fixed_point_position);

/** Convert a float to 8 bit fixed point
 *
 * @param[in] a                    Input to convert
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 8 bit fixed point
 */
qint8_t sqcvt_qs8_f32(float a, int fixed_point_position);

/** Convert a 16 bit fixed point to float
 *
 * @param[in] a                    Input to convert
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion 16 bit fixed point -> float
 */
float scvt_f32_qs16(qint16_t a, int fixed_point_position);

/** Convert a float to 16 bit fixed point
 *
 * @param[in] a                    Input to convert
 * @param[in] fixed_point_position Fixed point position that expresses the number of bits for the fractional part of the number
 *
 * @return The result of the conversion float -> 16 bit fixed point
 */
qint16_t sqcvt_qs16_f32(float a, int fixed_point_position);

/** Scalar saturating move and narrow.
 *
 * @param[in] a Input to convert to 8 bit fixed point
 *
 * @return The narrowing conversion to 8 bit
 */
qint8_t sqmovn_qs16(qint16_t a);

/** Scalar saturating move and narrow.
 *
 * @param[in] a Input to convert to 16 bit fixed point
 *
 * @return The narrowing conversion to 16 bit
 */
qint16_t sqmovn_qs32(qint32_t a);
}
#include "arm_compute/core/FixedPoint.inl"
#endif /* __ARM_COMPUTE_FIXEDPOINT_H__ */
