/*
 * Copyright (c) 2020-2024 Arm Limited.
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

#ifndef ACL_SRC_CORE_NEON_SVE2MATH_H
#define ACL_SRC_CORE_NEON_SVE2MATH_H

#ifdef ARM_COMPUTE_ENABLE_SVE2
namespace arm_compute
{
/** Calculate n power of a number.
 *
 * pow(x,n) = e^(n*log(x))
 *
 * @param[in] pg Input predicate.
 * @param[in] a  Input vector value in F16 format.
 * @param[in] b  Powers to raise the input to.
 *
 * @return The calculated power.
 */
svfloat16_t svpow_f16_z_sve2(svbool_t pg, svfloat16_t a, svfloat16_t b);

/** Calculate exponential
 *
 * @param[in] pg Input predicate.
 * @param[in] x  Input vector value in F16 format.
 *
 * @return The calculated exponent.
 */
svfloat16_t svexp_f16_z_sve2(svbool_t pg, svfloat16_t x);

/** Calculate logarithm
 *
 * @param[in] pg Input predicate.
 * @param[in] x  Input vector value in F32 format.
 *
 * @return The calculated logarithm.
 */
svfloat16_t svlog_f16_z_sve2(svbool_t pg, svfloat16_t x);

/** Calculate sine.
 *
 * @param[in] pg  Input predicate.
 * @param[in] val Input vector value in radians, F16 format.
 *
 * @return The calculated sine.
 */
svfloat16_t svsin_f16_z_sve2(svbool_t pg, svfloat16_t val);
} // namespace arm_compute
#include "src/core/NEON/SVE2Math.inl"
#endif // ARM_COMPUTE_ENABLE_SVE2
#endif // ACL_SRC_CORE_NEON_SVE2MATH_H
