/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef ARM_COMPUTE_SVEMATH_H
#define ARM_COMPUTE_SVEMATH_H

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#include <array>

namespace arm_compute
{
/** Calculate exponent.
 *
 * @param[in] pg  Input reciprocal.
 * @param[in] val Input vector value in F32 format.
 *
 * @return The calculated exponent.
 */
svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t val);

/** Calculate reciprocal.
 *
 * @param[in] pg Input reciprocal.
 * @param[in] x  Input value.
 *
 * @return The calculated reciprocal.
 */
svfloat32_t svinv_f32_z(svbool_t pg, svfloat32_t x);

/** Calculate logarithm
 *
 * @param[in] pg Input reciprocal.
 * @param[in] x  Input vector value in F32 format.
 *
 * @return The calculated logarithm.
 */
svfloat32_t svlog_f32_z(svbool_t pg, svfloat32_t x);

/** Calculate hyperbolic tangent.
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @note We clamp x to [-5,5] to avoid overflowing issues.
 *
 * @param[in] pg  Input reciprocal.
 * @param[in] val Input vector value in F32 format.
 *
 * @return The calculated Hyperbolic Tangent.
 */
svfloat32_t svtanh_f32_z(svbool_t pg, svfloat32_t val);

/** Calculate hyperbolic tangent.
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @note We clamp x to [-5,5] to avoid overflowing issues.
 *
 * @param[in] pg  Input reciprocal.
 * @param[in] val Input vector value in F16 format.
 *
 * @return The calculated Hyperbolic Tangent.
 */
svfloat16_t svtanh_f16_z(svbool_t pg, svfloat16_t val);

/** Calculate exponential
 *
 * @param[in] pg Input reciprocal.
 * @param[in] x  Input vector value in F16 format.
 *
 * @return The calculated exponent.
 */
svfloat16_t svexp_f16_z(svbool_t pg, svfloat16_t x);

/** Calculate reciprocal.
 *
 * @param[in] pg Input reciprocal.
 * @param[in] x  Input value.
 *
 * @return The calculated reciprocal.
 */
svfloat16_t svinv_f16_z(svbool_t pg, svfloat16_t x);

/** Calculate logarithm
 *
 * @param[in] pg Input reciprocal.
 * @param[in] x  Input vector value in F32 format.
 *
 * @return The calculated logarithm.
 */
svfloat16_t svlog_f16_z(svbool_t pg, svfloat16_t x);

} // namespace arm_compute
#include "src/core/NEON/SVEMath.inl"
#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* ARM_COMPUTE_SVEMATH_H */