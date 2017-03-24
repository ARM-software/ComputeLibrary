/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEMATH_H__
#define __ARM_COMPUTE_NEMATH_H__

#include <arm_neon.h>

namespace arm_compute
{
/* Exponent polynomial coefficients */
const std::array<float32x4_t, 8> exp_tab =
{
    {
        vdupq_n_f32(1.f),
        vdupq_n_f32(0.0416598916054f),
        vdupq_n_f32(0.500000596046f),
        vdupq_n_f32(0.0014122662833f),
        vdupq_n_f32(1.00000011921f),
        vdupq_n_f32(0.00833693705499f),
        vdupq_n_f32(0.166665703058f),
        vdupq_n_f32(0.000195780929062f),
    }
};

/* Logarithm polynomial coefficients */
const std::array<float32x4_t, 8> log_tab =
{
    {
        vdupq_n_f32(-2.29561495781f),
        vdupq_n_f32(-2.47071170807f),
        vdupq_n_f32(-5.68692588806f),
        vdupq_n_f32(-0.165253549814f),
        vdupq_n_f32(5.17591238022f),
        vdupq_n_f32(0.844007015228f),
        vdupq_n_f32(4.58445882797f),
        vdupq_n_f32(0.0141278216615f),
    }
};

/** Calculate inverse square root.
 *
 * @param x Input value.
 *
 * @return The calculated inverse square root.
 */
inline float32x4_t vinvsqrt_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

/** Calculate reciprocal.
 *
 * @param x Input value.
 *
 * @return The calculated reciprocal.
 */
inline float32x4_t vinv_f32(const float32x4_t &x)
{
    float32x4_t recip = vrecpeq_f32(x);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}

/** Perform a 7th degree polynomial approximation using Estrin's method.
 *
 * @param x       Input vector value in F32 format.
 * @param coeffs  Polynomial coefficients table.
 *
 * @return The calculated approximation.
 */
inline float32x4_t vtaylor_poly_f32(const float32x4_t &x, const std::array<float32x4_t, 8> &coeffs)
{
    float32x4_t A   = vmlaq_f32(coeffs[0], coeffs[4], x);
    float32x4_t B   = vmlaq_f32(coeffs[2], coeffs[6], x);
    float32x4_t C   = vmlaq_f32(coeffs[1], coeffs[5], x);
    float32x4_t D   = vmlaq_f32(coeffs[3], coeffs[7], x);
    float32x4_t x2  = vmulq_f32(x, x);
    float32x4_t x4  = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
    return res;
}

/** Calculate exponential
 *
 * @param x Input vector value in F32 format.
 *
 * @return The calculated exponent.
 */
inline float32x4_t vexp_f32(const float32x4_t &x)
{
    static const float32x4_t CONST_LN2     = vdupq_n_f32(0.6931471805f); // ln(2)
    static const float32x4_t CONST_INV_LN2 = vdupq_n_f32(1.4426950408f); // 1/ln(2)

    // Perform range reduction [-log(2),log(2)]
    int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    // Polynomial Approximation
    float32x4_t poly = vtaylor_poly_f32(val, exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(poly), vshlq_n_s32(m, 23)));

    return poly;
}

/** Calculate logarithm
 *
 * @param x Input vector value in F32 format.
 *
 * @return The calculated logarithm.
 */
inline float32x4_t vlog_f32(const float32x4_t &x)
{
    static const int32x4_t   CONST_127 = vdupq_n_s32(127);           // 127
    static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    int32x4_t   m   = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
    float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

    // Polynomial Approximation
    float32x4_t poly = vtaylor_poly_f32(val, log_tab);

    // Reconstruct
    poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

    return poly;
}

/** Calculate hyperbolic tangent.
 *
 * tanh(x) = (e^2x - 1)/(e^2x + 1)
 *
 * @param val Input vector value in F32 format.
 *
 * @return The calculated Hyperbolic Tangent.
 */
inline float32x4_t vtanh_f32(const float32x4_t &val)
{
    static const float32x4_t CONST_1 = vdupq_n_f32(1.f); // 1.f
    static const float32x4_t CONST_2 = vdupq_n_f32(2.f); // 2.f

    float32x4_t exp2x = vexp_f32(vmulq_f32(CONST_2, val));
    float32x4_t num   = vsubq_f32(exp2x, CONST_1);
    float32x4_t den   = vaddq_f32(exp2x, CONST_1);
    float32x4_t tanh  = vmulq_f32(num, vinv_f32(den));
    return tanh;
}

/** Calculate n power of a number.
 *
 * pow(x,n) = e^(n*log(x))
 *
 * @param val Input vector value in F32 format.
 * @param n   Powers to raise the input to.
 *
 * @return The calculated power.
 */
inline float32x4_t vpowq_f32(const float32x4_t &val, const float32x4_t &n)
{
    return vexp_f32(vmulq_f32(n, vlog_f32(val)));
}
}

#endif /* __ARM_COMPUTE_NEMATH_H__ */
