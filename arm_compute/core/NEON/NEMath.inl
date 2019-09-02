/*
 * Copyright (c) 2016-2019 ARM Limited.
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

#include <cmath>

namespace arm_compute
{
/** Exponent polynomial coefficients */
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

/** Logarithm polynomial coefficients */
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

/** Sin polynomial coefficients */
constexpr float te_sin_coeff2 = 0.166666666666f; // 1/(2*3)
constexpr float te_sin_coeff3 = 0.05f;           // 1/(4*5)
constexpr float te_sin_coeff4 = 0.023809523810f; // 1/(6*7)
constexpr float te_sin_coeff5 = 0.013888888889f; // 1/(8*9)

#ifndef DOXYGEN_SKIP_THIS
inline float32x4_t vfloorq_f32(float32x4_t val)
{
    static const float32x4_t CONST_1 = vdupq_n_f32(1.f);

    const int32x4_t   z = vcvtq_s32_f32(val);
    const float32x4_t r = vcvtq_f32_s32(z);

    return vbslq_f32(vcgtq_f32(r, val), vsubq_f32(r, CONST_1), r);
}

inline float32x4_t vroundq_rte_f32(float32x4_t val)
{
#ifdef __aarch64__
    return vrndnq_f32(val);
#else  // __aarch64__
    static const float32x4_t CONST_HALF_FLOAT = vdupq_n_f32(0.5f);
    static const float32x4_t CONST_1_FLOAT    = vdupq_n_f32(1.f);
    static const int32x4_t   CONST_1_INT      = vdupq_n_s32(1);
    const float32x4_t        floor_val        = vfloorq_f32(val);
    const float32x4_t        diff             = vsubq_f32(val, floor_val);

    /*
    * Select the floor value when (diff<0.5 || (diff==0.5 && floor_val%2==0).
    * This condition is checked by vorrq_u32(vcltq_f32(diff, CONST_HALF_FLOAT) ,vandq_u32(vceqq_f32(diff, CONST_HALF_FLOAT) , vmvnq_u32(vtstq_s32(vandq_s32(vcvtq_s32_f32(floor_val), CONST_1_INT),CONST_1_INT))))
    */

    return vbslq_f32(vorrq_u32(vcltq_f32(diff, CONST_HALF_FLOAT), vandq_u32(vceqq_f32(diff, CONST_HALF_FLOAT), vmvnq_u32(vtstq_s32(vandq_s32(vcvtq_s32_f32(floor_val), CONST_1_INT), CONST_1_INT)))),
                     floor_val, vaddq_f32(floor_val, CONST_1_FLOAT));
#endif // __aarch64__
}

inline float32x2_t vinvsqrt_f32(float32x2_t x)
{
    float32x2_t sqrt_reciprocal = vrsqrte_f32(x);
    sqrt_reciprocal             = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmul_f32(vrsqrts_f32(vmul_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

inline float32x4_t vinvsqrtq_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);

    return sqrt_reciprocal;
}

inline float32x2_t vinv_f32(float32x2_t x)
{
    float32x2_t recip = vrecpe_f32(x);
    recip             = vmul_f32(vrecps_f32(x, recip), recip);
    recip             = vmul_f32(vrecps_f32(x, recip), recip);
    return recip;
}

inline float32x4_t vinvq_f32(float32x4_t x)
{
    float32x4_t recip = vrecpeq_f32(x);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}

inline float32x4_t vtaylor_polyq_f32(float32x4_t x, const std::array<float32x4_t, 8> &coeffs)
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

inline float32x4_t vexpq_f32(float32x4_t x)
{
    static const float32x4_t CONST_LN2          = vdupq_n_f32(0.6931471805f); // ln(2)
    static const float32x4_t CONST_INV_LN2      = vdupq_n_f32(1.4426950408f); // 1/ln(2)
    static const float32x4_t CONST_0            = vdupq_n_f32(0.f);
    static const int32x4_t   CONST_NEGATIVE_126 = vdupq_n_s32(-126);

    // Perform range reduction [-log(2),log(2)]
    int32x4_t   m   = vcvtq_s32_f32(vmulq_f32(x, CONST_INV_LN2));
    float32x4_t val = vmlsq_f32(x, vcvtq_f32_s32(m), CONST_LN2);

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, exp_tab);

    // Reconstruct
    poly = vreinterpretq_f32_s32(vqaddq_s32(vreinterpretq_s32_f32(poly), vqshlq_n_s32(m, 23)));
    poly = vbslq_f32(vcltq_s32(m, CONST_NEGATIVE_126), CONST_0, poly);

    return poly;
}

inline float32x4_t vlogq_f32(float32x4_t x)
{
    static const int32x4_t   CONST_127 = vdupq_n_s32(127);           // 127
    static const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    int32x4_t   m   = vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23)), CONST_127);
    float32x4_t val = vreinterpretq_f32_s32(vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_f32(val, log_tab);

    // Reconstruct
    poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

    return poly;
}

inline float32x4_t vtanhq_f32(float32x4_t val)
{
    static const float32x4_t CONST_1        = vdupq_n_f32(1.f);
    static const float32x4_t CONST_2        = vdupq_n_f32(2.f);
    static const float32x4_t CONST_MIN_TANH = vdupq_n_f32(-10.f);
    static const float32x4_t CONST_MAX_TANH = vdupq_n_f32(10.f);

    float32x4_t x     = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
    float32x4_t exp2x = vexpq_f32(vmulq_f32(CONST_2, x));
    float32x4_t num   = vsubq_f32(exp2x, CONST_1);
    float32x4_t den   = vaddq_f32(exp2x, CONST_1);
    float32x4_t tanh  = vmulq_f32(num, vinvq_f32(den));
    return tanh;
}

inline float32x4_t vpowq_f32(float32x4_t val, float32x4_t n)
{
    return vexpq_f32(vmulq_f32(n, vlogq_f32(val)));
}

inline float32x4_t vsinq_f32(float32x4_t val)
{
    const float32x4_t pi_v   = vdupq_n_f32(M_PI);
    const float32x4_t pio2_v = vdupq_n_f32(M_PI / 2);
    const float32x4_t ipi_v  = vdupq_n_f32(1 / M_PI);

    //Find positive or negative
    const int32x4_t  c_v    = vabsq_s32(vcvtq_s32_f32(vmulq_f32(val, ipi_v)));
    const uint32x4_t sign_v = vcleq_f32(val, vdupq_n_f32(0));
    const uint32x4_t odd_v  = vandq_u32(vreinterpretq_u32_s32(c_v), vdupq_n_u32(1));

    uint32x4_t neg_v = veorq_u32(odd_v, sign_v);

    //Modulus a - (n * int(a*(1/n)))
    float32x4_t      ma    = vsubq_f32(vabsq_f32(val), vmulq_f32(pi_v, vcvtq_f32_s32(c_v)));
    const uint32x4_t reb_v = vcgeq_f32(ma, pio2_v);

    //Rebase a between 0 and pi/2
    ma = vbslq_f32(reb_v, vsubq_f32(pi_v, ma), ma);

    //Taylor series
    const float32x4_t ma2 = vmulq_f32(ma, ma);

    //2nd elem: x^3 / 3!
    float32x4_t elem = vmulq_f32(vmulq_f32(ma, ma2), vdupq_n_f32(te_sin_coeff2));
    float32x4_t res  = vsubq_f32(ma, elem);

    //3rd elem: x^5 / 5!
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff3));
    res  = vaddq_f32(res, elem);

    //4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff4));
    res  = vsubq_f32(res, elem);

    //5th elem: x^9 / 9!
    elem = vmulq_f32(vmulq_f32(elem, ma2), vdupq_n_f32(te_sin_coeff5));
    res  = vaddq_f32(res, elem);

    //Change of sign
    neg_v = vshlq_n_u32(neg_v, 31);
    res   = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(res), neg_v));
    return res;
}

inline float32x2_t vsin_f32(float32x2_t val)
{
    const float32x2_t pi_v   = vdup_n_f32(M_PI);
    const float32x2_t pio2_v = vdup_n_f32(M_PI / 2);
    const float32x2_t ipi_v  = vdup_n_f32(1 / M_PI);

    //Find positive or negative
    const int32x2_t  c_v    = vabs_s32(vcvt_s32_f32(vmul_f32(val, ipi_v)));
    const uint32x2_t sign_v = vcle_f32(val, vdup_n_f32(0));
    const uint32x2_t odd_v  = vand_u32(vreinterpret_u32_s32(c_v), vdup_n_u32(1));

    uint32x2_t neg_v = veor_u32(odd_v, sign_v);

    //Modulus a - (n * int(a*(1/n)))
    float32x2_t      ma    = vsub_f32(vabs_f32(val), vmul_f32(pi_v, vcvt_f32_s32(c_v)));
    const uint32x2_t reb_v = vcge_f32(ma, pio2_v);

    //Rebase a between 0 and pi/2
    ma = vbsl_f32(reb_v, vsub_f32(pi_v, ma), ma);

    //Taylor series
    const float32x2_t ma2 = vmul_f32(ma, ma);

    //2nd elem: x^3 / 3!
    float32x2_t elem = vmul_f32(vmul_f32(ma, ma2), vdup_n_f32(te_sin_coeff2));
    float32x2_t res  = vsub_f32(ma, elem);

    //3rd elem: x^5 / 5!
    elem = vmul_f32(vmul_f32(elem, ma2), vdup_n_f32(te_sin_coeff3));
    res  = vadd_f32(res, elem);

    //4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
    elem = vmul_f32(vmul_f32(elem, ma2), vdup_n_f32(te_sin_coeff4));
    res  = vsub_f32(res, elem);

    //5th elem: x^9 / 9!
    elem = vmul_f32(vmul_f32(elem, ma2), vdup_n_f32(te_sin_coeff5));
    res  = vadd_f32(res, elem);

    //Change of sign
    neg_v = vshl_n_u32(neg_v, 31);
    res   = vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(res), neg_v));
    return res;
}

#endif /* DOXYGEN_SKIP_THIS */

inline int32x4_t rounding_divide_by_pow2(int32x4_t x, int exponent)
{
    const int32x4_t shift_vec  = vdupq_n_s32(-exponent);
    const int32x4_t fixup      = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
    const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
    return vrshlq_s32(fixed_up_x, shift_vec);
}

inline int32_t rounding_divide_by_pow2(int32_t x, int exponent)
{
    const int32_t mask      = (1 << exponent) - 1;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + ((x & mask) > threshold ? 1 : 0);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/** Exponent polynomial coefficients */
/** Logarithm polynomial coefficients */
#ifndef DOXYGEN_SKIP_THIS
inline float16x8_t vfloorq_f16(float16x8_t val)
{
    static const float16x8_t CONST_1 = vdupq_n_f16(1.f);

    const int16x8_t   z = vcvtq_s16_f16(val);
    const float16x8_t r = vcvtq_f16_s16(z);

    return vbslq_f16(vcgtq_f16(r, val), vsubq_f16(r, CONST_1), r);
}

inline float16x8_t vroundq_rte_f16(float16x8_t val)
{
    return vrndnq_f16(val);
}

inline float16x4_t vinvsqrt_f16(float16x4_t x)
{
    float16x4_t sqrt_reciprocal = vrsqrte_f16(x);
    sqrt_reciprocal             = vmul_f16(vrsqrts_f16(vmul_f16(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmul_f16(vrsqrts_f16(vmul_f16(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    return sqrt_reciprocal;
}

inline float16x8_t vinvsqrtq_f16(float16x8_t x)
{
    float16x8_t sqrt_reciprocal = vrsqrteq_f16(x);
    sqrt_reciprocal             = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal             = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    return sqrt_reciprocal;
}

inline float16x4_t vinv_f16(float16x4_t x)
{
    float16x4_t recip = vrecpe_f16(x);
    recip             = vmul_f16(vrecps_f16(x, recip), recip);
    recip             = vmul_f16(vrecps_f16(x, recip), recip);
    return recip;
}

inline float16x8_t vinvq_f16(float16x8_t x)
{
    float16x8_t recip = vrecpeq_f16(x);
    recip             = vmulq_f16(vrecpsq_f16(x, recip), recip);
    recip             = vmulq_f16(vrecpsq_f16(x, recip), recip);
    return recip;
}

inline float16x8_t vtanhq_f16(float16x8_t val)
{
    const float16x8_t CONST_1        = vdupq_n_f16(1.f);
    const float16x8_t CONST_2        = vdupq_n_f16(2.f);
    const float16x8_t CONST_MIN_TANH = vdupq_n_f16(-10.f);
    const float16x8_t CONST_MAX_TANH = vdupq_n_f16(10.f);

    const float16x8_t x     = vminq_f16(vmaxq_f16(val, CONST_MIN_TANH), CONST_MAX_TANH);
    const float16x8_t exp2x = vexpq_f16(vmulq_f16(CONST_2, x));
    const float16x8_t num   = vsubq_f16(exp2x, CONST_1);
    const float16x8_t den   = vaddq_f16(exp2x, CONST_1);
    const float16x8_t tanh  = vmulq_f16(num, vinvq_f16(den));
    return tanh;
}

inline float16x8_t vtaylor_polyq_f16(float16x8_t x, const std::array<float16x8_t, 8> &coeffs)
{
    const float16x8_t A   = vaddq_f16(coeffs[0], vmulq_f16(coeffs[4], x));
    const float16x8_t B   = vaddq_f16(coeffs[2], vmulq_f16(coeffs[6], x));
    const float16x8_t C   = vaddq_f16(coeffs[1], vmulq_f16(coeffs[5], x));
    const float16x8_t D   = vaddq_f16(coeffs[3], vmulq_f16(coeffs[7], x));
    const float16x8_t x2  = vmulq_f16(x, x);
    const float16x8_t x4  = vmulq_f16(x2, x2);
    const float16x8_t res = vaddq_f16(vaddq_f16(A, vmulq_f16(B, x2)), vmulq_f16(vaddq_f16(C, vmulq_f16(D, x2)), x4));
    return res;
}

inline float16x8_t vexpq_f16(float16x8_t x)
{
    // TODO (COMPMID-1535) : Revisit FP16 approximations
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

    const float16x8_t res = vcvt_high_f16_f32(vcvt_f16_f32(vexpq_f32(x_low)), vexpq_f32(x_high));
    return res;
}

inline float16x8_t vlogq_f16(float16x8_t x)
{
    // TODO (COMPMID-1535) : Revisit FP16 approximations
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

    const float16x8_t res = vcvt_high_f16_f32(vcvt_f16_f32(vlogq_f32(x_low)), vlogq_f32(x_high));
    return res;
}

inline float16x8_t vpowq_f16(float16x8_t val, float16x8_t n)
{
    // TODO (giaiod01) - COMPMID-1535
    float32x4_t n0_f32   = vcvt_f32_f16(vget_low_f16(n));
    float32x4_t n1_f32   = vcvt_f32_f16(vget_high_f16(n));
    float32x4_t val0_f32 = vcvt_f32_f16(vget_low_f16(val));
    float32x4_t val1_f32 = vcvt_f32_f16(vget_high_f16(val));

    float32x4_t res0_f32 = vexpq_f32(vmulq_f32(n0_f32, vlogq_f32(val0_f32)));
    float32x4_t res1_f32 = vexpq_f32(vmulq_f32(n1_f32, vlogq_f32(val1_f32)));

    return vcombine_f16(vcvt_f16_f32(res0_f32), vcvt_f16_f32(res1_f32));
}

inline float16x8_t vsinq_f16(float16x8_t val)
{
    const float32x4_t val_high = vcvt_f32_f16(vget_high_f16(val));
    const float32x4_t val_low  = vcvt_f32_f16(vget_low_f16(val));

    const float32x4_t res_high = vsinq_f32(val_high);
    const float32x4_t res_low  = vsinq_f32(val_low);

    return vcombine_f16(vcvt_f16_f32(res_low), vcvt_f16_f32(res_high));
}

inline float16x4_t vsin_f16(float16x4_t val)
{
    const float32x4_t val_f32  = vcvt_f32_f16(val);
    const float32x2_t val_high = vget_high_f32(val_f32);
    const float32x2_t val_low  = vget_low_f32(val_f32);

    const float32x2_t res_high = vsin_f32(val_high);
    const float32x2_t res_low  = vsin_f32(val_low);

    return vcvt_f16_f32(vcombine_f32(res_low, res_high));
}

#endif /* DOXYGEN_SKIP_THIS */
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
