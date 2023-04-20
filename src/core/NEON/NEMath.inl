/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#include "support/ToolchainSupport.h"

#include <cmath>
#include <limits>

namespace arm_compute
{
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
inline float32x4_t prefer_vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c)
{
#ifdef __aarch64__
    return vfmaq_f32(a, b, c);
#else  // __aarch64__
    return vmlaq_f32(a, b, c);
#endif // __aarch64__
}

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
#else // __aarch64__
    static const float32x4_t CONST_HALF_FLOAT = vdupq_n_f32(0.5f);
    static const float32x4_t CONST_1_FLOAT    = vdupq_n_f32(1.f);
    static const int32x4_t   CONST_1_INT      = vdupq_n_s32(1);
    const float32x4_t        floor_val        = vfloorq_f32(val);
    const float32x4_t        diff             = vsubq_f32(val, floor_val);
    const float32x4_t        fp32_upper_limit = vreinterpretq_f32_u32(vdupq_n_u32(0x4B000000)); // 0x4B000000 = (23U + 127U) << 23U

    /*
    * 1. Select the floor value when (diff<0.5 || (diff==0.5 && floor_val%2==0).
    *    This condition is checked by vorrq_u32(vcltq_f32(diff, CONST_HALF_FLOAT) ,vandq_u32(vceqq_f32(diff, CONST_HALF_FLOAT) , vmvnq_u32(vtstq_s32(vandq_s32(vcvtq_s32_f32(floor_val), CONST_1_INT),CONST_1_INT))))
    *
    * 2. In case the input value (val) is out of signed int32 range, then simple use the input value as the rounded value
    *    Because:
    *    in this case converting to int32 would saturate
    *    If the input float value is >= 2^23 * 1.00... 23 Zeros ..0  then the rounded value is exactly equal to the input value.
    *    Because:
    *    in IEEE single precision floating point representation the fraction part is 23 bit, so if exponent is 23 it means the fraction part = 0 as any digits after decimal point are truncated.
    *    Hence, rounding has no effect:
    *    Threshold upper limit with format |S|E(8bits)|   Fraction(23bits)     | = (23 + 127) << 23 (assuming positive sign): Adding 127, because 127 represents the actual zero in this format.
    */

    float32x4_t rounded_val = vbslq_f32(vorrq_u32(vcltq_f32(diff, CONST_HALF_FLOAT),
                                                  vandq_u32(vceqq_f32(diff, CONST_HALF_FLOAT),
                                                            vmvnq_u32(vtstq_s32(vandq_s32(vcvtq_s32_f32(floor_val), CONST_1_INT),CONST_1_INT)))),
                                        floor_val, vaddq_f32(floor_val, CONST_1_FLOAT));

    float32x4_t result      = vbslq_f32(vcgeq_f32(vabsq_f32(val), fp32_upper_limit), val, rounded_val);

    return result;
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

static const uint32_t exp_f32_coeff[] =
{
    0x3f7ffff6, // x^1: 0x1.ffffecp-1f
    0x3efffedb, // x^2: 0x1.fffdb6p-2f
    0x3e2aaf33, // x^3: 0x1.555e66p-3f
    0x3d2b9f17, // x^4: 0x1.573e2ep-5f
    0x3c072010, // x^5: 0x1.0e4020p-7f
};

inline float32x4_t vexpq_f32(float32x4_t x)
{
    const auto c1 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[0]));
    const auto c2 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[1]));
    const auto c3 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[2]));
    const auto c4 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[3]));
    const auto c5 = vreinterpretq_f32_u32(vdupq_n_u32(exp_f32_coeff[4]));

    const auto shift      = vreinterpretq_f32_u32(vdupq_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2    = vreinterpretq_f32_u32(vdupq_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi = vreinterpretq_f32_u32(vdupq_n_u32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo = vreinterpretq_f32_u32(vdupq_n_u32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = vdupq_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = vdupq_n_f32(88.37f); // Approximately ln(2^127.5)
    const auto zero      = vdupq_n_f32(0.f);
    const auto min_input = vdupq_n_f32(-86.64f); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127 forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n) + 127 will occupy
    //     the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part of x / ln(2)
    //     (i.e. n) because the decimal part has been pushed out and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    const auto z     = prefer_vfmaq_f32(shift, x, inv_ln2);
    const auto n     = z - shift;
    const auto scale = vreinterpretq_f32_u32(vreinterpretq_u32_f32(z) << 23); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in term of accuracy and performance.
    const auto r_hi = prefer_vfmaq_f32(x, n, neg_ln2_hi);
    const auto r    = prefer_vfmaq_f32(r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = r * r;

    const auto p1     = c1 * r;
    const auto p23    = prefer_vfmaq_f32(c2, c3, r);
    const auto p45    = prefer_vfmaq_f32(c4, c5, r);
    const auto p2345  = prefer_vfmaq_f32(p23, p45, r2);
    const auto p12345 = prefer_vfmaq_f32(p1, p2345, r2);

    auto poly = prefer_vfmaq_f32(scale, p12345, scale);

    // Handle underflow and overflow.
    poly = vbslq_f32(vcltq_f32(x, min_input), zero, poly);
    poly = vbslq_f32(vcgtq_f32(x, max_input), inf, poly);

    return poly;
}

#ifdef __aarch64__
inline float32x4_t verfq_f32(float32x4_t x)
{
    static const float       erffdata[4] = { 0.278393f, 0.230389f, 0.000972f, 0.078108f };
    static const float32x4_t coeffdata   = vld1q_f32(erffdata);
    static const float32x4_t onev{ vdupq_n_f32(1.0f) };

    uint32x4_t selector = vcltzq_f32(x);

    float32x4_t absx  = vabsq_f32(x);
    float32x4_t absx2 = vmulq_f32(x, x);
    float32x4_t absx3 = vmulq_f32(absx2, absx);
    float32x4_t absx4 = vmulq_f32(absx2, absx2);

    float32x4_t denom = onev;
    denom             = vfmaq_laneq_f32(denom, absx, coeffdata, 0);
    denom             = vfmaq_laneq_f32(denom, absx2, coeffdata, 1);
    denom             = vfmaq_laneq_f32(denom, absx3, coeffdata, 2);
    denom             = vfmaq_laneq_f32(denom, absx4, coeffdata, 3);

    denom = vmulq_f32(denom, denom);
    denom = vmulq_f32(denom, denom);

    float32x4_t fract = onev;
    fract             = vdivq_f32(fract, denom);

    float32x4_t result = onev;
    result             = vsubq_f32(result, fract);

    float32x4_t inverse = vnegq_f32(result);

    result = vbslq_f32(selector, inverse, result);

    return result;
}
#endif // #ifdef __aarch64__

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
    static const float32x4_t CONST_THR      = vdupq_n_f32(5.e-3);
    static const float32x4_t CONST_1_3      = vdupq_n_f32(0.3333333f);

    float32x4_t x = vminq_f32(vmaxq_f32(val, CONST_MIN_TANH), CONST_MAX_TANH);
    // x * (1 - x^2/3) if |x| < 5.e-3 or (exp2x - 1) / (exp2x + 1) otherwise
    float32x4_t exp2x = vbslq_f32(vcgtq_f32(vabsq_f32(x), CONST_THR), vexpq_f32(vmulq_f32(CONST_2, x)), vmulq_f32(x, x));
    float32x4_t num   = vbslq_f32(vcgtq_f32(vabsq_f32(x), CONST_THR), vsubq_f32(exp2x, CONST_1), vmulq_f32(CONST_1_3, exp2x));
    float32x4_t den   = vbslq_f32(vcgtq_f32(vabsq_f32(x), CONST_THR), vaddq_f32(exp2x, CONST_1), vsubq_f32(CONST_1, num));
    float32x4_t tanh  = vbslq_f32(vcgtq_f32(vabsq_f32(x), CONST_THR), vmulq_f32(num, vinvq_f32(den)), vmulq_f32(x, den));
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

inline int32x4_t rounding_divide_by_pow2(int32x4_t x, int32x4_t exponent)
{
    const int32x4_t shift_vec  = vnegq_s32(exponent);
    const int32x4_t fixup      = vshrq_n_s32(vandq_s32(x, shift_vec), 31);
    const int32x4_t fixed_up_x = vqaddq_s32(x, fixup);
    return vrshlq_s32(fixed_up_x, shift_vec);
}

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

inline float32x4x4_t convert_uint8x16_to_float32x4x4(const uint8x16_t &in)
{
    float32x4x4_t out;

    const auto tmp1 = vmovl_u8(vget_low_u8(in));
    out.val[0]      = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp1)));
    out.val[1]      = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp1)));

    const auto tmp2 = vmovl_u8(vget_high_u8(in));
    out.val[2]      = vcvtq_f32_u32(vmovl_u16(vget_low_u16(tmp2)));
    out.val[3]      = vcvtq_f32_u32(vmovl_u16(vget_high_u16(tmp2)));
    return out;
}

inline float32x4x4_t convert_int8x16_to_float32x4x4(const int8x16_t &in)
{
    float32x4x4_t out;

    const auto tmp1 = vmovl_s8(vget_low_s8(in));
    out.val[0]      = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp1)));
    out.val[1]      = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp1)));

    const auto tmp2 = vmovl_s8(vget_high_s8(in));
    out.val[2]      = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp2)));
    out.val[3]      = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp2)));
    return out;
}

template <>
inline float32x4x4_t convert_to_float32x4x4(const uint8x16_t &in)
{
    return convert_uint8x16_to_float32x4x4(in);
}

template <>
inline float32x4x4_t convert_to_float32x4x4(const int8x16_t &in)
{
    return convert_int8x16_to_float32x4x4(in);
}

inline void convert_float32x4x3_to_uint8x8x3(const float32x4x3_t &in1, const float32x4x3_t &in2, uint8x8x3_t &out)
{
    out.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(in1.val[0])),
                                         vqmovn_u32(vcvtq_u32_f32(in2.val[0]))));
    out.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(in1.val[1])),
                                         vqmovn_u32(vcvtq_u32_f32(in2.val[1]))));
    out.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vcvtq_u32_f32(in1.val[2])),
                                         vqmovn_u32(vcvtq_u32_f32(in2.val[2]))));
}

inline void convert_float32x4x4_to_uint8x16(const float32x4x4_t &in, uint8x16_t &out)
{
    const auto low = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(in.val[0])),
                                   vqmovn_u32(vcvtq_u32_f32(in.val[1])));
    const auto high = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(in.val[2])),
                                   vqmovn_u32(vcvtq_u32_f32(in.val[3])));
    out = vcombine_u8(vqmovn_u16(low), vqmovn_u16(high));
}

inline void convert_float32x4x4_to_int8x16(const float32x4x4_t &in, int8x16_t &out)
{
    const auto low = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(in.val[0])),
                                   vqmovn_s32(vcvtq_s32_f32(in.val[1])));
    const auto high = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(in.val[2])),
                                   vqmovn_s32(vcvtq_s32_f32(in.val[3])));
    out = vcombine_s8(vqmovn_s16(low), vqmovn_s16(high));
}

template <>
inline uint8x16_t convert_float_to_int<float32x4x4_t, uint8x16_t>(const float32x4x4_t &in)
{
    uint8x16_t out;
    convert_float32x4x4_to_uint8x16(in, out);
    return out;
}

template <>
inline float32x4x4_t convert_int_to_float<float32x4x4_t, uint8x16_t>(const uint8x16_t &in)
{
    return convert_uint8x16_to_float32x4x4(in);
}

template <>
inline int8x16_t convert_float_to_int<float32x4x4_t, int8x16_t>(const float32x4x4_t &in)
{
    int8x16_t out;
    convert_float32x4x4_to_int8x16(in, out);
    return out;
}

template <>
inline float32x4x4_t convert_int_to_float<float32x4x4_t, int8x16_t>(const int8x16_t &in)
{
    return convert_int8x16_to_float32x4x4(in);
}

inline float vreduce(const float32x4_t &v)
{
    const float32x2_t v0    = vget_high_f32(v);
    const float32x2_t v1    = vget_low_f32(v);
    const float32x2_t v_out = vadd_f32(v0, v1);

    const float a = vget_lane_f32(v_out, 0);
    const float b = vget_lane_f32(v_out, 1);

    return a + b;
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

inline float16x4_t vtanh_rational_approx_f16(float16x4_t x16)
{
    // Calculate rational approximation part of tanh exactly on a half-register of F16 by using F32s
    // Note: doesn't handle overflows, needs truncating at |x| = 4.508
    const float32x4_t x = vcvt_f32_f16(x16);

    const float32x4_t ONE = vdupq_n_f32(1.0f);
    const float32x4_t C1  = vdupq_n_f32(0.43760237f);
    const float32x4_t C2  = vdupq_n_f32(0.104402f);
    const float32x4_t C3  = vdupq_n_f32(0.013442706f);
    const float32x4_t C4  = vdupq_n_f32(0.00073561433f);

    const float32x4_t x2 = vmulq_f32(x, x);

    // Denominator polynomial 1 + C1*x^2 + C3*x^4
    float32x4_t denom = vfmaq_f32(C1, C3, x2);
    denom             = vfmaq_f32(ONE, x2, denom);

    // Numerator polynomial x*(1 + C2*x^2 + C4*x^4)
    float32x4_t numer = vfmaq_f32(C2, C4, x2);
    numer             = vfmaq_f32(ONE, x2, numer);
    numer             = vmulq_f32(numer, x);

    return vcvt_f16_f32(vdivq_f32(numer, denom));
}

inline float16x8_t vtanhq_f16(float16x8_t x)
{
    // Split into high/low and use rational approximation on both parts exactly
    const float16x8_t tanh = vcombine_f16(vtanh_rational_approx_f16(vget_low_f16(x)),
                                          vtanh_rational_approx_f16(vget_high_f16(x)));

    // tanh(x) == sign(x) to F16 precision for |x| >= 4.508, use sign after this
    const float16x8_t ONE      = vdupq_n_f16(1.0f);
    const float16x8_t MAX_X    = vdupq_n_f16(4.508f);
    const auto        at_limit = vcageq_f16(x, MAX_X); // |x| >= 4.508
    const float16x8_t sign_x   = vbslq_f16(vclezq_f16(x), -ONE, ONE);
    return vbslq_f16(at_limit, sign_x, tanh);
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
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

    const float16x8_t res = vcombine_f16(vcvt_f16_f32(vexpq_f32(x_low)), vcvt_f16_f32(vexpq_f32(x_high)));
    return res;
}

#ifdef __aarch64__
inline float16x8_t verfq_f16(float16x8_t x)
{
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

    const float16x8_t res = vcombine_f16(vcvt_f16_f32(verfq_f32(x_low)), vcvt_f16_f32(verfq_f32(x_high)));
    return res;
}
#endif // #ifdef __aarch64__

inline float16x8_t vlogq_f16(float16x8_t x)
{
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

    const float16x8_t res = vcombine_f16(vcvt_f16_f32(vlogq_f32(x_low)), vcvt_f16_f32(vlogq_f32(x_high)));
    return res;
}

inline float16x8_t vpowq_f16(float16x8_t val, float16x8_t n)
{
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

inline float16_t vreduce(const float16x8_t &v)
{
    const float16x4_t v0    = vget_high_f16(v);
    const float16x4_t v1    = vget_low_f16(v);
    const float16x4_t v_out = vadd_f16(v0, v1);

    const float16_t a = vget_lane_f16(v_out, 0);
    const float16_t b = vget_lane_f16(v_out, 1);
    const float16_t c = vget_lane_f16(v_out, 2);
    const float16_t d = vget_lane_f16(v_out, 3);

    return a + b + c + d;
}
#endif /* DOXYGEN_SKIP_THIS */
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
