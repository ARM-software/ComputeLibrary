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

inline float32x4_t vfloorq_f32(float32x4_t val)
{
    static const float32x4_t CONST_1 = vdupq_n_f32(1.f);

    const int32x4_t   z = vcvtq_s32_f32(val);
    const float32x4_t r = vcvtq_f32_s32(z);

    return vbslq_f32(vcgtq_f32(r, val), vsubq_f32(r, CONST_1), r);
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* Exponent polynomial coefficients */
const std::array<float16x8_t, 8> exp_tab_f16 =
{
    {
        vdupq_n_f16(1.f),
        vdupq_n_f16(0.0416598916054f),
        vdupq_n_f16(0.500000596046f),
        vdupq_n_f16(0.0014122662833f),
        vdupq_n_f16(1.00000011921f),
        vdupq_n_f16(0.00833693705499f),
        vdupq_n_f16(0.166665703058f),
        vdupq_n_f16(0.000195780929062f),
    }
};

/* Logarithm polynomial coefficients */
const std::array<float16x8_t, 8> log_tab_f16 =
{
    {
        vdupq_n_f16(-2.29561495781f),
        vdupq_n_f16(-2.47071170807f),
        vdupq_n_f16(-5.68692588806f),
        vdupq_n_f16(-0.165253549814f),
        vdupq_n_f16(5.17591238022f),
        vdupq_n_f16(0.844007015228f),
        vdupq_n_f16(4.58445882797f),
        vdupq_n_f16(0.0141278216615f),
    }
};

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
    static const float16x8_t CONST_LN2          = vdupq_n_f16(0.6931471805f); // ln(2)
    static const float16x8_t CONST_INV_LN2      = vdupq_n_f16(1.4426950408f); // 1/ln(2)
    static const float16x8_t CONST_0            = vdupq_n_f16(0.f);
    static const int16x8_t   CONST_NEGATIVE_126 = vdupq_n_s16(-126);

    // Perform range reduction [-log(2),log(2)]
    const int16x8_t   m   = vcvtq_s16_f16(vmulq_f16(x, CONST_INV_LN2));
    const float16x8_t val = vsubq_f16(x, vmulq_f16(vcvtq_f16_s16(m), CONST_LN2));

    // Polynomial Approximation
    float16x8_t poly = vtaylor_polyq_f16(val, exp_tab_f16);

    // Reconstruct
    poly = vreinterpretq_f16_s16(vqaddq_s16(vreinterpretq_s16_f16(poly), vqshlq_n_s16(m, 9)));
    poly = vbslq_f16(vcltq_s16(m, CONST_NEGATIVE_126), CONST_0, poly);

    return poly;
}

inline float16x8_t vlogq_f16(float16x8_t x)
{
    static const int16x8_t   CONST_127 = vdupq_n_s16(127);           // 127
    static const float16x8_t CONST_LN2 = vdupq_n_f16(0.6931471805f); // ln(2)

    // Extract exponent
    const int16x8_t   m   = vsubq_s16(vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_f16(x), 9)), CONST_127);
    const float16x8_t val = vreinterpretq_f16_s16(vsubq_s16(vreinterpretq_s16_f16(x), vshlq_n_s16(m, 9)));

    // Polynomial Approximation
    float16x8_t poly = vtaylor_polyq_f16(val, log_tab_f16);

    // Reconstruct
    poly = vaddq_f16(poly, vmulq_f16(vcvtq_f16_s16(m), CONST_LN2));

    return poly;
}

inline float16x8_t vpowq_f16(float16x8_t val, float16x8_t n)
{
    return vexpq_f16(vmulq_f16(n, vlogq_f16(val)));
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
} // namespace arm_compute
