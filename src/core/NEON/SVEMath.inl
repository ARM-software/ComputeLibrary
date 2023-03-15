/*
 * Copyright (c) 2020-2023 Arm Limited.
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
#include <limits>

#if defined(__ARM_FEATURE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE)

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

namespace arm_compute
{
inline svfloat32_t svtaylor_poly_f32_z(svbool_t pg, svfloat32_t x, svfloat32_t coeff_1, svfloat32_t coeff_2, svfloat32_t coeff_3,
                                       svfloat32_t coeff_4, svfloat32_t coeff_5, svfloat32_t coeff_6, svfloat32_t coeff_7, svfloat32_t coeff_8)
{
    const auto A   = svmla_f32_z(pg, coeff_1, coeff_5, x);
    const auto B   = svmla_f32_z(pg, coeff_3, coeff_7, x);
    const auto C   = svmla_f32_z(pg, coeff_2, coeff_6, x);
    const auto D   = svmla_f32_z(pg, coeff_4, coeff_8, x);
    const auto x2  = svmul_f32_z(pg, x, x);
    const auto x4  = svmul_f32_z(pg, x2, x2);
    const auto res = svmla_f32_z(pg, svmla_f32_z(pg, A, B, x2), svmla_f32_z(pg, C, D, x2), x4);
    return res;
}

inline svfloat16_t svtaylor_poly_f16_z(svbool_t pg, svfloat16_t x, svfloat16_t coeff_1, svfloat16_t coeff_2, svfloat16_t coeff_3,
                                       svfloat16_t coeff_4, svfloat16_t coeff_5, svfloat16_t coeff_6, svfloat16_t coeff_7, svfloat16_t coeff_8)
{
    const auto A   = svmla_f16_z(pg, coeff_1, coeff_5, x);
    const auto B   = svmla_f16_z(pg, coeff_3, coeff_7, x);
    const auto C   = svmla_f16_z(pg, coeff_2, coeff_6, x);
    const auto D   = svmla_f16_z(pg, coeff_4, coeff_8, x);
    const auto x2  = svmul_f16_z(pg, x, x);
    const auto x4  = svmul_f16_z(pg, x2, x2);
    const auto res = svmla_f16_z(pg, svmla_f16_z(pg, A, B, x2), svmla_f16_z(pg, C, D, x2), x4);
    return res;
}

inline svfloat16_t svinv_f16_z(svbool_t pg, svfloat16_t x)
{
    auto recip = svrecpe_f16(x);
    recip      = svmul_f16_z(pg, svrecps_f16(x, recip), recip);
    recip      = svmul_f16_z(pg, svrecps_f16(x, recip), recip);
    return recip;
}

inline svfloat32_t svinv_f32_z(svbool_t pg, svfloat32_t x)
{
    auto recip = svrecpe_f32(x);
    recip      = svmul_f32_z(pg, svrecps_f32(x, recip), recip);
    recip      = svmul_f32_z(pg, svrecps_f32(x, recip), recip);
    return recip;
}

static const uint32_t svexp_f32_coeff[] = {
    0x3f7ffff6, // x^1: 0x1.ffffecp-1f
    0x3efffedb, // x^2: 0x1.fffdb6p-2f
    0x3e2aaf33, // x^3: 0x1.555e66p-3f
    0x3d2b9f17, // x^4: 0x1.573e2ep-5f
    0x3c072010, // x^5: 0x1.0e4020p-7f
};

inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x)
{
    const auto c1 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[0]));
    const auto c2 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[1]));
    const auto c3 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[2]));
    const auto c4 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[3]));
    const auto c5 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[4]));

    const auto shift   = svreinterpret_f32_u32(svdup_n_u32(0x4b00007f));  // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = svreinterpret_f32_u32(svdup_n_u32(0x3fb8aa3b));  // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi  = svreinterpret_f32_u32(svdup_n_u32(0xbf317200));  // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo  = svreinterpret_f32_u32(svdup_n_u32(0xb5bfbe8e));  // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f);  // Approximately ln(2^127.5)
    const auto zero      = svdup_n_f32(0.f);
    const auto min_input = svdup_n_f32(-86.64f);  // Approximately ln(2^-125)

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
    const auto z = svmla_f32_z(pg, shift, x, inv_ln2);
    const auto n = svsub_f32_z(pg, z, shift);
    const auto scale = svreinterpret_f32_u32(svlsl_n_u32_z(pg, svreinterpret_u32_f32(z), 23));  // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in term of accuracy and performance.
    const auto r_hi = svmla_f32_z(pg, x, n, neg_ln2_hi);
    const auto r = svmla_f32_z(pg, r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = svmul_f32_z(pg, r, r);

    const auto p1 = svmul_f32_z(pg, c1, r);
    const auto p23 = svmla_f32_z(pg, c2, c3, r);
    const auto p45 = svmla_f32_z(pg, c4, c5, r);
    const auto p2345 = svmla_f32_z(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_z(pg, p1, p2345, r2);

    auto poly = svmla_f32_z(pg, scale, p12345, scale);

    // Handle underflow and overflow.
    poly = svsel_f32(svcmplt_f32(pg, x, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, x, max_input), inf, poly);

    return poly;
}

inline svfloat16_t svexp_f16_z(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
#if defined(ARM_COMPUTE_ENABLE_SVE2)
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(x))));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

    bottom = svexp_f32_z(pg, bottom);
    top    = svexp_f32_z(pg_top, top);

#if defined(ARM_COMPUTE_ENABLE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
}

inline svfloat32_t svtanh_f32_z(svbool_t pg, svfloat32_t val)
{
    const svfloat32_t CONST_1        = svdup_n_f32(1.f);
    const svfloat32_t CONST_2        = svdup_n_f32(2.f);
    const svfloat32_t CONST_MIN_TANH = svdup_n_f32(-10.f);
    const svfloat32_t CONST_MAX_TANH = svdup_n_f32(10.f);

    svfloat32_t x     = svmin_f32_z(pg, svmax_f32_z(pg, val, CONST_MIN_TANH), CONST_MAX_TANH);
    svfloat32_t exp2x = svexp_f32_z(pg, svmul_f32_z(pg, CONST_2, x));
    svfloat32_t num   = svsub_f32_z(pg, exp2x, CONST_1);
    svfloat32_t den   = svadd_f32_z(pg, exp2x, CONST_1);
    svfloat32_t tanh  = svdiv_f32_z(pg, num, den);
    return tanh;
}

inline svfloat16_t svtanh_f16_z(svbool_t pg, svfloat16_t val)
{
    const svfloat16_t CONST_1        = svdup_n_f16(1.f);
    const svfloat16_t CONST_2        = svdup_n_f16(2.f);
    const svfloat16_t CONST_MIN_TANH = svdup_n_f16(-10.f);
    const svfloat16_t CONST_MAX_TANH = svdup_n_f16(10.f);

    const svfloat16_t x     = svmin_f16_z(pg, svmax_f16_z(pg, val, CONST_MIN_TANH), CONST_MAX_TANH);
    const svfloat16_t exp2x = svexp_f16_z(pg, svmul_f16_z(pg, CONST_2, x));
    const svfloat16_t num   = svsub_f16_z(pg, exp2x, CONST_1);
    const svfloat16_t den   = svadd_f16_z(pg, exp2x, CONST_1);
    const svfloat16_t tanh  = svdiv_f16_z(pg, num, den);
    return tanh;
}

inline svfloat32_t svlog_f32_z(svbool_t pg, svfloat32_t x)
{
    /** Logarithm polynomial coefficients */
    const svfloat32_t log_tab_1 = svdup_n_f32(-2.29561495781f);
    const svfloat32_t log_tab_2 = svdup_n_f32(-2.47071170807f);
    const svfloat32_t log_tab_3 = svdup_n_f32(-5.68692588806f);
    const svfloat32_t log_tab_4 = svdup_n_f32(-0.165253549814f);
    const svfloat32_t log_tab_5 = svdup_n_f32(5.17591238022f);
    const svfloat32_t log_tab_6 = svdup_n_f32(0.844007015228f);
    const svfloat32_t log_tab_7 = svdup_n_f32(4.58445882797f);
    const svfloat32_t log_tab_8 = svdup_n_f32(0.0141278216615f);

    const auto CONST_127 = svdup_n_s32(127);           // 127
    const auto CONST_LN2 = svdup_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    auto m   = svsub_s32_z(pg, svasr_n_s32_z(pg, svreinterpret_s32_f32(x), 23), CONST_127);
    auto val = svreinterpret_f32_s32(svsub_s32_z(pg, svreinterpret_s32_f32(x), svlsl_n_s32_z(pg, m, 23)));

    // Polynomial Approximation
    auto poly = svtaylor_poly_f32_z(pg, val, log_tab_1, log_tab_2, log_tab_3, log_tab_4, log_tab_5, log_tab_6, log_tab_7, log_tab_8);

    // Reconstruct
    poly = svmla_f32_z(pg, poly, svcvt_f32_s32_z(pg, m), CONST_LN2);

    return poly;
}

inline svfloat16_t svlog_f16_z(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
#if defined(ARM_COMPUTE_ENABLE_SVE2)
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(x))));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

    bottom = svlog_f32_z(pg, bottom);
    top    = svlog_f32_z(pg_top, top);

#if defined(ARM_COMPUTE_ENABLE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
}

inline svfloat32_t svsin_f32_z(svbool_t pg, svfloat32_t val)
{
    using ScalarType = float;
    using IntType    = uint32_t;

    constexpr float te_sin_coeff2 = 0.166666666666f; // 1/(2*3)
    constexpr float te_sin_coeff3 = 0.05f;           // 1/(4*5)
    constexpr float te_sin_coeff4 = 0.023809523810f; // 1/(6*7)
    constexpr float te_sin_coeff5 = 0.013888888889f; // 1/(8*9)

    const auto pi_v   = wrapper::svdup_n(ScalarType(M_PI));
    const auto pio2_v = wrapper::svdup_n(ScalarType(M_PI / 2));
    const auto ipi_v  = wrapper::svdup_n(ScalarType(1 / M_PI));

    //Find positive or negative
    const auto c_v    = svabs_z(pg, wrapper::svcvt_z<int32_t>(pg, svmul_z(pg, val, ipi_v)));
    const auto sign_v = svcmple(pg, val, wrapper::svdup_n(ScalarType(0)));
    const auto odd_v  = svcmpne(pg, svand_z(pg, wrapper::svreinterpret<IntType>(c_v), wrapper::svdup_n(IntType(1))), wrapper::svdup_n(IntType(0)));

    auto neg_v = sveor_z(pg, odd_v, sign_v);

    //Modulus a - (n * int(a*(1/n)))
    auto       ma    = svsub_z(pg, svabs_z(pg, val), svmul_z(pg, pi_v, wrapper::svcvt_z<ScalarType>(pg, c_v)));
    const auto reb_v = svcmpge(pg, ma, pio2_v);

    //Rebase a between 0 and pi/2
    ma = svsel(reb_v, svsub_z(pg, pi_v, ma), ma);

    //Taylor series
    const auto ma2 = svmul_z(pg, ma, ma);

    //2nd elem: x^3 / 3!
    auto elem = svmul_z(pg, svmul_z(pg, ma, ma2), wrapper::svdup_n(ScalarType(te_sin_coeff2)));
    auto res  = svsub_z(pg, ma, elem);

    //3rd elem: x^5 / 5!
    elem = svmul_z(pg, svmul_z(pg, elem, ma2), wrapper::svdup_n(ScalarType(te_sin_coeff3)));
    res  = svadd_z(pg, res, elem);

    //4th elem: x^7 / 7!float32x2_t vsin_f32(float32x2_t val)
    elem = svmul_z(pg, svmul_z(pg, elem, ma2), wrapper::svdup_n(ScalarType(te_sin_coeff4)));
    res  = svsub_z(pg, res, elem);

    //5th elem: x^9 / 9!
    elem = svmul_z(pg, svmul_z(pg, elem, ma2), wrapper::svdup_n(ScalarType(te_sin_coeff5)));
    res  = svadd_z(pg, res, elem);

    //Change of sign
    res = svneg_m(res, neg_v, res);
    return res;
}

inline svfloat16_t svsin_f16_z(svbool_t pg, svfloat16_t val)
{
    auto bottom = svcvt_f32_z(pg, val);
#if defined(ARM_COMPUTE_ENABLE_SVE2)
    auto top    = svcvtlt_f32_x(pg, val);
    auto pg_top = pg;
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(val))));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

    bottom = svsin_f32_z(pg, bottom);
    top    = svsin_f32_z(pg_top, top);

#if defined(ARM_COMPUTE_ENABLE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
}

inline svfloat32_t svpow_f32_z(svbool_t pg, svfloat32_t a, svfloat32_t b)
{
    return svexp_f32_z(pg, svmul_z(pg, b, svlog_f32_z(pg, a)));
}

inline svfloat16_t svpow_f16_z(svbool_t pg, svfloat16_t a, svfloat16_t b)
{
    auto a_bottom = svcvt_f32_z(pg, a);
    auto b_bottom = svcvt_f32_z(pg, b);

#if defined(ARM_COMPUTE_ENABLE_SVE2)
    auto pg_top = pg;
    auto a_top  = svcvtlt_f32_x(pg, a);
    auto b_top  = svcvtlt_f32_x(pg, b);
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    auto pg_top = svptrue_b16();
    auto a_top  = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(a))));
    auto b_top  = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(b))));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

    auto res_bottom = svpow_f32_z(pg, a_bottom, b_bottom);
    auto res_top    = svpow_f32_z(pg_top, a_top, b_top);

#if defined(ARM_COMPUTE_ENABLE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, res_bottom), pg_top, res_top);
#else  /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, res_bottom), svcvt_f16_z(pg_top, res_top));
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
}

#if defined(ARM_COMPUTE_ENABLE_SVE2)
template <>
inline svuint8_t convert_float_to_int<svuint8_t>(const svfloat32_t &in_0, const svfloat32_t &in_1, const svfloat32_t &in_2, const svfloat32_t &in_3)
{
    svuint8_t  out;
    const auto all_true_pg = svptrue_b32();
    auto       tmp_0       = svcvt_u32_f32_z(all_true_pg, in_0);
    auto       tmp_1       = svcvt_u32_f32_z(all_true_pg, in_1);
    auto       tmp_2       = svcvt_u32_f32_z(all_true_pg, in_2);
    auto       tmp_3       = svcvt_u32_f32_z(all_true_pg, in_3);

    auto tmp_16_0 = svqxtnt_u32(svqxtnb_u32(tmp_0), tmp_1);
    auto tmp_16_1 = svqxtnt_u32(svqxtnb_u32(tmp_2), tmp_3);

    auto tmp_16_uzp_0 = svuzp1(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_1 = svuzp2(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_2 = svuzp1(tmp_16_1, tmp_16_1);
    auto tmp_16_uzp_3 = svuzp2(tmp_16_1, tmp_16_1);

    auto pg = svwhilelt_b16_s32(0, svcnth() / 2);

    tmp_16_0 = svsplice(pg, tmp_16_uzp_0, tmp_16_uzp_1);
    tmp_16_1 = svsplice(pg, tmp_16_uzp_2, tmp_16_uzp_3);

    out = svqxtnt_u16(svqxtnb_u16(tmp_16_0), tmp_16_1);

    auto out_uzp_0 = svuzp1(out, out);
    auto out_uzp_1 = svuzp2(out, out);

    pg  = svwhilelt_b8_s32(0, svcntb() / 2);
    out = svsplice(pg, out_uzp_0, out_uzp_1);

    return out;
}

template <>
inline svint8_t convert_float_to_int<svint8_t>(const svfloat32_t &in_0, const svfloat32_t &in_1, const svfloat32_t &in_2, const svfloat32_t &in_3)
{
    svint8_t   out;
    const auto all_true_pg = svptrue_b32();
    auto       tmp_0       = svcvt_s32_f32_z(all_true_pg, in_0);
    auto       tmp_1       = svcvt_s32_f32_z(all_true_pg, in_1);
    auto       tmp_2       = svcvt_s32_f32_z(all_true_pg, in_2);
    auto       tmp_3       = svcvt_s32_f32_z(all_true_pg, in_3);

    auto tmp_16_0 = svqxtnt_s32(svqxtnb_s32(tmp_0), tmp_1);
    auto tmp_16_1 = svqxtnt_s32(svqxtnb_s32(tmp_2), tmp_3);

    auto tmp_16_uzp_0 = svuzp1(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_1 = svuzp2(tmp_16_0, tmp_16_0);
    auto tmp_16_uzp_2 = svuzp1(tmp_16_1, tmp_16_1);
    auto tmp_16_uzp_3 = svuzp2(tmp_16_1, tmp_16_1);

    auto pg = svwhilelt_b16_s32(0, svcnth() / 2);

    tmp_16_0 = svsplice(pg, tmp_16_uzp_0, tmp_16_uzp_1);
    tmp_16_1 = svsplice(pg, tmp_16_uzp_2, tmp_16_uzp_3);

    out = svqxtnt_s16(svqxtnb_s16(tmp_16_0), tmp_16_1);

    auto out_uzp_0 = svuzp1(out, out);
    auto out_uzp_1 = svuzp2(out, out);

    pg  = svwhilelt_b8_s32(0, svcntb() / 2);
    out = svsplice(pg, out_uzp_0, out_uzp_1);

    return out;
}
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

} // namespace arm_compute
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
