/*
 * Copyright (c) 2020-2021 Arm Limited.
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

#if defined(__ARM_FEATURE_SVE)

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // M_PI

namespace arm_compute
{
inline svfloat32_t svtaylor_poly_f32_z(svbool_t pg, svfloat32_t x, const std::array<svfloat32_t, 8> &coeffs)
{
    const auto A   = svmla_f32_z(pg, coeffs[0], coeffs[4], x);
    const auto B   = svmla_f32_z(pg, coeffs[2], coeffs[6], x);
    const auto C   = svmla_f32_z(pg, coeffs[1], coeffs[5], x);
    const auto D   = svmla_f32_z(pg, coeffs[3], coeffs[7], x);
    const auto x2  = svmul_f32_z(pg, x, x);
    const auto x4  = svmul_f32_z(pg, x2, x2);
    const auto res = svmla_f32_z(pg, svmla_f32_z(pg, A, B, x2), svmla_f32_z(pg, C, D, x2), x4);
    return res;
}

inline svfloat16_t svtaylor_poly_f16_z(svbool_t pg, svfloat16_t x, const std::array<svfloat16_t, 8> &coeffs)
{
    const auto A   = svmla_f16_z(pg, coeffs[0], coeffs[4], x);
    const auto B   = svmla_f16_z(pg, coeffs[2], coeffs[6], x);
    const auto C   = svmla_f16_z(pg, coeffs[1], coeffs[5], x);
    const auto D   = svmla_f16_z(pg, coeffs[3], coeffs[7], x);
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

inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x)
{
    const auto CONST_LN2          = svdup_n_f32(0.6931471805f); // ln(2)
    const auto CONST_INV_LN2      = svdup_n_f32(1.4426950408f); // 1/ln(2)
    const auto CONST_INF          = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto CONST_MAX_INPUT    = svdup_n_f32(88.7f);
    const auto CONST_0            = svdup_n_f32(0.f);
    const auto CONST_NEGATIVE_126 = svdup_n_s32(-126);

    /** Exponent polynomial coefficients */
    const std::array<svfloat32_t, 8> exp_tab =
    {
        {
            svdup_n_f32(1.f),
            svdup_n_f32(0.0416598916054f),
            svdup_n_f32(0.500000596046f),
            svdup_n_f32(0.0014122662833f),
            svdup_n_f32(1.00000011921f),
            svdup_n_f32(0.00833693705499f),
            svdup_n_f32(0.166665703058f),
            svdup_n_f32(0.000195780929062f),
        }
    };

    // Perform range reduction [-log(2),log(2)]
    auto m   = svcvt_s32_f32_z(pg, svmul_f32_z(pg, x, CONST_INV_LN2));
    auto val = svmls_f32_z(pg, x, svcvt_f32_s32_z(pg, m), CONST_LN2);

    // Polynomial Approximation
    auto poly = svtaylor_poly_f32_z(pg, val, exp_tab);

    // Reconstruct
    poly = svreinterpret_f32_s32(svqadd_s32(svreinterpret_s32_f32(poly), svlsl_n_s32_z(pg, m, 23)));

    // Handle underflow
    svbool_t ltpg = svcmplt_s32(pg, m, CONST_NEGATIVE_126);
    poly          = svsel_f32(ltpg, CONST_0, poly);

    // Handle overflow
    svbool_t gtpg = svcmpgt_f32(pg, x, CONST_MAX_INPUT);
    poly          = svsel_f32(gtpg, CONST_INF, poly);

    return poly;
}

inline svfloat16_t svexp_f16_z(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
#if defined(__ARM_FEATURE_SVE2)
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;
#else  /* defined(__ARM_FEATURE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(x))));
#endif /* defined(__ARM_FEATURE_SVE2) */

    bottom = svexp_f32_z(pg, bottom);
    top    = svexp_f32_z(pg_top, top);

#if defined(__ARM_FEATURE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(__ARM_FEATURE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(__ARM_FEATURE_SVE2) */
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
    const std::array<svfloat32_t, 8> log_tab =
    {
        {
            svdup_n_f32(-2.29561495781f),
            svdup_n_f32(-2.47071170807f),
            svdup_n_f32(-5.68692588806f),
            svdup_n_f32(-0.165253549814f),
            svdup_n_f32(5.17591238022f),
            svdup_n_f32(0.844007015228f),
            svdup_n_f32(4.58445882797f),
            svdup_n_f32(0.0141278216615f),
        }
    };

    const auto CONST_127 = svdup_n_s32(127);           // 127
    const auto CONST_LN2 = svdup_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    auto m   = svsub_s32_z(pg, svasr_n_s32_z(pg, svreinterpret_s32_f32(x), 23), CONST_127);
    auto val = svreinterpret_f32_s32(svsub_s32_z(pg, svreinterpret_s32_f32(x), svlsl_n_s32_z(pg, m, 23)));

    // Polynomial Approximation
    auto poly = svtaylor_poly_f32_z(pg, val, log_tab);

    // Reconstruct
    poly = svmla_f32_z(pg, poly, svcvt_f32_s32_z(pg, m), CONST_LN2);

    return poly;
}

inline svfloat16_t svlog_f16_z(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
#if defined(__ARM_FEATURE_SVE2)
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;
#else  /* defined(__ARM_FEATURE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(x))));
#endif /* defined(__ARM_FEATURE_SVE2) */

    bottom = svlog_f32_z(pg, bottom);
    top    = svlog_f32_z(pg_top, top);

#if defined(__ARM_FEATURE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(__ARM_FEATURE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(__ARM_FEATURE_SVE2) */
}

inline svfloat32_t svsin_f32_z(svbool_t pg, svfloat32_t val)
{
    using ScalarType = float;
    using IntType    = u32;

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
#if defined(__ARM_FEATURE_SVE2)
    auto top    = svcvtlt_f32_x(pg, val);
    auto pg_top = pg;
#else  /* defined(__ARM_FEATURE_SVE2) */
    auto pg_top = svptrue_b16();
    auto top    = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(val))));
#endif /* defined(__ARM_FEATURE_SVE2) */

    bottom = svsin_f32_z(pg, bottom);
    top    = svsin_f32_z(pg_top, top);

#if defined(__ARM_FEATURE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
#else  /* defined(__ARM_FEATURE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, bottom), svcvt_f16_z(pg_top, top));
#endif /* defined(__ARM_FEATURE_SVE2) */
}

inline svfloat32_t svpow_f32_z(svbool_t pg, svfloat32_t a, svfloat32_t b)
{
    return svexp_f32_z(pg, svmul_z(pg, b, svlog_f32_z(pg, a)));
}

inline svfloat16_t svpow_f16_z(svbool_t pg, svfloat16_t a, svfloat16_t b)
{
    auto a_bottom = svcvt_f32_z(pg, a);
    auto b_bottom = svcvt_f32_z(pg, b);

#if defined(__ARM_FEATURE_SVE2)
    auto pg_top = pg;
    auto a_top  = svcvtlt_f32_x(pg, a);
    auto b_top  = svcvtlt_f32_x(pg, b)
#else  /* defined(__ARM_FEATURE_SVE2) */
    auto pg_top = svptrue_b16();
    auto a_top  = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(a))));
    auto b_top  = svcvt_f32_z(pg_top, svreinterpret_f16(svrevh_z(svptrue_b16(), svreinterpret_u32(b))));
#endif /* defined(__ARM_FEATURE_SVE2) */

                  auto res_bottom = svpow_f32_z(pg, a_bottom, b_bottom);
    auto     res_top    = svpow_f32_z(pg_top, a_top, b_top);

#if defined(__ARM_FEATURE_SVE2)
    return svcvtnt_f16_m(svcvt_f16_z(pg, res_bottom), pg_top, res_top);
#else  /* defined(__ARM_FEATURE_SVE2) */
    return svtrn1(svcvt_f16_z(pg, res_bottom), svcvt_f16_z(pg_top, res_top));
#endif /* defined(__ARM_FEATURE_SVE2) */
}

} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_SVE) */
