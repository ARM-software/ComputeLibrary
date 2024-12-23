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

#ifndef ACL_SRC_CORE_NEON_SVE2MATH_INL
#define ACL_SRC_CORE_NEON_SVE2MATH_INL

#ifdef ARM_COMPUTE_ENABLE_SVE2

namespace arm_compute
{
inline svfloat16_t svexp_f16_z_sve2(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;

    bottom = svexp_f32_z(pg, bottom);
    top    = svexp_f32_z(pg_top, top);

    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
}

inline svfloat16_t svlog_f16_z_sve2(svbool_t pg, svfloat16_t x)
{
    auto bottom = svcvt_f32_z(pg, x);
    auto top    = svcvtlt_f32_x(pg, x);
    auto pg_top = pg;

    bottom = svlog_f32_z(pg, bottom);
    top    = svlog_f32_z(pg_top, top);

    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
}

inline svfloat16_t svsin_f16_z_sve2(svbool_t pg, svfloat16_t val)
{
    auto bottom = svcvt_f32_z(pg, val);
    auto top    = svcvtlt_f32_x(pg, val);
    auto pg_top = pg;

    bottom = svsin_f32_z(pg, bottom);
    top    = svsin_f32_z(pg_top, top);

    return svcvtnt_f16_m(svcvt_f16_z(pg, bottom), pg_top, top);
}

inline svfloat16_t svpow_f16_z_sve2(svbool_t pg, svfloat16_t a, svfloat16_t b)
{
    auto a_bottom = svcvt_f32_z(pg, a);
    auto b_bottom = svcvt_f32_z(pg, b);

    auto pg_top = pg;
    auto a_top  = svcvtlt_f32_x(pg, a);
    auto b_top  = svcvtlt_f32_x(pg, b);

    auto res_bottom = svpow_f32_z(pg, a_bottom, b_bottom);
    auto res_top    = svpow_f32_z(pg_top, a_top, b_top);

    return svcvtnt_f16_m(svcvt_f16_z(pg, res_bottom), pg_top, res_top);
}

template <>
inline svuint8_t convert_float_to_int<svuint8_t>(const svfloat32_t &in_0,
                                                 const svfloat32_t &in_1,
                                                 const svfloat32_t &in_2,
                                                 const svfloat32_t &in_3)
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
inline svint8_t convert_float_to_int<svint8_t>(const svfloat32_t &in_0,
                                               const svfloat32_t &in_1,
                                               const svfloat32_t &in_2,
                                               const svfloat32_t &in_3)
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
} // namespace arm_compute

#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */

#endif // ACL_SRC_CORE_NEON_SVE2MATH_INL
