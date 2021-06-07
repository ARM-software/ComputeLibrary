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
namespace arm_compute
{
#if defined(ARM_COMPUTE_ENABLE_SVE2)
inline svuint8_t svmla_qasymm8_z(svbool_t pg, svuint8_t vd, svfloat32_t vs, svfloat32_t vo)
{
    // Convert uint8 vectors to uint16 vectors
    auto vd_low_u16  = svmovlb_u16(vd);
    auto vd_high_u16 = svmovlt_u16(vd);

    // Convert uint16 vectors to uint32 vectors
    auto A_u32 = svmovlb_u32(vd_low_u16);
    auto B_u32 = svmovlt_u32(vd_low_u16);
    auto C_u32 = svmovlb_u32(vd_high_u16);
    auto D_u32 = svmovlt_u32(vd_high_u16);

    // Convert uint32 vectors to float32 vectors
    auto A_f32 = svcvt_f32_u32_z(pg, A_u32);
    auto B_f32 = svcvt_f32_u32_z(pg, B_u32);
    auto C_f32 = svcvt_f32_u32_z(pg, C_u32);
    auto D_f32 = svcvt_f32_u32_z(pg, D_u32);

    // vd = vd*vs + vo
    A_f32 = svmla_f32_z(pg, vo, A_f32, vs);
    B_f32 = svmla_f32_z(pg, vo, B_f32, vs);
    C_f32 = svmla_f32_z(pg, vo, C_f32, vs);
    D_f32 = svmla_f32_z(pg, vo, D_f32, vs);

    // Convert float32 vectors to uint32 vectors
    A_u32 = svcvt_u32_f32_z(pg, A_f32);
    B_u32 = svcvt_u32_f32_z(pg, B_f32);
    C_u32 = svcvt_u32_f32_z(pg, C_f32);
    D_u32 = svcvt_u32_f32_z(pg, D_f32);

    // Convert uint32 vectors to uint16 vectors (with saturation)
    vd_low_u16  = svqxtnt_u32(svqxtnb_u32(A_u32), B_u32);
    vd_high_u16 = svqxtnt_u32(svqxtnb_u32(C_u32), D_u32);

    // convert uint16 vectors to uint8 vectors (with saturation)
    const auto res = svqxtnt_u16(svqxtnb_u16(vd_low_u16), vd_high_u16);
    return res;
}

inline svint8_t svmla_qasymm8_signed_z(svbool_t pg, svint8_t vd, svfloat32_t vs, svfloat32_t vo)
{
    // Convert uint8 vectors to int16 vectors
    auto vd_low_s16  = svmovlb_s16(vd);
    auto vd_high_s16 = svmovlt_s16(vd);

    // Convert int16 vectors to int32 vectors
    auto A_s32 = svmovlb_s32(vd_low_s16);
    auto B_s32 = svmovlt_s32(vd_low_s16);
    auto C_s32 = svmovlb_s32(vd_high_s16);
    auto D_s32 = svmovlt_s32(vd_high_s16);

    // Convert int32 vectors to float32 vectors
    auto A_f32 = svcvt_f32_s32_z(pg, A_s32);
    auto B_f32 = svcvt_f32_s32_z(pg, B_s32);
    auto C_f32 = svcvt_f32_s32_z(pg, C_s32);
    auto D_f32 = svcvt_f32_s32_z(pg, D_s32);

    // vd = vd*vs + vo
    A_f32 = svmla_f32_z(pg, vo, A_f32, vs);
    B_f32 = svmla_f32_z(pg, vo, B_f32, vs);
    C_f32 = svmla_f32_z(pg, vo, C_f32, vs);
    D_f32 = svmla_f32_z(pg, vo, D_f32, vs);

    // Convert float32 vectors to int32 vectors
    A_s32 = svcvt_s32_f32_z(pg, A_f32);
    B_s32 = svcvt_s32_f32_z(pg, B_f32);
    C_s32 = svcvt_s32_f32_z(pg, C_f32);
    D_s32 = svcvt_s32_f32_z(pg, D_f32);

    // Convert uint32 vectors to uint16 vectors (with saturation)
    vd_low_s16  = svqxtnt_s32(svqxtnb_s32(A_s32), B_s32);
    vd_high_s16 = svqxtnt_s32(svqxtnb_s32(C_s32), D_s32);

    // convert uint16 vectors to uint8 vectors (with saturation)
    const auto res = svqxtnt_s16(svqxtnb_s16(vd_low_s16), vd_high_s16);
    return res;
}
#endif /* (ARM_COMPUTE_ENABLE_SVE2) */
} // namespace arm_compute
