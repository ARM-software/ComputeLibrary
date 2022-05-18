/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_SVEASYMM_H
#define ARM_COMPUTE_SVEASYMM_H

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#include "src/core/NEON/SVEMath.h"
#include <arm_sve.h>

namespace arm_compute
{
/** Perform a multiply-accumulate on all components of a QASYMM8 vector
 *
 * vd*vs + vo
 *
 * @param[in] pg Predicate value.
 * @param[in] vd Input vector value in QASYMM8 format
 * @param[in] vs Vector multiplier in F32 format. The multiplier value must be duplicated across all four lanes.
 * @param[in] vo Vector addend in F32 format. The addend value must be duplicated across all four lanes.
 *
 * @return A vector in QASYMM8 format, saturated to fit
 */
svuint8_t svmla_qasymm8_z(svbool_t pg, svuint8_t vd, svfloat32_t vs, svfloat32_t vo);

/** Perform a multiply-accumulate on all components of a QASYMM8_SIGNED vector
 *
 * vd*vs + vo
 *
 * @param[in] pg Predicate value.
 * @param[in] vd Input vector value in QASYMM8_SIGNED format
 * @param[in] vs Vector multiplier in F32 format. The multiplier value must be duplicated across all four lanes.
 * @param[in] vo Vector addend in F32 format. The addend value must be duplicated across all four lanes.
 *
 * @return A vector in QASYMM8_SIGNED format, saturated to fit
 */
svint8_t svmla_qasymm8_signed_z(svbool_t pg, svint8_t vd, svfloat32_t vs, svfloat32_t vo);

/** Dequantize following an asymmetric quantization scheme a sve vector.
 *
 * @param[in] pg     Predicate value.
 * @param[in] qv     Input values to be dequantized.
 * @param[in] scale  Quantization scaling factor.
 * @param[in] offset Zero quantization offset.
 *
 * @return Dequantized values in an sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svuint8_t &qv, float scale, int32_t offset)
{
    const auto          voffset            = svdup_n_s32(offset);
    const auto          vscale             = svdup_n_f32(scale);
    const svfloat32x4_t vdequantized_input = svcreate4_f32(
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svreinterpret_s32_u32(svmovlb_u32(svmovlb_u16(qv))), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svreinterpret_s32_u32(svmovlt_u32(svmovlb_u16(qv))), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svreinterpret_s32_u32(svmovlb_u32(svmovlt_u16(qv))), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svreinterpret_s32_u32(svmovlt_u32(svmovlt_u16(qv))), voffset)), vscale));
    return vdequantized_input;
}

/** Dequantize an sve vector
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in an sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svuint8_t &qv, const UniformQuantizationInfo &qi)
{
    return svdequantize_z(pg, qv, qi.scale, qi.offset);
}

/** Dequantize an sve vector stored as signed asymmetric.
 *
 * @param[in] pg     Predicate value.
 * @param[in] qv     Input values to be dequantized.
 * @param[in] scale  Quantization scaling factor.
 * @param[in] offset Zero quantization offset.
 *
 * @return Dequantized values in a sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svint8_t &qv, float scale, int32_t offset)
{
    const auto          voffset            = svdup_n_s32(offset);
    const auto          vscale             = svdup_n_f32(scale);
    const svfloat32x4_t vdequantized_input = svcreate4_f32(
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlb_s16(qv)), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlb_s16(qv)), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlb_s32(svmovlt_s16(qv)), voffset)), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svsub_s32_z(pg, svmovlt_s32(svmovlt_s16(qv)), voffset)), vscale));

    return vdequantized_input;
}

/** Dequantize an sve vector.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in an sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svint8_t &qv, const UniformQuantizationInfo &qi)
{
    return svdequantize_z(pg, qv, qi.scale, qi.offset);
}

/** Dequantize following symmetric quantization scheme on an sve vector.
 *
 * @param[in] pg     Predicate value.
 * @param[in] qv     Input values to be dequantized.
 * @param[in] vscale Vector containing quantization scaling factors.
 *
 * @return Dequantized values in a sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svint8_t &qv, const svfloat32x4_t vscale)
{
    const svfloat32x4_t vdequantized_input = svcreate4_f32(
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svmovlb_s16(qv))), svget4_f32(vscale, 0)),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svmovlb_s16(qv))), svget4_f32(vscale, 1)),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svmovlt_s16(qv))), svget4_f32(vscale, 2)),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svmovlt_s16(qv))), svget4_f32(vscale, 3)));

    return vdequantized_input;
}

/** Dequantize following a symmetric quantization scheme an sve vector.
 *
 * @param[in] qv    Input values to be dequantized.
 * @param[in] scale Quantization scaling factor.
 *
 * @return Dequantized values in a sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svint8_t &qv, float scale)
{
    const auto          vscale             = svdup_n_f32(scale);
    const svfloat32x4_t vdequantized_input = svcreate4_f32(
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svmovlb_s16(qv))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svmovlb_s16(qv))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svmovlt_s16(qv))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svmovlt_s16(qv))), vscale));
    return vdequantized_input;
}

/** Quantize an sve vector holding floating point values.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return An sve vector holding the quantized values
 */
inline svuint8_t svquantize_z(svbool_t pg, const svfloat32x4_t qv, const UniformQuantizationInfo &qi)
{
    const float scale     = qi.scale;
    const int   offset    = qi.offset;
    const auto  voffset   = svdup_n_f32(offset);
    const auto  vinvscale = svdup_n_f32(1.f / scale);

    const auto rf_0 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 0), vinvscale));
    const auto rf_1 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 1), vinvscale));
    const auto rf_2 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 2), vinvscale));
    const auto rf_3 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 3), vinvscale));

    const auto pa = svqxtnt_u32(svqxtnb_u32(rf_0), rf_1);
    const auto pb = svqxtnt_u32(svqxtnb_u32(rf_2), rf_3);

    return svqxtnt_u16(svqxtnb_u16(pa), pb);
}

/** Signed quantize an sve vector holding floating point values.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return An sve vector holding the quantized values
 */
inline svint8_t svquantize_signed_z(svbool_t pg, const svfloat32x4_t qv, const UniformQuantizationInfo &qi)
{
    const float scale     = qi.scale;
    const int   offset    = qi.offset;
    const auto  voffset   = svdup_n_f32(offset);
    const auto  vinvscale = svdup_n_f32(1.f / scale);
    const auto  rf_0      = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 0), vinvscale));
    const auto  rf_1      = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 1), vinvscale));
    const auto  rf_2      = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 2), vinvscale));
    const auto  rf_3      = svcvt_s32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 3), vinvscale));

    const auto pa = svqxtnt_s32(svqxtnb_s32(rf_0), rf_1);
    const auto pb = svqxtnt_s32(svqxtnb_s32(rf_2), rf_3);

    return svqxtnt_s16(svqxtnb_s16(pa), pb);
}

/** Quantize to QASYMM16 an sve vector holding 16 floating point values.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return An sve vector holding the quantized values
 */
inline svuint16x2_t svquantize_qasymm16_z(svbool_t pg, const svfloat32x4_t qv, const UniformQuantizationInfo &qi)
{
    const float scale     = qi.scale;
    const int   offset    = qi.offset;
    const auto  voffset   = svdup_n_f32(offset);
    const auto  vinvscale = svdup_n_f32(1.f / scale);

    const auto rf_0 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 0), vinvscale));
    const auto rf_1 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 1), vinvscale));
    const auto rf_2 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 2), vinvscale));
    const auto rf_3 = svcvt_u32_f32_z(pg, svmla_f32_z(pg, voffset, svget4_f32(qv, 3), vinvscale));

    const auto pa = svqxtnt_u32(svqxtnb_u32(rf_0), rf_1);
    const auto pb = svqxtnt_u32(svqxtnb_u32(rf_2), rf_3);

    return svcreate2_u16(pa, pb);
}
} // namespace arm_compute
#include "src/core/NEON/SVEAsymm.inl"
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
#endif // ARM_COMPUTE_NEASYMM_H
