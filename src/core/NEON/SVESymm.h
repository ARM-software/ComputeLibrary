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
#ifndef ARM_COMPUTE_SVESYMM_H
#define ARM_COMPUTE_SVESYMM_H

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#if defined(ARM_COMPUTE_ENABLE_SVE2)
#include "src/core/NEON/SVEMath.h"
#include <arm_sve.h>

namespace arm_compute
{
/** Dequantize an sve vector holding 16-bit quantized values.
 *
 * @param[in] pg    Predicate value.
 * @param[in] qv    Input values to be dequantized.
 * @param[in] scale Quantization scale
 *
 * @return Dequantized values in an sve vector
 */
inline svfloat32x2_t svdequantize_qsymm16_z(svbool_t pg, const svint16_t &qv, float scale)
{
    const auto          vscale             = svdup_n_f32(scale);
    const svfloat32x2_t vdequantized_input = svcreate2_f32(svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(qv)), vscale), svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(qv)), vscale));
    return vdequantized_input;
}

/** Quantize an sve vector holding 8 floating point values.
 *
 * @param[in] pg    Predicate value.
 * @param[in] qv    Input values to be quantized.
 * @param[in] scale Quantization scale
 *
 * @return An sve vector holding the quantized values
 */
inline svint16_t svquantize_qsymm16_z(svbool_t pg, const svfloat32x2_t qv, float scale)
{
    const svfloat32_t vinvscale = svdup_n_f32(1.f / scale);

    const auto rf_0 = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget2_f32(qv, 0), vinvscale));
    const auto rf_1 = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget2_f32(qv, 1), vinvscale));
    const auto pa   = svqxtnt_s32(svqxtnb_s32(rf_0), rf_1);

    return pa;
}

/** Dequantize an sve vector holding 16 16-bit quantized values.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be dequantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return Dequantized values in an sve vector
 */
inline svfloat32x4_t svdequantize_z(svbool_t pg, const svint16x2_t qv, const UniformQuantizationInfo &qi)
{
    const float         scale              = qi.scale;
    const auto          vscale             = svdup_n_f32(scale);
    const svfloat32x4_t vdequantized_input = svcreate4_f32(
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svget2_s16(qv, 0))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svget2_s16(qv, 0))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlb_s32(svget2_s16(qv, 1))), vscale),
                                                 svmul_f32_z(pg, svcvt_f32_s32_z(pg, svmovlt_s32(svget2_s16(qv, 1))), vscale));
    return vdequantized_input;
}

/** Quantize an sve vector holding 16 floating point values.
 *
 * @param[in] pg Predicate value.
 * @param[in] qv Input values to be quantized.
 * @param[in] qi Quantization information to be used in the computation.
 *
 * @return An sve vector holding the quantized values
 */
inline svint16x2_t svquantize_qsymm16_z(svbool_t pg, const svfloat32x4_t qv, const UniformQuantizationInfo &qi)
{
    const float scale = qi.scale;
    ARM_COMPUTE_ERROR_ON(scale == 0.f);
    const auto vinvscale = svdup_n_f32(1.f / scale);
    const auto rf_0      = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget4_f32(qv, 0), vinvscale));
    const auto rf_1      = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget4_f32(qv, 1), vinvscale));
    const auto rf_2      = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget4_f32(qv, 2), vinvscale));
    const auto rf_3      = svcvt_s32_f32_z(pg, svmul_f32_z(pg, svget4_f32(qv, 3), vinvscale));

    const auto pa = svqxtnt_s32(svqxtnb_s32(rf_0), rf_1);
    const auto pb = svqxtnt_s32(svqxtnb_s32(rf_2), rf_3);

    return svcreate2_s16(pa, pb);
}

} // namespace arm_compute
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
#endif // ARM_COMPUTE_NESYMM_H