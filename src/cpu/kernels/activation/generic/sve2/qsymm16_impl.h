/*
 * Copyright (c) 2020-2023, 2025 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE2_QSYMM16_IMPL_H
#define ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE2_QSYMM16_IMPL_H

#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/NEON/SVEMath.h"
#include "src/core/NEON/SVESymm.h"

#include <arm_sve.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
template <typename F>
void dispatch_sve2_qasymm16_activation_function(ActivationLayerInfo::ActivationFunction act,
                                                const ActivationLayerInfo              &act_info,
                                                const UniformQuantizationInfo          &qi_in,
                                                const UniformQuantizationInfo          &qi_out,
                                                F                                     &&fn)
{
    const auto vconst_1 = svdup_n_f32(1.f);
    const auto va_f32   = svdup_n_f32(act_info.a());
    const auto vb_f32   = svdup_n_f32(act_info.b());

    switch (act)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            fn(
                [&](auto vin, auto pg)
                {
                    // De-quantize
                    auto vin_deq = svdequantize_qsymm16_z(pg, vin, qi_in.scale);
                    // Perform activation
                    const svfloat32x2_t tmp_dep = svcreate2_f32(
                        svdiv_f32_z(
                            pg, vconst_1,
                            svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget2_f32(vin_deq, 0))))),
                        svdiv_f32_z(
                            pg, vconst_1,
                            svadd_f32_z(pg, vconst_1, svexp_f32_z(pg, svneg_f32_z(pg, svget2_f32(vin_deq, 1))))));
                    // Re-quantize to new output space
                    return svquantize_qsymm16_z(pg, tmp_dep, qi_out.scale);
                });
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            fn(
                [&](auto vin, auto pg)
                {
                    // De-quantize
                    auto vin_deq = svdequantize_qsymm16_z(pg, vin, qi_in.scale);
                    // Perform activation
                    const svfloat32x2_t tmp_dep = svcreate2_f32(
                        svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget2_f32(vin_deq, 0), vb_f32))),
                        svmul_f32_z(pg, va_f32, svtanh_f32_z(pg, svmul_f32_z(pg, svget2_f32(vin_deq, 1), vb_f32))));
                    // Re-quantize to new output space
                    return svquantize_qsymm16_z(pg, tmp_dep, qi_out.scale);
                });
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            fn(
                [&](auto vin, auto pg)
                {
                    // De-quantize
                    auto vin_deq = svdequantize_qsymm16_z(pg, vin, qi_in.scale);
                    // Perform activation
                    const svfloat32x2_t tmp_dep =
                        svcreate2_f32(svmin_f32_z(pg, va_f32, svmax_f32_z(pg, vb_f32, svget2_f32(vin_deq, 0))),
                                      svmin_f32_z(pg, va_f32, svmax_f32_z(pg, vb_f32, svget2_f32(vin_deq, 1))));
                    // Re-quantize to new output space
                    return svquantize_qsymm16_z(pg, tmp_dep, qi_out.scale);
                });
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported activation function");
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE2_QSYMM16_IMPL_H
