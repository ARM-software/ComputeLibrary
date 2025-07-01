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

#ifndef ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE_FP32_IMPL_H
#define ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE_FP32_IMPL_H

#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/NEON/SVEMath.h"

#include <arm_sve.h>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
template <typename F>
void dispatch_sve_fp32_activation_function(ActivationLayerInfo::ActivationFunction act,
                                           const ActivationLayerInfo              &act_info,
                                           F                                     &&fn)
{
    const auto const_1          = svdup_n_f32(1.f);
    const auto const_0          = svdup_n_f32(0.f);
    const auto const_6          = svdup_n_f32(6.f);
    const auto const_3          = svdup_n_f32(3.f);
    const auto const_inv_6      = svdup_n_f32(0.166666667f);
    const auto soft_relu_thresh = svdup_n_f32(16.63553047f);

    const auto va = svdup_n_f32(act_info.a());
    const auto vb = svdup_n_f32(act_info.b());
    switch (act)
    {
        case ActivationLayerInfo::ActivationFunction::ABS:
            fn([&](auto vin, svbool_t pg) { return svabs_f32_z(pg, vin); });
            break;
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            fn([&](auto vin, svbool_t pg) { return svmla_f32_z(pg, vb, va, vin); });
            break;
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            fn([&](auto vin, svbool_t pg)
               { return svinv_f32_z(pg, svadd_f32_z(pg, const_1, svexp_f32_z(pg, svneg_f32_z(pg, vin)))); });
            break;
        case ActivationLayerInfo::ActivationFunction::RELU:
            fn([&](auto vin, svbool_t pg) { return svmax_f32_z(pg, const_0, vin); });
            break;
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            fn([&](auto vin, svbool_t pg) { return svmin_f32_z(pg, va, svmax_f32_z(pg, const_0, vin)); });
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            fn([&](auto vin, svbool_t pg) { return svmin_f32_z(pg, va, svmax_f32_z(pg, vb, vin)); });
            break;
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            fn(
                [&](auto vin, svbool_t pg) {
                    return svadd_f32_z(pg, svmul_f32_z(pg, svmin_f32_z(pg, vin, const_0), va),
                                       svmax_f32_z(pg, vin, const_0));
                });
            break;
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            fn(
                [&](auto vin, svbool_t pg)
                {
                    return svsel_f32(svcmpgt_f32(pg, vin, soft_relu_thresh), vin,
                                     svlog_f32_z(pg, svadd_f32_z(pg, const_1, svexp_f32_z(pg, vin))));
                });
            break;
        case ActivationLayerInfo::ActivationFunction::ELU:
            fn(
                [&](auto vin, svbool_t pg)
                {
                    return svsel_f32(svcmpgt_f32(pg, vin, const_0), vin,
                                     svmul_f32_z(pg, va, svsub_f32_z(pg, svexp_f32_z(pg, vin), const_1)));
                });
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            fn([&](auto vin, svbool_t pg) { return svsqrt_f32_z(pg, vin); });
            break;
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            fn([&](auto vin, svbool_t pg) { return svmul_f32_z(pg, vin, vin); });
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            fn([&](auto vin, svbool_t pg) { return svmul_f32_z(pg, va, svtanh_f32_z(pg, svmul_f32_z(pg, vb, vin))); });
            break;
        case ActivationLayerInfo::ActivationFunction::IDENTITY:
            fn([&](auto vin, svbool_t) { return vin; });
            break;
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
            fn(
                [&](auto vin, svbool_t pg)
                {
                    return svmul_f32_z(
                        pg, vin,
                        svmul_f32_z(pg, const_inv_6,
                                    svmin_f32_z(pg, const_6, svmax_f32_z(pg, const_0, svadd_f32_z(pg, vin, const_3)))));
                });
            break;
        case ActivationLayerInfo::ActivationFunction::SWISH:
            fn(
                [&](auto vin, svbool_t pg)
                {
                    return svmul_f32_z(
                        pg, vin,
                        svinv_f32_z(
                            pg, svadd_f32_z(pg, const_1, svexp_f32_z(pg, svneg_f32_z(pg, svmul_f32_z(pg, va, vin))))));
                });
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported activation function");
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_SVE_FP32_IMPL_H
