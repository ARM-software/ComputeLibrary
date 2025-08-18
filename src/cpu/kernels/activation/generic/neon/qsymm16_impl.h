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

#ifndef ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QSYMM16_IMPL_H
#define ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QSYMM16_IMPL_H

#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/NEON/NESymm.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{

template <typename F>
void dispatch_neon_qsymm16_activation_function(ActivationLayerInfo::ActivationFunction act,
                                               const ActivationLayerInfo              &act_info,
                                               const UniformQuantizationInfo          &qi_in,
                                               const UniformQuantizationInfo          &qi_out,
                                               F                                     &&fn)
{
    const auto        vconst_1 = vdupq_n_f32(1.f);
    const float32x4_t va_f32   = vdupq_n_f32(act_info.a());
    const float32x4_t vb_f32   = vdupq_n_f32(act_info.b());
    const float       a_f32    = act_info.a();
    const float       b_f32    = act_info.b();

    switch (act)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                    // Perform activation
                    const float32x4x2_t tmp_dep = {{
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                    }};
                    // Re-quantize to new output space
                    return vquantize_int16(tmp_dep, qi_out.scale);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                    tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                    return quantize_qsymm16(tmp_f, qi_out);
                });
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                    // Perform activation
                    const float32x4x2_t tmp_dep = {{
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[0], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[1], vb_f32))),
                    }};
                    // Re-quantize to new output space
                    return vquantize_int16(tmp_dep, qi_out.scale);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                    tmp_f       = a_f32 * std::tanh(b_f32 * tmp_f);
                    return quantize_qsymm16(tmp_f, qi_out);
                });
            break;

        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                    // Perform activation
                    const float32x4x2_t tmp_dep = {{wrapper::vmin(va_f32, wrapper::vmax(vb_f32, vin_deq.val[0])),
                                                    wrapper::vmin(va_f32, wrapper::vmax(vb_f32, vin_deq.val[1]))}};
                    // Re-quantize to new output space
                    return vquantize_int16(tmp_dep, qi_out.scale);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                    tmp_f       = std::min<float>(a_f32, std::max<float>(b_f32, tmp_f));
                    return quantize_qsymm16(tmp_f, qi_out);
                });
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported activation function");
    }
}

} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QSYMM16_IMPL_H
