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

#ifndef ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QASYMM8_IMPL_H
#define ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QASYMM8_IMPL_H

#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace cpu
{

template <typename F>
void dispatch_qasymm8_activation_function(ActivationLayerInfo::ActivationFunction act,
                                          const ActivationLayerInfo              &act_info,
                                          const UniformQuantizationInfo          &qi_in,
                                          const UniformQuantizationInfo          &qi_out,
                                          F                                     &&fn)
{
    const qasymm8x16_t va       = vdupq_n_u8(quantize_qasymm8(act_info.a(), qi_in));
    const qasymm8x16_t vb       = vdupq_n_u8(quantize_qasymm8(act_info.b(), qi_in));
    const qasymm8_t    a        = quantize_qasymm8(act_info.a(), qi_in);
    const qasymm8_t    b        = quantize_qasymm8(act_info.b(), qi_in);
    const qasymm8_t    const_0  = quantize_qasymm8(0.f, qi_in);
    const qasymm8x16_t vconst_0 = vdupq_n_u8(const_0);

    const auto vconst_1 = vdupq_n_f32(1.f);
#ifndef __aarch64__
    const auto vconst_0_f32 = vdupq_n_f32(0);
#else  // __aarch64__
    const auto const_inv_2      = vdupq_n_f32(0.5f);
    const auto const_inv_sqrt_2 = vdupq_n_f32(0.70710678118f);
#endif // __aarch64__
    const float32x4_t va_f32 = vdupq_n_f32(act_info.a());
    const float32x4_t vb_f32 = vdupq_n_f32(act_info.b());
    const float       a_f32  = act_info.a();
    const float       b_f32  = act_info.b();

#ifndef __aarch64__
    const auto const_6_f32     = vdupq_n_f32(6.f);
    const auto const_0_f32     = vdupq_n_f32(0.f);
    const auto const_3_f32     = vdupq_n_f32(3.f);
    const auto const_inv_6_f32 = vdupq_n_f32(0.166666667f);
#endif // __aarch64__

    // Initialise scale/offset for re-quantization
    float       s  = qi_in.scale / qi_out.scale;
    float       o  = -qi_in.offset * s + qi_out.offset;
    float32x4_t vs = vdupq_n_f32(s);
    float32x4_t vo = vdupq_n_f32(o);

    switch (act)
    {
        case ActivationLayerInfo::ActivationFunction::RELU:
            fn(
                [&](auto vin)
                {
                    // Perform activation
                    auto tmp = vmaxq_u8(vconst_0, vin);
                    // Re-quantize to new output space
                    return vmlaq_qasymm8<RoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                },
                [&](auto in)
                {
                    auto tmp = std::max(const_0, in);
                    return utility::clamp<int32_t, qasymm8_t>(support::cpp11::lround(tmp * s + o));
                });
            break;

        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            fn(
                [&](auto vin)
                {
                    // Perform activation
                    auto tmp = vminq_u8(va, vmaxq_u8(vconst_0, vin));
                    // Re-quantize to new output space
                    return vmlaq_qasymm8<RoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                },
                [&](auto in)
                {
                    auto tmp = std::min(a, std::max(const_0, in));
                    return utility::clamp<int32_t, qasymm8_t>(support::cpp11::lround(tmp * s + o));
                });
            break;

        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            fn(
                [&](auto vin)
                {
                    // Perform activation
                    auto tmp = vminq_u8(va, vmaxq_u8(vb, vin));
                    // Re-quantize to new output space
                    return vmlaq_qasymm8<RoundingPolicy::TO_NEAREST_UP>(tmp, vs, vo);
                },
                [&](auto in)
                {
                    auto tmp = std::min(a, std::max(b, in));
                    return utility::clamp<int32_t, qasymm8_t>(support::cpp11::lround(tmp * s + o));
                });
            break;

#ifndef __aarch64__ // LUT-based implementation is used for aarch64 instead.
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize(vin, qi_in);
                    // Perform activation
                    const float32x4x4_t tmp_dep = {{
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[2])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[3])))),
                    }};
                    // Re-quantize to new output space
                    return vquantize(tmp_dep, qi_out);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qasymm8(in, qi_in);
                    tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                    return quantize_qasymm8(tmp_f, qi_out);
                });
            break;
#endif // __aarch64__
        case ActivationLayerInfo::ActivationFunction::TANH:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize(vin, qi_in);
                    // Perform activation
                    const float32x4x4_t tmp_dep = {{
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[0], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[1], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[2], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[3], vb_f32))),
                    }};
                    // Re-quantize to new output space
                    return vquantize(tmp_dep, qi_out);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qasymm8(in, qi_in);
                    tmp_f       = a_f32 * std::tanh(b_f32 * tmp_f);
                    return quantize_qasymm8(tmp_f, qi_out);
                });
            break;
#ifndef __aarch64__ // LUT-based implementation is used for aarch64 instead.
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
            fn(
                [&](auto vin)
                {
                    // De-quantize
                    const auto vin_deq = vdequantize(vin, qi_in);
                    // Perform activation
                    const float32x4x4_t tmp_dep = {{
                        wrapper::vmul(
                            vin_deq.val[0],
                            wrapper::vmul(
                                const_inv_6_f32,
                                wrapper::vmin(const_6_f32,
                                              wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[0], const_3_f32))))),
                        wrapper::vmul(
                            vin_deq.val[1],
                            wrapper::vmul(
                                const_inv_6_f32,
                                wrapper::vmin(const_6_f32,
                                              wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[1], const_3_f32))))),
                        wrapper::vmul(
                            vin_deq.val[2],
                            wrapper::vmul(
                                const_inv_6_f32,
                                wrapper::vmin(const_6_f32,
                                              wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[2], const_3_f32))))),
                        wrapper::vmul(
                            vin_deq.val[3],
                            wrapper::vmul(
                                const_inv_6_f32,
                                wrapper::vmin(const_6_f32,
                                              wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[3], const_3_f32))))),
                    }};
                    // Re-quantize to new output space
                    return vquantize(tmp_dep, qi_out);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qasymm8(in, qi_in);
                    tmp_f       = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
                    return quantize_qasymm8(tmp_f, qi_out);
                });
            break;
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            fn(
                [&](auto vin)
                {
                    const auto vin_deq = vdequantize(vin, qi_in);

                    const uint32x4x4_t pos_mask = {{
                        wrapper::vcgt(vin_deq.val[0], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[1], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[2], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[3], vconst_0_f32),
                    }};

                    const float32x4x4_t tmp_dep = {{
                        wrapper::vbsl(pos_mask.val[0], vin_deq.val[0], wrapper::vmul(va_f32, vin_deq.val[0])),
                        wrapper::vbsl(pos_mask.val[1], vin_deq.val[1], wrapper::vmul(va_f32, vin_deq.val[1])),
                        wrapper::vbsl(pos_mask.val[2], vin_deq.val[2], wrapper::vmul(va_f32, vin_deq.val[2])),
                        wrapper::vbsl(pos_mask.val[3], vin_deq.val[3], wrapper::vmul(va_f32, vin_deq.val[3])),
                    }};

                    return vquantize(tmp_dep, qi_out);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qasymm8(in, qi_in);
                    tmp_f       = tmp_f > 0 ? tmp_f : tmp_f * a_f32;
                    return quantize_qasymm8(tmp_f, qi_out);
                });
            break;
#else  // __aarch64__
        case ActivationLayerInfo::ActivationFunction::GELU:
            fn(
                [&](auto vin)
                {
                    const auto vin_deq = vdequantize(vin, qi_in);
                    // Perform activation
                    const float32x4x4_t tmp_dep = {{
                        wrapper::vmul(vin_deq.val[0],
                                      wrapper::vmul(const_inv_2,
                                                    wrapper::vadd(vconst_1, wrapper::verf(wrapper::vmul(
                                                                                vin_deq.val[0], const_inv_sqrt_2))))),
                        wrapper::vmul(vin_deq.val[1],
                                      wrapper::vmul(const_inv_2,
                                                    wrapper::vadd(vconst_1, wrapper::verf(wrapper::vmul(
                                                                                vin_deq.val[1], const_inv_sqrt_2))))),
                        wrapper::vmul(vin_deq.val[2],
                                      wrapper::vmul(const_inv_2,
                                                    wrapper::vadd(vconst_1, wrapper::verf(wrapper::vmul(
                                                                                vin_deq.val[2], const_inv_sqrt_2))))),
                        wrapper::vmul(vin_deq.val[3],
                                      wrapper::vmul(const_inv_2,
                                                    wrapper::vadd(vconst_1, wrapper::verf(wrapper::vmul(
                                                                                vin_deq.val[3], const_inv_sqrt_2))))),
                    }};
                    // Re-quantize to new output space
                    return vquantize(tmp_dep, qi_out);
                },
                [&](auto in)
                {
                    float tmp_f = dequantize_qasymm8(in, qi_in);
                    tmp_f       = tmp_f * 0.5f * (1.0f + erff(in / 1.41421356237f));
                    return quantize_qasymm8(tmp_f, qi_out);
                });
            break;
#endif // __aarch64__
        default:
            ARM_COMPUTE_ERROR("Unsupported activation function");
    }
}

} // namespace cpu
} // namespace arm_compute

#endif // ACL_SRC_CPU_KERNELS_ACTIVATION_GENERIC_NEON_QASYMM8_IMPL_H
