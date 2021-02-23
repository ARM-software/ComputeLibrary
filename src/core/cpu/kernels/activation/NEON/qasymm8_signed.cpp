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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Validate.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
void qasymm8_signed_neon_activation(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    constexpr int                                 window_step_x  = 16;
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const UniformQuantizationInfo qi_in    = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out   = dst->info()->quantization_info().uniform();
    const qasymm8x16_signed_t     va       = vdupq_n_s8(quantize_qasymm8_signed(act_info.a(), qi_in));
    const qasymm8x16_signed_t     vb       = vdupq_n_s8(quantize_qasymm8_signed(act_info.b(), qi_in));
    const qasymm8_signed_t        a        = quantize_qasymm8_signed(act_info.a(), qi_in);
    const qasymm8_signed_t        b        = quantize_qasymm8_signed(act_info.b(), qi_in);
    const qasymm8_signed_t        const_0  = quantize_qasymm8_signed(0.f, qi_in);
    const qasymm8x16_signed_t     vconst_0 = vdupq_n_s8(const_0);
    const auto                    vconst_1 = vdupq_n_f32(1.f);
#ifndef __aarch64__
    const auto vconst_0_f32 = vdupq_n_f32(1.f);
#endif // __aarch64__
    const float32x4_t va_f32          = vdupq_n_f32(act_info.a());
    const float32x4_t vb_f32          = vdupq_n_f32(act_info.b());
    const float       a_f32           = act_info.a();
    const float       b_f32           = act_info.b();
    const auto        const_6_f32     = vdupq_n_f32(6.f);
    const auto        const_0_f32     = vdupq_n_f32(0.f);
    const auto        const_3_f32     = vdupq_n_f32(3.f);
    const auto        const_inv_6_f32 = vdupq_n_f32(0.166666667f);

    // Initialise scale/offset for re-quantization
    float       s  = qi_in.scale / qi_out.scale;
    float       o  = -qi_in.offset * s + qi_out.offset;
    float32x4_t vs = vdupq_n_f32(s);
    float32x4_t vo = vdupq_n_f32(o);

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const qasymm8_signed_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<qasymm8_signed_t *>(output.ptr());

        wrapper::traits::neon_bitvector_t<qasymm8_signed_t, wrapper::traits::BitWidth::W128> tmp;

        // Compute S elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);
            if(act == ActivationLayerInfo::ActivationFunction::RELU)
            {
                // Perform activation
                tmp = vmaxq_s8(vconst_0, vin);
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8_signed(tmp, vs, vo);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
            {
                // Perform activation
                tmp = vminq_s8(va, vmaxq_s8(vconst_0, vin));
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8_signed(tmp, vs, vo);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
            {
                // Perform activation
                tmp = vminq_s8(va, vmaxq_s8(vb, vin));
                // Re-quantize to new output space
                tmp = vmlaq_qasymm8_signed(tmp, vs, vo);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                // De-quantize
                const auto vin_deq = vdequantize(vin, qi_in);
                // Perform activation
                const float32x4x4_t tmp_dep =
                {
                    {
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[2])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[3])))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize_signed(tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::TANH)
            {
                // De-quantize
                const auto vin_deq = vdequantize(vin, qi_in);
                // Perform activation
                const float32x4x4_t tmp_dep =
                {
                    {
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[0], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[1], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[2], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[3], vb_f32))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize_signed(tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::HARD_SWISH)
            {
                // De-quantize
                const auto vin_deq = vdequantize(vin, qi_in);
                // Perform activation
                const float32x4x4_t tmp_dep =
                {
                    {
                        wrapper::vmul(vin_deq.val[0], wrapper::vmul(const_inv_6_f32, wrapper::vmin(const_6_f32, wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[0], const_3_f32))))),
                        wrapper::vmul(vin_deq.val[1], wrapper::vmul(const_inv_6_f32, wrapper::vmin(const_6_f32, wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[1], const_3_f32))))),
                        wrapper::vmul(vin_deq.val[2], wrapper::vmul(const_inv_6_f32, wrapper::vmin(const_6_f32, wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[2], const_3_f32))))),
                        wrapper::vmul(vin_deq.val[3], wrapper::vmul(const_inv_6_f32, wrapper::vmin(const_6_f32, wrapper::vmax(const_0_f32, wrapper::vadd(vin_deq.val[3], const_3_f32))))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize_signed(tmp_dep, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LEAKY_RELU)
            {
                const auto vin_deq = vdequantize(vin, qi_in);

#ifdef __aarch64__
                const uint32x4x4_t pos_mask =
                {
                    {
                        wrapper::vcgtz(vin_deq.val[0]),
                        wrapper::vcgtz(vin_deq.val[1]),
                        wrapper::vcgtz(vin_deq.val[2]),
                        wrapper::vcgtz(vin_deq.val[3]),
                    }
                };
#else  // __aarch64__
                const uint32x4x4_t pos_mask =
                {
                    {
                        wrapper::vcgt(vin_deq.val[0], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[1], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[2], vconst_0_f32),
                        wrapper::vcgt(vin_deq.val[3], vconst_0_f32),
                    }
                };
#endif // __aarch64__

                const float32x4x4_t tmp_dep =
                {
                    {
                        wrapper::vbsl(pos_mask.val[0], vin_deq.val[0], wrapper::vmul(va_f32, vin_deq.val[0])),
                        wrapper::vbsl(pos_mask.val[1], vin_deq.val[1], wrapper::vmul(va_f32, vin_deq.val[1])),
                        wrapper::vbsl(pos_mask.val[2], vin_deq.val[2], wrapper::vmul(va_f32, vin_deq.val[2])),
                        wrapper::vbsl(pos_mask.val[3], vin_deq.val[3], wrapper::vmul(va_f32, vin_deq.val[3])),
                    }
                };

                tmp = vquantize_signed(tmp_dep, qi_out);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            wrapper::vstore(output_ptr + x, tmp);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            qasymm8_signed_t in  = *(reinterpret_cast<const qasymm8_signed_t *>(input_ptr + x));
            qasymm8_signed_t tmp = 0;
            if(act == ActivationLayerInfo::ActivationFunction::RELU)
            {
                tmp = std::max(const_0, in);
                tmp = utility::clamp<int32_t, qasymm8_signed_t>(tmp * s + o);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
            {
                tmp = std::min(a, std::max(const_0, in));
                tmp = utility::clamp<int32_t, qasymm8_signed_t>(tmp * s + o);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
            {
                tmp = std::min(a, std::max(b, in));
                tmp = utility::clamp<int32_t, qasymm8_signed_t>(tmp * s + o);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                tmp         = quantize_qasymm8_signed(tmp_f, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::TANH)
            {
                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                tmp_f       = a_f32 * std::tanh(b_f32 * tmp_f);
                tmp         = quantize_qasymm8_signed(tmp_f, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::HARD_SWISH)
            {
                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                tmp_f       = tmp_f * ((std::min(std::max((tmp_f + 3), 0.0f), 6.0f)) * 0.166666667f);
                tmp         = quantize_qasymm8_signed(tmp_f, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::LEAKY_RELU)
            {
                float tmp_f = dequantize_qasymm8_signed(in, qi_in);
                tmp_f       = tmp_f > 0 ? tmp_f : tmp_f * a_f32;
                tmp         = quantize_qasymm8_signed(tmp_f, qi_out);
            }
            else
            {
                ARM_COMPUTE_ERROR("Unsupported activation function");
            }
            *(output_ptr + x) = tmp;
        }
    },
    input, output);
}
} // namespace cpu
} // namespace arm_compute
