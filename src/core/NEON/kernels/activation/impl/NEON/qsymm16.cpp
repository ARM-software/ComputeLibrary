/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/experimental/Types.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/NESymm.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/StdTypes.h"
#include "src/core/common/Validate.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace cpu
{
void qsymm16_neon_activation(const ITensor *src, ITensor *dst, const ActivationLayerInfo &act_info, const Window &window)
{
    constexpr int                                 window_step_x  = 8;
    const auto                                    window_start_x = static_cast<int>(window.x().start());
    const auto                                    window_end_x   = static_cast<int>(window.x().end());
    const ActivationLayerInfo::ActivationFunction act            = act_info.activation();

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const UniformQuantizationInfo qi_in    = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo qi_out   = dst->info()->quantization_info().uniform();
    const auto                    vconst_1 = vdupq_n_f32(1.f);
    const float32x4_t             va_f32   = vdupq_n_f32(act_info.a());
    const float32x4_t             vb_f32   = vdupq_n_f32(act_info.b());
    const float                   a_f32    = act_info.a();
    const float                   b_f32    = act_info.b();

    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const qsymm16_t *>(input.ptr());
        const auto output_ptr = reinterpret_cast<qsymm16_t *>(output.ptr());

        wrapper::traits::neon_bitvector_t<qsymm16_t, wrapper::traits::BitWidth::W128> tmp;
        ARM_COMPUTE_UNUSED(tmp);

        // Compute S elements per iteration
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const auto vin = wrapper::vloadq(input_ptr + x);
            if(act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                // De-quantize
                const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                // Perform activation
                const float32x4x2_t tmp_dep =
                {
                    {
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[0])))),
                        wrapper::vdiv(vconst_1, wrapper::vadd(vconst_1, wrapper::vexpq(wrapper::vneg(vin_deq.val[1])))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize_int16(tmp_dep, qi_out.scale);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::TANH)
            {
                // De-quantize
                const auto vin_deq = vdequantize_int16(vin, qi_in.scale);
                // Perform activation
                const float32x4x2_t tmp_dep =
                {
                    {
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[0], vb_f32))),
                        wrapper::vmul(va_f32, wrapper::vtanh(wrapper::vmul(vin_deq.val[1], vb_f32))),
                    }
                };
                // Re-quantize to new output space
                tmp = vquantize_int16(tmp_dep, qi_out.scale);
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
            qsymm16_t in  = *(reinterpret_cast<const qsymm16_t *>(input_ptr + x));
            qsymm16_t tmp = 0;
            if(act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
            {
                float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                tmp_f       = 1.f / (1.f + std::exp(-tmp_f));
                tmp         = quantize_qsymm16(tmp_f, qi_out);
            }
            else if(act == ActivationLayerInfo::ActivationFunction::TANH)
            {
                float tmp_f = dequantize_qsymm16(in, qi_in.scale);
                tmp_f       = a_f32 * std::tanh(b_f32 * tmp_f);
                tmp         = quantize_qsymm16(tmp_f, qi_out);
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
