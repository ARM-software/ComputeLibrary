/*
 * Copyright (c) 2021 Arm Limited.
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
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
void sub_qsymm16_neon(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x         = 8;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    const UniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = dst->info()->quantization_info().uniform();

    const float32x4_t vscale1    = vdupq_n_f32(iq1_info.scale);
    const float32x4_t vscale2    = vdupq_n_f32(iq2_info.scale);
    const float32x4_t invvscaleo = vdupq_n_f32(1.f / oq_info.scale);

    if(is_broadcast_across_x)
    {
        const bool                    is_broadcast_input_2 = input2_win.x().step() == 0;
        Window                        broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window                        non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor                *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor                *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;
        const UniformQuantizationInfo broadcast_qinfo      = broadcast_tensor->info()->quantization_info().uniform();
        const UniformQuantizationInfo non_broadcast_qinfo  = non_broadcast_tensor->info()->quantization_info().uniform();

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const int16_t *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<int16_t *>(output.ptr());

            const int16_t   broadcast_value     = *reinterpret_cast<const int16_t *>(broadcast_input.ptr());
            const int16x8_t broadcast_value_vec = vdupq_n_s16(broadcast_value);

            const float32x4x2_t bf =
            {
                {
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(broadcast_value_vec))), vscale2),
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(broadcast_value_vec))), vscale2),
                }
            };
            const float bfs = static_cast<int32_t>(broadcast_value) * broadcast_qinfo.scale;

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int16x8_t     a = vld1q_s16(non_broadcast_input_ptr + x);
                const float32x4x2_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1),
                    }
                };

                const int32x4x4_t rf =
                {
                    {
#ifdef __aarch64__
                        vcvtnq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[0], af.val[0]) : vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmulq_f32(is_broadcast_input_2 ? vsubq_f32(bf.val[1], af.val[1]) : vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1]));
                vst1q_s16(output_ptr + x, pa);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs   = static_cast<int32_t>(*(non_broadcast_input_ptr + x)) * non_broadcast_qinfo.scale;
                *(output_ptr + x) = quantize_qsymm16(is_broadcast_input_2 ? (bfs - afs) : (afs - bfs), oq_info);
            }
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(src0, input1_win);
        Iterator input2(src1, input2_win);
        Iterator output(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const int16_t *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const int16x8_t a = vld1q_s16(input1_ptr + x);
                const int16x8_t b = vld1q_s16(input2_ptr + x);

                const float32x4x2_t af =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), vscale1),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a))), vscale1),
                    }
                };

                const float32x4x2_t bf =
                {
                    {
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))), vscale2),
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(b))), vscale2),
                    }
                };

                const int32x4x2_t rf =
                {
                    {
#ifdef __aarch64__
                        vcvtnq_s32_f32(vmulq_f32(vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtnq_s32_f32(vmulq_f32(vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#else  //__aarch64__
                        vcvtq_s32_f32(vmulq_f32(vsubq_f32(af.val[0], bf.val[0]), invvscaleo)),
                        vcvtq_s32_f32(vmulq_f32(vsubq_f32(af.val[1], bf.val[1]), invvscaleo)),
#endif //__aarch64__
                    }
                };

                const int16x8_t pa = vcombine_s16(vqmovn_s32(rf.val[0]), vqmovn_s32(rf.val[1]));
                vst1q_s16(output_ptr + x, pa);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const float afs   = static_cast<int32_t>((*(input1_ptr + x))) * iq1_info.scale;
                const float bfs   = static_cast<int32_t>((*(input2_ptr + x))) * iq2_info.scale;
                *(output_ptr + x) = quantize_qsymm16((afs - bfs), dst->info()->quantization_info());
            }
        },
        input1, input2, output);
    }
}
} // namespace cpu
} // namespace arm_compute