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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/add/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
void add_qasymm8_neon(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x      = 16;
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    const UniformQuantizationInfo iq1_info = src0->info()->quantization_info().uniform();
    const UniformQuantizationInfo iq2_info = src1->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info  = dst->info()->quantization_info().uniform();

    const auto scale1 = iq1_info.scale / oq_info.scale;
    const auto scale2 = iq2_info.scale / oq_info.scale;
    const auto offset = float(oq_info.offset) - scale1 * float(iq1_info.offset) - scale2 * float(iq2_info.offset);

    if(is_broadcast_across_x)
    {
        const bool                    is_broadcast_input_2 = input2_win.x().step() == 0;
        Window                        broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window                        non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor                *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor                *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        const auto af_scale = is_broadcast_input_2 ? scale1 : scale2;
        const auto bf_scale = is_broadcast_input_2 ? scale2 : scale1;
        const auto vscale1  = vdupq_n_f32(af_scale);

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = non_broadcast_input.ptr();
            const auto output_ptr              = output.ptr();

            const auto broadcast_value = *broadcast_input.ptr();
            const auto bf = vdupq_n_f32(float(broadcast_value) * scale2 + offset);
            const auto bfs = float(broadcast_value) * bf_scale + offset;

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const uint8x16_t a    = vld1q_u8(non_broadcast_input_ptr + x);

                const auto a_u16_0 = vmovl_u8(vget_low_u8(a));
                const auto a_u16_1 = vmovl_u8(vget_high_u8(a));

                const auto af_0 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_0))), vscale1);
                const auto af_1 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_0))), vscale1);
                const auto af_2 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_1))), vscale1);
                const auto af_3 = vmlaq_f32(bf, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_1))), vscale1);

                int32x4_t rf_0{};
                int32x4_t rf_1{};
                int32x4_t rf_2{};
                int32x4_t rf_3{};

#ifdef __aarch64__
                rf_0 = vcvtnq_s32_f32(af_0);
                rf_1 = vcvtnq_s32_f32(af_1);
                rf_2 = vcvtnq_s32_f32(af_2);
                rf_3 = vcvtnq_s32_f32(af_3);
#else  //__aarch64__
                rf_0 = vcvtq_s32_f32(af_0);
                rf_1 = vcvtq_s32_f32(af_1);
                rf_2 = vcvtq_s32_f32(af_2);
                rf_3 = vcvtq_s32_f32(af_3);
#endif //__aarch64__

                const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                vst1q_u8(output_ptr + x, vcombine_u8(pa, pb));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto result = float(non_broadcast_input_ptr[x]) * af_scale + bfs;
#ifdef __aarch64__
                output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::trunc(result));
#endif  // __aarch64__
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

        const auto vscale1 = vdupq_n_f32(scale1);
        const auto vscale2 = vdupq_n_f32(scale2);
        const auto voffset = vdupq_n_f32(offset);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = input1.ptr();
            const auto input2_ptr = input2.ptr();
            const auto output_ptr = output.ptr();

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const uint8x16_t a = vld1q_u8(input1_ptr + x);
                const uint8x16_t b = vld1q_u8(input2_ptr + x);

                const auto a_u16_0 = vmovl_u8(vget_low_u8(a));
                const auto a_u16_1 = vmovl_u8(vget_high_u8(a));
                const auto b_u16_0 = vmovl_u8(vget_low_u8(b));
                const auto b_u16_1 = vmovl_u8(vget_high_u8(b));

                const auto af_0 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_0))), vscale1);
                const auto af_1 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_0))), vscale1);
                const auto af_2 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_low_u16(a_u16_1))), vscale1);
                const auto af_3 = vmlaq_f32(voffset, vcvtq_f32_u32(vmovl_u16(vget_high_u16(a_u16_1))), vscale1);

                const auto bf_0 = vmlaq_f32(af_0, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_0))), vscale2);
                const auto bf_1 = vmlaq_f32(af_1, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_0))), vscale2);
                const auto bf_2 = vmlaq_f32(af_2, vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_u16_1))), vscale2);
                const auto bf_3 = vmlaq_f32(af_3, vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_u16_1))), vscale2);

                int32x4_t rf_0{};
                int32x4_t rf_1{};
                int32x4_t rf_2{};
                int32x4_t rf_3{};

#ifdef __aarch64__
                rf_0 = vcvtnq_s32_f32(bf_0);
                rf_1 = vcvtnq_s32_f32(bf_1);
                rf_2 = vcvtnq_s32_f32(bf_2);
                rf_3 = vcvtnq_s32_f32(bf_3);
#else  //__aarch64__
                rf_0 = vcvtq_s32_f32(bf_0);
                rf_1 = vcvtq_s32_f32(bf_1);
                rf_2 = vcvtq_s32_f32(bf_2);
                rf_3 = vcvtq_s32_f32(bf_3);
#endif //__aarch64__

                const uint8x8_t pa = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_0), vqmovn_s32(rf_1)));
                const uint8x8_t pb = vqmovun_s16(vcombine_s16(vqmovn_s32(rf_2), vqmovn_s32(rf_3)));
                vst1q_u8(output_ptr + x, vcombine_u8(pa, pb));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto result = float(input1_ptr[x]) * scale1 + float(input2_ptr[x]) * scale2 + offset;
#ifdef __aarch64__
                output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::lround(result));
#else  // __aarch64__
                output_ptr[x] = utility::clamp<int, uint8_t>(support::cpp11::trunc(result));
#endif  // __aarch64__
            }
        },
        input1, input2, output);
    }
}
} // namespace cpu
} // namespace arm_compute