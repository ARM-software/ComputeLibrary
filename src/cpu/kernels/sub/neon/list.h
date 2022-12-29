/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_SUB_LIST_H
#define SRC_CORE_NEON_KERNELS_SUB_LIST_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
#define DECLARE_SUB_KERNEL(func_name) \
    void func_name(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)

DECLARE_SUB_KERNEL(sub_qasymm8_neon_fixedpoint);
DECLARE_SUB_KERNEL(sub_qasymm8_signed_neon_fixedpoint);
DECLARE_SUB_KERNEL(sub_qasymm8_neon);
DECLARE_SUB_KERNEL(sub_qasymm8_signed_neon);
DECLARE_SUB_KERNEL(sub_qsymm16_neon);

#undef DECLARE_SUB_KERNEL

template <typename T>
void sub_same_neon(const ITensor *src0, const ITensor *src1, ITensor *dst, const ConvertPolicy &policy, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    bool is_sat = policy == ConvertPolicy::SATURATE;

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(src0->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(src1->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    constexpr int window_step_x         = 16 / sizeof(T);
    const auto    window_start_x        = static_cast<int>(window.x().start());
    const auto    window_end_x          = static_cast<int>(window.x().end());
    const bool    is_broadcast_across_x = src0->info()->tensor_shape().x() != src1->info()->tensor_shape().x();

    Iterator input1(src0, window.broadcast_if_dimension_le_one(src0->info()->tensor_shape()));
    Iterator input2(src1, window.broadcast_if_dimension_le_one(src1->info()->tensor_shape()));
    Iterator output(dst, window);

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? src1 : src0;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? src1 : src0;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(dst, win);

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const T *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<T *>(output.ptr());

            const T    broadcast_value     = *reinterpret_cast<const T *>(broadcast_input.ptr());
            const auto broadcast_value_vec = wrapper::vdup_n(broadcast_value, ExactTagType{});

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto non_broadcast_v = wrapper::vloadq(non_broadcast_input_ptr + x);
                auto       res             = is_sat ? wrapper::vqsub(broadcast_value_vec, non_broadcast_v) : wrapper::vsub(broadcast_value_vec, non_broadcast_v);
                if(is_broadcast_input_2)
                {
                    res = wrapper::vmul(res, wrapper::vdup_n(static_cast<T>(-1), ExactTagType{}));
                }
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto non_broadcast_v = *(non_broadcast_input_ptr + x);
                auto       res             = is_sat ? wrapper::sub_sat(broadcast_value, non_broadcast_v) : broadcast_value - non_broadcast_v;
                if(is_broadcast_input_2)
                {
                    res = static_cast<T>(-1) * res;
                }

                *(output_ptr + x) = res;
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

        execute_window_loop(
            win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const T *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const T *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<T *>(output.ptr());

            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto val1 = wrapper::vloadq(input1_ptr + x);
                const auto val2 = wrapper::vloadq(input2_ptr + x);
                const auto res  = is_sat ? wrapper::vqsub(val1, val2) : wrapper::vsub(val1, val2);
                wrapper::vstore(output_ptr + x, res);
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                const auto val1   = *(input1_ptr + x);
                const auto val2   = *(input2_ptr + x);
                *(output_ptr + x) = is_sat ? wrapper::sub_sat(val1, val2) : val1 - val2;
            }
        },
        input1, input2, output);
    }
}
} // namespace cpu
} // namespace arm_compute
#endif // SRC_CORE_NEON_KERNELS_SUB_LIST_H
