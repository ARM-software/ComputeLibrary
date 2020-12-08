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
#ifndef SRC_CORE_SVE_KERNELS_ARITHMETIC_ADDITION_LIST_H
#define SRC_CORE_SVE_KERNELS_ARITHMETIC_ADDITION_LIST_H

#if defined(__ARM_FEATURE_SVE)
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/SVEMath.h"
#include "src/core/NEON/wrapper/intrinsics/intrinsics.h"
#include <arm_sve.h>

namespace arm_compute
{
namespace cpu
{
#define DECLARE_ARITHMETIC_ADDITION_KERNEL(func_name) \
    void func_name(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)

DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_qasymm8_sve);
DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_qasymm8_signed_sve);
DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_qsymm16_sve);
DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_S16_U8_S16_sve);
DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_U8_S16_S16_sve);
DECLARE_ARITHMETIC_ADDITION_KERNEL(arithmetic_addition_U8_U8_S16_sve);

#undef DECLARE_ARITHMETIC_ADDITION_KERNEL

template <typename ScalarType>
void arithmetic_addition_same_sve(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)
{
    const auto all_true_pg           = wrapper::svptrue<ScalarType>();
    const auto window_start_x        = static_cast<int>(window.x().start());
    const auto window_end_x          = static_cast<int>(window.x().end());
    const bool is_broadcast_across_x = in1->info()->tensor_shape().x() != in2->info()->tensor_shape().x();
    const bool is_sat                = (policy == ConvertPolicy::SATURATE);

    // Clear X Dimension on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    // Create input windows
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    Iterator input1(in1, window.broadcast_if_dimension_le_one(in1->info()->tensor_shape()));
    Iterator input2(in2, window.broadcast_if_dimension_le_one(in2->info()->tensor_shape()));
    Iterator output(out, window);

    if(is_broadcast_across_x)
    {
        const bool     is_broadcast_input_2 = input2_win.x().step() == 0;
        Window         broadcast_win        = is_broadcast_input_2 ? input2_win : input1_win;
        Window         non_broadcast_win    = !is_broadcast_input_2 ? input2_win : input1_win;
        const ITensor *broadcast_tensor     = is_broadcast_input_2 ? in2 : in1;
        const ITensor *non_broadcast_tensor = !is_broadcast_input_2 ? in2 : in1;

        // Clear X Dimension on execution window as we handle manually
        non_broadcast_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator broadcast_input(broadcast_tensor, broadcast_win);
        Iterator non_broadcast_input(non_broadcast_tensor, non_broadcast_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto non_broadcast_input_ptr = reinterpret_cast<const ScalarType *>(non_broadcast_input.ptr());
            const auto output_ptr              = reinterpret_cast<ScalarType *>(output.ptr());

            const ScalarType broadcast_value     = *reinterpret_cast<const ScalarType *>(broadcast_input.ptr());
            const auto       broadcast_value_vec = wrapper::svdup_n(broadcast_value);

            int      x  = window_start_x;
            svbool_t pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
            do
            {
                const auto non_broadcast_v = svld1(pg, non_broadcast_input_ptr + x);
                auto       res             = is_sat ? wrapper::svqadd(broadcast_value_vec, non_broadcast_v) : svadd_z(pg, broadcast_value_vec, non_broadcast_v);
                svst1(pg, output_ptr + x, res);

                x += wrapper::svcnt<ScalarType>();
                pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
        },
        broadcast_input, non_broadcast_input, output);
    }
    else
    {
        // Clear X Dimension on execution window as we handle manually
        input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
        input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

        Iterator input1(in1, input1_win);
        Iterator input2(in2, input2_win);
        Iterator output(out, win);

        execute_window_loop(win, [&](const Coordinates &)
        {
            const auto input1_ptr = reinterpret_cast<const ScalarType *>(input1.ptr());
            const auto input2_ptr = reinterpret_cast<const ScalarType *>(input2.ptr());
            const auto output_ptr = reinterpret_cast<ScalarType *>(output.ptr());

            int      x  = window_start_x;
            svbool_t pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
            do
            {
                const auto val1 = svld1(pg, input1_ptr + x);
                const auto val2 = svld1(pg, input2_ptr + x);
                const auto res  = is_sat ? wrapper::svqadd(val1, val2) : svadd_z(pg, val1, val2);
                svst1(pg, output_ptr + x, res);

                x += wrapper::svcnt<ScalarType>();
                pg = wrapper::svwhilelt<ScalarType>(x, window_end_x);
            }
            while(svptest_any(all_true_pg, pg));
        },
        input1, input2, output);
    }
}
} // namespace cpu
} // namespace arm_compute
#endif // defined(__ARM_FEATURE_SVE)
#endif // SRC_CORE_SVE_KERNELS_ARITHMETIC_ADDITION_LIST_H