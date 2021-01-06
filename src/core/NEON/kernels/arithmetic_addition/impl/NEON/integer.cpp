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
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/Traits.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
void arithmetic_addition_U8_U8_S16_neon(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const uint8_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(policy == ConvertPolicy::WRAP)
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input1_ptr + x)));
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vadd(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = static_cast<int16_t>(*(input1_ptr + x)) + static_cast<int16_t>(*(input2_ptr + x));
            }
        }
        else
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input1_ptr + x)));
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vqadd(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = wrapper::add_sat(static_cast<int16_t>(*(input1_ptr + x)),
                                                     static_cast<int16_t>(*(input2_ptr + x)));
            }
        }
    },
    input1, input2, output);
}

void arithmetic_addition_S16_U8_S16_neon(const ITensor *in1, const ITensor *in2, ITensor *out, const ConvertPolicy &policy, const Window &window)
{
    // Create input windows
    Window win        = window;
    Window input1_win = window.broadcast_if_dimension_le_one(in1->info()->tensor_shape());
    Window input2_win = window.broadcast_if_dimension_le_one(in2->info()->tensor_shape());

    // Clear X Dimension on execution window as we handle manually
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input1_win.set(Window::DimX, Window::Dimension(0, 1, 1));
    input2_win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input1(in1, input1_win);
    Iterator input2(in2, input2_win);
    Iterator output(out, win);

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto input1_ptr = reinterpret_cast<const int16_t *>(input1.ptr());
        const auto input2_ptr = reinterpret_cast<const uint8_t *>(input2.ptr());
        const auto output_ptr = reinterpret_cast<int16_t *>(output.ptr());

        if(policy == ConvertPolicy::WRAP)
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = wrapper::vloadq(input1_ptr + x);
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vadd(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = *(input1_ptr + x) + static_cast<int16_t>(*(input2_ptr + x));
            }
        }
        else
        {
            // Compute S elements per iteration
            int x = window_start_x;
            for(; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vin1 = wrapper::vloadq(input1_ptr + x);
                const auto vin2 = vreinterpretq_s16_u16(wrapper::vmovl(wrapper::vload(input2_ptr + x)));
                wrapper::vstore(output_ptr + x, wrapper::vqadd(vin1, vin2));
            }

            // Compute left-over elements
            for(; x < window_end_x; ++x)
            {
                *(output_ptr + x) = wrapper::add_sat(*(input1_ptr + x), static_cast<int16_t>(*(input2_ptr + x)));
            }
        }
    },
    input1, input2, output);
}

void arithmetic_addition_U8_S16_S16_neon(const ITensor *input1, const ITensor *input2, ITensor *output, const ConvertPolicy &policy, const Window &window)
{
    // Simply swap the two input buffers:
    arithmetic_addition_S16_U8_S16_neon(input2, input1, output, policy, window);
}
} // namespace cpu
} // namespace arm_compute