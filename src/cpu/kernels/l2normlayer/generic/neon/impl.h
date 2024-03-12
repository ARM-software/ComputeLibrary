/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H
#define SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/common/Registrars.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <cstddef>

namespace arm_compute
{
namespace cpu
{
template <typename T, int S>
void l2_normalize_x(const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window)
{
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const int  window_step_x  = 16 / data_size_from_type(in->info()->data_type());
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input_it(in, win_collapsed);
    Iterator sum_it(sum, win_collapsed);
    Iterator output_it(out, win_collapsed);

    execute_window_loop(
        win_collapsed,
        [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

            const T    sum_value      = *reinterpret_cast<const T *>(sum_it.ptr());
            const T    norm_value     = static_cast<T>(1.f) / std::sqrt(std::max(sum_value, static_cast<T>(epsilon)));
            const auto vec_norm_value = wrapper::vdup_n(norm_value, ExactTagType{});

            // Compute elements over vector steps
            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                wrapper::vstore(out_ptr + x, wrapper::vmul(wrapper::vloadq(in_ptr + x), vec_norm_value));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                out_ptr[x] = in_ptr[x] * norm_value;
            }
        },
        input_it, sum_it, output_it);
}

template <typename T, int S>
void l2_normalize_yz(
    const ITensor *in, const ITensor *sum, ITensor *out, float epsilon, const Window &window, size_t axis)
{
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    const int  window_step_x  = 16 / data_size_from_type(in->info()->data_type());
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Window window_sum(win);
    window_sum.set(axis, Window::Dimension(0, 0, 0));

    Iterator input_it(in, win);
    Iterator sum_it(sum, window_sum);
    Iterator output_it(out, win);

    const auto vec_eps = wrapper::vdup_n(static_cast<T>(epsilon), ExactTagType{});

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const T *>(input_it.ptr());
            const auto sum_ptr = reinterpret_cast<const T *>(sum_it.ptr());
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());

            // Compute elements over vector steps
            int x = window_start_x;
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto vec_norm_value = wrapper::vinvsqrt(wrapper::vmax(wrapper::vloadq(sum_ptr + x), vec_eps));
                wrapper::vstore(out_ptr + x, wrapper::vmul(wrapper::vloadq(in_ptr + x), vec_norm_value));
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const T norm_value = static_cast<T>(1.f) / std::sqrt(std::max(sum_ptr[x], static_cast<T>(epsilon)));
                out_ptr[x]         = in_ptr[x] * norm_value;
            }
        },
        input_it, sum_it, output_it);
}
} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_L2NORMLAYER_LIST_H
