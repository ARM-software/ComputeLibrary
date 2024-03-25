/*
 * Copyright (c) 2019-2023 Arm Limited.
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

#include "src/cpu/kernels/meanstddevnorm/generic/neon/impl.h"

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename ScalarType, int size>
void mean_stddev_normalization(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    using ExactTagType = typename wrapper::traits::neon_vector<ScalarType, size>::tag_type;

    // Set build options
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x  = size;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Iterator input_itr(input, win);
    Iterator output_itr(output, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            int  x       = window_start_x;
            auto in_ptr  = reinterpret_cast<const ScalarType *>(input_itr.ptr());
            auto out_ptr = reinterpret_cast<ScalarType *>(output_itr.ptr());

            auto sum_vec    = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});
            auto sum_sq_vec = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});

            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                auto data  = wrapper::vloadq(in_ptr + x);
                sum_vec    = wrapper::vadd(sum_vec, data);
                sum_sq_vec = wrapper::vadd(sum_sq_vec, wrapper::vmul(data, data));
            }

            auto sum_carry_res    = wrapper::vpadd(wrapper::vgethigh(sum_vec), wrapper::vgetlow(sum_vec));
            auto sum_sq_carry_res = wrapper::vpadd(wrapper::vgethigh(sum_sq_vec), wrapper::vgetlow(sum_sq_vec));
            for (int i = 0; i < size / 4; ++i)
            {
                sum_carry_res    = wrapper::vpadd(sum_carry_res, sum_carry_res);
                sum_sq_carry_res = wrapper::vpadd(sum_sq_carry_res, sum_sq_carry_res);
            }

            auto sum    = wrapper::vgetlane(sum_carry_res, 0);
            auto sum_sq = wrapper::vgetlane(sum_sq_carry_res, 0);

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                ScalarType data = *(in_ptr + x);
                sum += data;
                sum_sq += data * data;
            }

            ScalarType mean       = sum / input->info()->dimension(0);
            ScalarType var        = (sum_sq / input->info()->dimension(0)) - (mean * mean);
            ScalarType stddev_inv = 1.f / sqrt(var + epsilon);

            auto mean_vec       = wrapper::vdup_n(mean, ExactTagType{});
            auto stddev_inv_vec = wrapper::vdup_n(stddev_inv, ExactTagType{});
            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                auto data = wrapper::vloadq(in_ptr + x);
                auto res  = wrapper::vmul(wrapper::vsub(data, mean_vec), stddev_inv_vec);
                // Store results
                wrapper::vstore(out_ptr + x, res);
            }
            for (; x < window_end_x; ++x)
            {
                *(out_ptr + x) = (*(in_ptr + x) - mean) * stddev_inv;
            }
        },
        input_itr, output_itr);
}
template void mean_stddev_normalization<float, 4>(ITensor *input, ITensor *output, float epsilon, const Window &window);
} // namespace cpu
} // namespace arm_compute
