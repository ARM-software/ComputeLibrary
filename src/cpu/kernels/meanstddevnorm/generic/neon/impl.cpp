/*
 * Copyright (c) 2019-2023, 2026 Arm Limited.
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

#include <algorithm>

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

    // Sum of the lanes of a vector, added pairwise
    const auto add_lanes = [](const auto &vec)
    {
        auto res = wrapper::vpadd(wrapper::vgethigh(vec), wrapper::vgetlow(vec));
        for (int i = 0; i < size / 4; ++i)
        {
            res = wrapper::vpadd(res, res);
        }
        return wrapper::vgetlane(res, 0);
    };

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            int  x       = window_start_x;
            auto in_ptr  = reinterpret_cast<const ScalarType *>(input_itr.ptr());
            auto out_ptr = reinterpret_cast<ScalarType *>(output_itr.ptr());

            // First pass: the mean
            auto sum_vec = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                sum_vec = wrapper::vadd(sum_vec, wrapper::vloadq(in_ptr + x));
            }
            auto sum = add_lanes(sum_vec);

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                sum += *(in_ptr + x);
            }
            const ScalarType mean     = sum / input->info()->dimension(0);
            const auto       mean_vec = wrapper::vdup_n(mean, ExactTagType{});

            // Second pass: sums of the centred values d = x - mean (corrected two-pass algorithm). Sum(d^2) gives the
            // variance without the cancellation of E[x^2] - mean^2, and sum(d) corrects the rounding error of the mean,
            // which would otherwise be added to the variance and leak into the output, e.g. for constant rows.
            auto sum_d_vec  = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});
            auto sum_sq_vec = wrapper::vdup_n(static_cast<ScalarType>(0.f), ExactTagType{});
            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                const auto centred = wrapper::vsub(wrapper::vloadq(in_ptr + x), mean_vec);
                sum_d_vec          = wrapper::vadd(sum_d_vec, centred);
                sum_sq_vec         = wrapper::vadd(sum_sq_vec, wrapper::vmul(centred, centred));
            }
            auto sum_d  = add_lanes(sum_d_vec);
            auto sum_sq = add_lanes(sum_sq_vec);

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const ScalarType centred = *(in_ptr + x) - mean;
                sum_d += centred;
                sum_sq += centred * centred;
            }

            // shift is the rounding error of the mean; the clamp keeps rounding from making a zero variance negative
            const ScalarType shift = sum_d / input->info()->dimension(0);
            const ScalarType var =
                std::max(sum_sq / input->info()->dimension(0) - shift * shift, static_cast<ScalarType>(0.f));
            const ScalarType stddev_inv = 1.f / sqrt(var + epsilon);

            const auto shift_vec      = wrapper::vdup_n(shift, ExactTagType{});
            const auto stddev_inv_vec = wrapper::vdup_n(stddev_inv, ExactTagType{});
            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                auto centred = wrapper::vsub(wrapper::vsub(wrapper::vloadq(in_ptr + x), mean_vec), shift_vec);
                // Store results
                wrapper::vstore(out_ptr + x, wrapper::vmul(centred, stddev_inv_vec));
            }
            for (; x < window_end_x; ++x)
            {
                *(out_ptr + x) = ((*(in_ptr + x) - mean) - shift) * stddev_inv;
            }
        },
        input_itr, output_itr);
}
template void mean_stddev_normalization<float, 4>(ITensor *input, ITensor *output, float epsilon, const Window &window);
} // namespace cpu
} // namespace arm_compute
