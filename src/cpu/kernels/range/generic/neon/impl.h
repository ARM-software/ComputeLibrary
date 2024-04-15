/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
#ifndef ACL_SRC_CPU_KERNELS_RANGE_GENERIC_NEON_IMPL_H
#define ACL_SRC_CPU_KERNELS_RANGE_GENERIC_NEON_IMPL_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

#include "src/core/common/Registrars.h"
#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void neon_range_function(ITensor *output, float start, float step, const Window &window)
{
    /** SIMD vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector<T, wrapper::traits::BitWidth::W128>::tag_type;

    const auto step_vec  = wrapper::vdup_n(static_cast<T>(step), ExactTagType{});
    const auto start_vec = wrapper::vdup_n(static_cast<T>(start), ExactTagType{});
    auto       id_vec    = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16 / sizeof(T);

    Window win{window};
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator output_it(output, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            int        x       = window_start_x;
            const auto out_ptr = reinterpret_cast<T *>(output_it.ptr());
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                for (int count = 0; count < window_step_x; ++count)
                {
                    id_vec = wrapper::vsetlane(static_cast<T>(x + count), id_vec, count);
                }

                // start + step * id
                const auto res_vec = wrapper::vmla(start_vec, id_vec, step_vec);
                wrapper::vstore(out_ptr + x, res_vec);
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const auto res = start + x * step;
                *(out_ptr + x) = res;
            }
        },
        output_it);
}
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_RANGE_GENERIC_NEON_IMPL_H
