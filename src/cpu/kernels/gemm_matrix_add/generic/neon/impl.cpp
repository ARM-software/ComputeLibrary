/*
 * Copyright (c) 2016-2023 Arm Limited.
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

#include "src/cpu/kernels/gemm_matrix_add/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
void matrix_addition_f32(const ITensor *src, ITensor *dst, const Window &window, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const float32x4_t beta_f32 = vdupq_n_f32(beta);

    constexpr int window_step_x  = 16;
    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());

    Window win = window.collapse_if_possible(window, Window::DimZ);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(src, win);
    Iterator out(dst, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            const auto in_ptr  = reinterpret_cast<const float *>(in.ptr());
            const auto out_ptr = reinterpret_cast<float *>(out.ptr());

            int x = window_start_x;
            for (; x < (window_end_x - window_step_x); x += window_step_x)
            {
                float32x4x4_t       alpha_ab = vld4q_f32(out_ptr + x);
                const float32x4x4_t c        = vld4q_f32(in_ptr + x);

                // Multiply matrix C by its weight and accumulate
                alpha_ab.val[0] = vmlaq_f32(alpha_ab.val[0], c.val[0], beta_f32);
                alpha_ab.val[1] = vmlaq_f32(alpha_ab.val[1], c.val[1], beta_f32);
                alpha_ab.val[2] = vmlaq_f32(alpha_ab.val[2], c.val[2], beta_f32);
                alpha_ab.val[3] = vmlaq_f32(alpha_ab.val[3], c.val[3], beta_f32);

                vst4q_f32(out_ptr + x, alpha_ab);
            }

            // Left-over loop
            for (; x < window_end_x; ++x)
            {
                *(out_ptr + x) += *(in_ptr + x) * beta;
            }
        },
        in, out);
}
} // namespace cpu
} // namespace arm_compute
