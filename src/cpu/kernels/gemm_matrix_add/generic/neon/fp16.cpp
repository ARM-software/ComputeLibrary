/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "src/cpu/kernels/gemm_matrix_add/generic/neon/impl.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace
{
void matrix_addition_f16(const ITensor *src, ITensor *dst, const Window &window, float beta)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    const float16x8_t beta_f16 = vdupq_n_f16(beta);

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
            const auto in_ptr  = reinterpret_cast<const float16_t *>(in.ptr());
            const auto out_ptr = reinterpret_cast<float16_t *>(out.ptr());

            int x = window_start_x;
            for (; x < (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8x2_t       alpha_ab = vld2q_f16(out_ptr + x);
                const float16x8x2_t c        = vld2q_f16(in_ptr + x);
                // Multiply matrix C by its weight and accumulate
                alpha_ab.val[0] = vaddq_f16(alpha_ab.val[0], vmulq_f16(c.val[0], beta_f16));
                alpha_ab.val[1] = vaddq_f16(alpha_ab.val[1], vmulq_f16(c.val[1], beta_f16));

                vst2q_f16(out_ptr + x, alpha_ab);
            }

            // Left-over loop
            for (; x < window_end_x; ++x)
            {
                *(out_ptr + x) += *(in_ptr + x) * static_cast<float16_t>(beta);
            }
        },
        in, out);
}
} // namespace
void neon_fp16_gemm_matrix_add(const ITensor *src, ITensor *dst, const Window &window, float beta)
{
    return matrix_addition_f16(src, dst, window, beta);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
