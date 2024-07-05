/*
 * Copyright (c) 2022-2024 Arm Limited.
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

#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/CpuTypes.h"
#include "src/cpu/kernels/meanstddevnorm/generic/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
template <>
void mean_stddev_normalization<float16_t, 8>(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    // Set build options
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x  = 8;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Iterator input_itr(input, win);
    Iterator output_itr(output, win);

    execute_window_loop(
        win,
        [&](const Coordinates &)
        {
            int  x       = window_start_x;
            auto in_ptr  = reinterpret_cast<const float16_t *>(input_itr.ptr());
            auto out_ptr = reinterpret_cast<float16_t *>(output_itr.ptr());

            float32x4x2_t sum_vec = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};

            float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);

            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8_t data = vld1q_f16(in_ptr + x);
                float32x4_t dl   = vcvt_f32_f16(vget_low_f16(data));
                float32x4_t dh   = vcvt_f32_f16(vget_high_f16(data));
                sum_vec.val[0]   = vaddq_f32(sum_vec.val[0], dl);
                sum_vec.val[1]   = vaddq_f32(sum_vec.val[1], dh);
                sum_sq_vec       = vaddq_f32(sum_sq_vec, vmulq_f32(dl, dl));
                sum_sq_vec       = vaddq_f32(sum_sq_vec, vmulq_f32(dh, dh));
            }

            float32x4_t sum_carry_res = vpaddq_f32(sum_vec.val[0], sum_vec.val[1]);
            float       sum           = vaddvq_f32(sum_carry_res);
            float       sum_sq        = vaddvq_f32(sum_sq_vec);

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const float fdata = static_cast<float>(*(in_ptr + x));
                sum += fdata;
                sum_sq += fdata * fdata;
            }

            float16_t mean       = static_cast<float16_t>(sum / input->info()->dimension(0));
            float     var        = (sum_sq / input->info()->dimension(0)) - (mean * mean);
            float16_t stddev_inv = static_cast<float16_t>(1.f / sqrt(var + epsilon));

            float16x8_t mean_vec       = vdupq_n_f16(mean);
            float16x8_t stddev_inv_vec = vdupq_n_f16(stddev_inv);

            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8_t data = vld1q_f16(in_ptr + x);
                float16x8_t res  = vmulq_f16(vsubq_f16(data, mean_vec), stddev_inv_vec);
                // Store results
                vst1q_f16(out_ptr + x, res);
            }
            for (; x < window_end_x; ++x)
            {
                *(out_ptr + x) = (*(in_ptr + x) - mean) * stddev_inv;
            }
        },
        input_itr, output_itr);
}

void neon_fp16_meanstddevnorm(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    return mean_stddev_normalization<float16_t, 8>(input, output, epsilon, window);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
