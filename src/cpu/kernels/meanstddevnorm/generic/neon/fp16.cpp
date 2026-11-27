/*
 * Copyright (c) 2022-2026 Arm Limited.
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

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/CpuTypes.h"
#include "src/cpu/kernels/meanstddevnorm/generic/neon/impl.h"

#include <algorithm>

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

            // First pass: the mean, accumulated in f32
            float32x4x2_t sum_vec = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8_t data = vld1q_f16(in_ptr + x);
                sum_vec.val[0]   = vaddq_f32(sum_vec.val[0], vcvt_f32_f16(vget_low_f16(data)));
                sum_vec.val[1]   = vaddq_f32(sum_vec.val[1], vcvt_f32_f16(vget_high_f16(data)));
            }
            float sum = vaddvq_f32(vpaddq_f32(sum_vec.val[0], sum_vec.val[1]));

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                sum += static_cast<float>(*(in_ptr + x));
            }
            const float mean = sum / input->info()->dimension(0);

            // Second pass: sums of the centred values d = x - mean in f32 (corrected two-pass algorithm). Sum(d^2) gives
            // the variance without the cancellation of E[x^2] - mean^2, and sum(d) corrects the rounding error of the
            // mean, which would otherwise be added to the variance and leak into the output, e.g. for constant rows.
            // The mean stays in f32: rounded to f16 it is off by up to half an f16 step, more than many variances.
            const float32x4_t mean_vec   = vdupq_n_f32(mean);
            float32x4x2_t     sum_d_vec  = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            float32x4x2_t     sum_sq_vec = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8_t data  = vld1q_f16(in_ptr + x);
                float32x4_t dl    = vsubq_f32(vcvt_f32_f16(vget_low_f16(data)), mean_vec);
                float32x4_t dh    = vsubq_f32(vcvt_f32_f16(vget_high_f16(data)), mean_vec);
                sum_d_vec.val[0]  = vaddq_f32(sum_d_vec.val[0], dl);
                sum_d_vec.val[1]  = vaddq_f32(sum_d_vec.val[1], dh);
                sum_sq_vec.val[0] = vfmaq_f32(sum_sq_vec.val[0], dl, dl);
                sum_sq_vec.val[1] = vfmaq_f32(sum_sq_vec.val[1], dh, dh);
            }
            float sum_d  = vaddvq_f32(vaddq_f32(sum_d_vec.val[0], sum_d_vec.val[1]));
            float sum_sq = vaddvq_f32(vaddq_f32(sum_sq_vec.val[0], sum_sq_vec.val[1]));

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                const float centred = static_cast<float>(*(in_ptr + x)) - mean;
                sum_d += centred;
                sum_sq += centred * centred;
            }

            // shift is the rounding error of the mean; the clamp keeps rounding from making a zero variance negative
            const float shift      = sum_d / input->info()->dimension(0);
            const float var        = std::max(sum_sq / input->info()->dimension(0) - shift * shift, 0.f);
            const float stddev_inv = 1.f / sqrt(var + epsilon);

            // Normalise in f32 and convert once when storing
            const float32x4_t shift_vec      = vdupq_n_f32(shift);
            const float32x4_t stddev_inv_vec = vdupq_n_f32(stddev_inv);
            for (x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                float16x8_t data = vld1q_f16(in_ptr + x);
                float32x4_t dl   = vsubq_f32(vsubq_f32(vcvt_f32_f16(vget_low_f16(data)), mean_vec), shift_vec);
                float32x4_t dh   = vsubq_f32(vsubq_f32(vcvt_f32_f16(vget_high_f16(data)), mean_vec), shift_vec);
                float16x8_t res  = vcombine_f16(vcvt_f16_f32(vmulq_f32(dl, stddev_inv_vec)),
                                                vcvt_f16_f32(vmulq_f32(dh, stddev_inv_vec)));
                // Store results
                vst1q_f16(out_ptr + x, res);
            }
            for (; x < window_end_x; ++x)
            {
                const float centred = (static_cast<float>(*(in_ptr + x)) - mean) - shift;
                *(out_ptr + x)      = static_cast<float16_t>(centred * stddev_inv);
            }
        },
        input_itr, output_itr);
}

void neon_fp16_meanstddevnorm(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "neon_fp16_meanstddevnorm");
    mean_stddev_normalization<float16_t, 8>(input, output, epsilon, window);
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
