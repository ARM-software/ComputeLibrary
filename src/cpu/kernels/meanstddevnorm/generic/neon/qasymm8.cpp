/*
 * Copyright (c) 2022 Arm Limited.
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
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEAsymm.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <arm_neon.h>
namespace
{
inline float32x4_t clamp_v4f32(float32x4_t block, float32x4_t quant_min_vec, float32x4_t quant_max_vec)
{
    return vminq_f32(vmaxq_f32(block, quant_min_vec), quant_max_vec);
}
inline uint16x8_t fuse_words_f32(float32x4_t fb1, float32x4_t fb2)
{
    return vcombine_u16(vmovn_u32(vcvtq_u32_f32(fb1)), vmovn_u32(vcvtq_u32_f32(fb2)));
}
inline uint8x16_t fuse_shorts_u16(uint16x8_t sb1, uint16x8_t sb2)
{
    return vcombine_u8(vmovn_u16(sb1), vmovn_u16(sb2));
}
} // namespace

namespace arm_compute
{
namespace cpu
{
void neon_qasymm8_meanstddevnorm(ITensor *input, ITensor *output, float epsilon, const Window &window)
{
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int window_step_x  = 16;
    const int window_start_x = static_cast<int>(window.x().start());
    const int window_end_x   = static_cast<int>(window.x().end());

    const UniformQuantizationInfo qi_out        = output->info()->quantization_info().uniform();
    const float                   output_scale  = qi_out.scale;
    const int                     output_offset = qi_out.offset;

    Iterator input_itr(input, win);
    Iterator output_itr(output, win);

    const float       output_inv_scale = 1.0f / output_scale;
    const float32x4_t quant_max_vec    = vdupq_n_f32(255.0f);
    const float32x4_t quant_min_vec    = vdupq_n_f32(0.0f);

    execute_window_loop(
        win, [&](const Coordinates &)
    {
        int  x       = window_start_x;
        auto in_ptr  = reinterpret_cast<const uint8_t *>(input_itr.ptr());
        auto out_ptr = reinterpret_cast<uint8_t *>(output_itr.ptr());

        uint32x4_t sum_vec    = vdupq_n_u32(0);
        uint32x4_t sum_sq_vec = vdupq_n_u32(0);

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const uint8x16_t data         = vld1q_u8(in_ptr + x);
            sum_vec                       = vaddq_u32(sum_vec, vpaddlq_u16(vpaddlq_u8(data)));
            const uint16x8_t squares_low  = vmull_u8(vget_low_u8(data), vget_low_u8(data));
            const uint16x8_t squares_high = vmull_u8(vget_high_u8(data), vget_high_u8(data));
            sum_sq_vec                    = vaddq_u32(sum_sq_vec, vaddq_u32(vpaddlq_u16(squares_low), vpaddlq_u16(squares_high)));
        }

#ifdef __aarch64__
        sum_vec         = vpaddq_u32(sum_vec, sum_vec);
        sum_vec         = vpaddq_u32(sum_vec, sum_vec);
        uint32_t sum    = vgetq_lane_u32(sum_vec, 0);
        sum_sq_vec      = vpaddq_u32(sum_sq_vec, sum_sq_vec);
        sum_sq_vec      = vpaddq_u32(sum_sq_vec, sum_sq_vec);
        uint32_t sum_sq = vgetq_lane_u32(sum_sq_vec, 0);
#elif __arm__ // #ifdef __aarch64__
        uint32_t sum =  vgetq_lane_u32(sum_vec, 0) +
                        vgetq_lane_u32(sum_vec, 1) +
                        vgetq_lane_u32(sum_vec, 2) +
                        vgetq_lane_u32(sum_vec, 3);

        uint32_t sum_sq =   vgetq_lane_u32(sum_sq_vec, 0) +
                            vgetq_lane_u32(sum_sq_vec, 1) +
                            vgetq_lane_u32(sum_sq_vec, 2) +
                            vgetq_lane_u32(sum_sq_vec, 3);
#endif        // #ifdef __aarch64__
        for(; x < window_end_x; ++x)
        {
            auto data = static_cast<uint32_t>(*(in_ptr + x));
            sum += data;
            sum_sq += (data * data);
        }

        const float       mean      = (static_cast<float>(sum) / static_cast<float>(input->info()->dimension(0)));
        const float       var       = (static_cast<float>(sum_sq) / static_cast<float>(input->info()->dimension(0))) - (mean * mean);
        const float       stdev_inv = 1.0f / sqrtf(var + epsilon);
        const float32x4_t v_scale   = vdupq_n_f32(stdev_inv * output_inv_scale);
        const float32x4_t v_offset  = vdupq_n_f32(-mean * stdev_inv * output_inv_scale + output_offset);
        for(x = window_start_x; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const uint8x16_t data = vld1q_u8(in_ptr + x);
            float32x4_t      db1  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(data)))));
            float32x4_t      db2  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(data)))));
            float32x4_t      db3  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(data)))));
            float32x4_t      db4  = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(data)))));
            db1                   = clamp_v4f32(vaddq_f32(vmulq_f32(db1, v_scale), v_offset), quant_min_vec, quant_max_vec);
            db2                   = clamp_v4f32(vaddq_f32(vmulq_f32(db2, v_scale), v_offset), quant_min_vec, quant_max_vec);
            db3                   = clamp_v4f32(vaddq_f32(vmulq_f32(db3, v_scale), v_offset), quant_min_vec, quant_max_vec);
            db4                   = clamp_v4f32(vaddq_f32(vmulq_f32(db4, v_scale), v_offset), quant_min_vec, quant_max_vec);
            const uint8x16_t out  = fuse_shorts_u16(fuse_words_f32(db1, db2), fuse_words_f32(db3, db4));
            vst1q_u8(out_ptr + x, out);
        }

        for(; x < window_end_x; ++x)
        {
            auto          data = static_cast<float32_t>(*(in_ptr + x));
            const uint8_t res  = data * (stdev_inv * output_inv_scale) + (-mean * stdev_inv * output_inv_scale + output_offset);
            *(out_ptr + x)     = res;
        }
    },
    input_itr, output_itr);
}
} // namespace cpu
} // namespace arm_compute
