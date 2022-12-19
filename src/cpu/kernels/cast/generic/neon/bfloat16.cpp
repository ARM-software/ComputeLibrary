/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#if defined(ARM_COMPUTE_ENABLE_BF16)

#include "arm_compute/core/TensorInfo.h"
#include "src/cpu/kernels/CpuCastKernel.h"
#include "src/cpu/kernels/cast/list.h"

namespace arm_compute
{
namespace cpu
{
void neon_fp32_to_bfloat16_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());
    constexpr int window_step_x  = 8;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);
    /* Down-conversion F32 -> BFLOAT16 */
    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr     = reinterpret_cast<const float *>(src.ptr());
        const auto dst_ptr     = reinterpret_cast<bfloat16_t *>(dst.ptr());
        int        x           = window_start_x;
        const int  right_bound = (window_end_x - window_step_x);
        for(; x <= right_bound; x += window_step_x)
        {
            const auto vbf16_0 = vcombine_bf16(
                                     vcvt_bf16_f32(vld1q_f32(src_ptr + x)),
                                     vcvt_bf16_f32(vld1q_f32(src_ptr + x + 4)));
            vst1q_bf16(dst_ptr + x, vbf16_0);
        }
        for(; x < window_end_x; ++x)
        {
            *(reinterpret_cast<bfloat16 *>(dst.ptr()) + x) = *(src_ptr + x);
        }
    },
    src, dst);
}

void neon_bfloat16_to_fp32_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto    window_start_x = static_cast<int>(window.x().start());
    const auto    window_end_x   = static_cast<int>(window.x().end());
    constexpr int window_step_x  = 8;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);
    /* Up-conversion BFLOAT16 -> F32 */
    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr = reinterpret_cast<const bfloat16_t *>(src.ptr());
        const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

        int       x = window_start_x;
        const int right_bound(window_end_x - window_step_x);
        for(; x <= right_bound; x += window_step_x)
        {
            const bfloat16x8_t vinput = vld1q_bf16(src_ptr + x);
            vst1q_f32(dst_ptr + x, vcvt_f32_bf16(vget_low_bf16(vinput)));
            vst1q_f32(dst_ptr + x + 4, vcvt_f32_bf16(vget_high_bf16(vinput)));
        }
        for(; x < window_end_x; ++x)
        {
            *(dst_ptr + x) = float(*(reinterpret_cast<const bfloat16 *>(src_ptr) + x));
        }
    },
    src, dst);
}

} // namespace cpu
} // namespace arm_compute

#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
