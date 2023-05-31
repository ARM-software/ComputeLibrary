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
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)

#include "arm_compute/core/TensorInfo.h"
#include "src/cpu/kernels/CpuCastKernel.h"
#include "src/cpu/kernels/cast/list.h"
#include "support/SaturateCast.h"

#include "arm_neon.h"

namespace arm_compute
{
namespace cpu
{
void neon_qasymm8_signed_to_fp16_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);
    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr = reinterpret_cast<const int8_t *>(src.ptr());
        const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());
        int        x       = window_start_x;

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const int8x16_t texels_s8 = vld1q_s8(src_ptr + x);

            const int16x8x2_t texels =
            {
                {
                    vmovl_s8(vget_low_s8(texels_s8)),
                    vmovl_s8(vget_high_s8(texels_s8))
                }
            };
            vst1q_f16(dst_ptr + x, vcvtq_f16_s16(texels.val[0]));
            vst1q_f16(dst_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
        }
    },
    src, dst);
}

void neon_s32_to_fp16_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr = reinterpret_cast<const int32_t *>(src.ptr());
        const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const float32x4x4_t texels =
            {
                {
                    vcvtq_f32_s32(vld1q_s32(src_ptr + x)),
                    vcvtq_f32_s32(vld1q_s32(src_ptr + x + 4)),
                    vcvtq_f32_s32(vld1q_s32(src_ptr + x + 8)),
                    vcvtq_f32_s32(vld1q_s32(src_ptr + x + 12))
                }
            };

            vst1q_f16(dst_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
            vst1q_f16(dst_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
        }
    },
    src, dst);
}

void neon_fp32_to_fp16_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr = reinterpret_cast<const float *>(src.ptr());
        const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const float32x4x4_t texels =
            {
                {
                    vld1q_f32(src_ptr + x),
                    vld1q_f32(src_ptr + x + 4),
                    vld1q_f32(src_ptr + x + 8),
                    vld1q_f32(src_ptr + x + 12)
                }
            };

            vst1q_f16(dst_ptr + x, vcombine_f16(vcvt_f16_f32(texels.val[0]), vcvt_f16_f32(texels.val[1])));
            vst1q_f16(dst_ptr + x + 8, vcombine_f16(vcvt_f16_f32(texels.val[2]), vcvt_f16_f32(texels.val[3])));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
        }
    },
    src, dst);
}

void neon_fp16_to_other_dt_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);
    switch(_dst->info()->data_type())
    {
        case DataType::QASYMM8_SIGNED:
        {
            /* Down-conversion F16 -> QASYMM8_SIGNED (Always saturating) */
            execute_window_loop(win, [&](const Coordinates &)
            {
                const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<int8_t *>(dst.ptr());

                int x = window_start_x;
                for(; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float16x8x2_t texels =
                    {
                        {
                            vld1q_f16(src_ptr + x),
                            vld1q_f16(src_ptr + x + 8),
                        }
                    };

                    vst1q_s8(dst_ptr + x, vcombine_s8(vqmovn_s16(vcvtq_s16_f16(texels.val[0])), vqmovn_s16(vcvtq_s16_f16(texels.val[1]))));
                }

                // Compute left-over elements
                for(; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = utils::cast::saturate_cast<int8_t>(*(src_ptr + x));
                }
            },
            src, dst);
            break;
        }
        case DataType::QASYMM8:
        case DataType::U8:
        {
            /* Down-conversion F16 -> QASYMM8/U8 (Always saturating) */
            execute_window_loop(win, [&](const Coordinates &)
            {
                const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<uint8_t *>(dst.ptr());

                int x = window_start_x;
                for(; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float16x8x2_t texels =
                    {
                        {
                            vld1q_f16(src_ptr + x),
                            vld1q_f16(src_ptr + x + 8),
                        }
                    };

                    vst1q_u8(dst_ptr + x, vcombine_u8(vqmovun_s16(vcvtq_s16_f16(texels.val[0])), vqmovun_s16(vcvtq_s16_f16(texels.val[1]))));
                }

                // Compute left-over elements
                for(; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = utils::cast::saturate_cast<uint8_t>(*(src_ptr + x));
                }

            },
            src, dst);
            break;
        }
        case DataType::F32:
        {
            /* Up-conversion F16 -> F32 */
            execute_window_loop(win, [&](const Coordinates &)
            {
                const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<float *>(dst.ptr());

                int x = window_start_x;
                for(; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float16x8x2_t texels =
                    {
                        {
                            vld1q_f16(src_ptr + x),
                            vld1q_f16(src_ptr + x + 8)
                        }
                    };
                    vst1q_f32(dst_ptr + x, vcvt_f32_f16(vget_low_f16(texels.val[0])));
                    vst1q_f32(dst_ptr + x + 4, vcvt_f32_f16(vget_high_f16(texels.val[0])));
                    vst1q_f32(dst_ptr + x + 8, vcvt_f32_f16(vget_low_f16(texels.val[1])));
                    vst1q_f32(dst_ptr + x + 12, vcvt_f32_f16(vget_high_f16(texels.val[1])));
                }

                // Compute left-over elements
                for(; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<float>(*(src_ptr + x));
                }
            },
            src, dst);
            break;
        }
        case DataType::S32:
        {
            /* Up-conversion F16 -> S32 */
            execute_window_loop(win, [&](const Coordinates &)
            {
                const auto src_ptr = reinterpret_cast<const float16_t *>(src.ptr());
                const auto dst_ptr = reinterpret_cast<int32_t *>(dst.ptr());

                int x = window_start_x;
                for(; x <= (window_end_x - window_step_x); x += window_step_x)
                {
                    const float16x8x2_t texels =
                    {
                        {
                            vld1q_f16(src_ptr + x),
                            vld1q_f16(src_ptr + x + 8)
                        }
                    };

                    vst1q_s32(dst_ptr + x, vcvtq_s32_f32(vcvt_f32_f16(vget_low_f16(texels.val[0]))));
                    vst1q_s32(dst_ptr + x + 4, vcvtq_s32_f32(vcvt_f32_f16(vget_high_f16(texels.val[0]))));
                    vst1q_s32(dst_ptr + x + 8, vcvtq_s32_f32(vcvt_f32_f16(vget_low_f16(texels.val[1]))));
                    vst1q_s32(dst_ptr + x + 12, vcvtq_s32_f32(vcvt_f32_f16(vget_high_f16(texels.val[1]))));
                }

                // Compute left-over elements
                for(; x < window_end_x; ++x)
                {
                    *(dst_ptr + x) = static_cast<int32_t>(*(src_ptr + x));
                }
            },
            src, dst);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("dst data type not supported");
    }
}

void neon_u8_to_fp16_cast(const ITensor *_src, ITensor *_dst, const ThreadInfo &info, ConvertPolicy _policy, const Window &window)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_UNUSED(_policy);

    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());
    const int  window_step_x  = 16;

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);
    ARM_COMPUTE_ERROR_ON(_src == _dst);

    ARM_COMPUTE_ERROR_ON_NULLPTR(_src, _dst);

    Window win{ window };
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator src(_src, win);
    Iterator dst(_dst, win);
    /* Up-conversion U8 -> F16 */
    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto src_ptr = reinterpret_cast<const uint8_t *>(src.ptr());
        const auto dst_ptr = reinterpret_cast<float16_t *>(dst.ptr());

        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            const uint8x16_t texels_u8 = vld1q_u8(src_ptr + x);

            const int16x8x2_t texels =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(texels_u8))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(texels_u8)))
                }
            };
            vst1q_f16(dst_ptr + x, vcvtq_f16_s16(texels.val[0]));
            vst1q_f16(dst_ptr + x + 8, vcvtq_f16_s16(texels.val[1]));
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(dst_ptr + x) = static_cast<float16_t>(*(src_ptr + x));
        }
    },
    src, dst);
    return;
}

} // namespace cpu
} // namespace arm_compute
#endif /* #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
