/*
 * Copyright (c) 2021-2022 Arm Limited.
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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"

#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"

#include <arm_sve.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace
{
void fp16_sve_scale_nearest(const ITensor *src,
                            ITensor       *dst,
                            const ITensor *offsets,
                            float          sampling_offset,
                            bool           align_corners,
                            const Window  &window)
{
    const size_t in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
    const size_t in_stride_w  = src->info()->dimension(1) + src->info()->padding().top + src->info()->padding().bottom;
    const size_t in_stride_wc = in_stride_w * in_stride_c;
    const size_t in_dim_h     = src->info()->dimension(2);

    // Compute the ratio between source height and destination height
    const auto hr             = scale_utils::calculate_resize_ratio(in_dim_h, dst->info()->dimension(2), align_corners);
    const auto window_start_x = static_cast<int32_t>(window.x().start());
    const auto window_end_x   = static_cast<int32_t>(window.x().end());

    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    Iterator out(dst, win);

    const uint8_t     *in_ptr_start        = src->buffer() + src->info()->offset_first_element_in_bytes();
    const unsigned int in_stride_bytes_hwc = src->info()->strides_in_bytes()[3];

    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            const int32_t offset =
                *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z()))) * in_stride_c;
            const auto in_hi = static_cast<int>(
                align_corners ? utils::rounding::round_half_away_from_zero((id.z() + sampling_offset) * hr)
                              : std::floor((id.z() + sampling_offset) * hr));
            const int  offset_row = in_hi * in_stride_wc;
            const auto in_ptr     = reinterpret_cast<const float16_t *>(in_ptr_start + in_stride_bytes_hwc * id[3]);
            const auto out_ptr    = reinterpret_cast<float16_t *>(out.ptr());

            // Compute S elements per iteration
            int      x  = window_start_x;
            svbool_t pg = svwhilelt_b16(x, window_end_x);
            do
            {
                // Store results
                svst1_f16(pg, out_ptr + x, svld1_f16(pg, in_ptr + offset + offset_row + x));

                x += svcntw();
                pg = svwhilelt_b16(x, window_end_x);
            } while (svptest_any(svptrue_b16(), pg));
        },
        out);
}
} // namespace
namespace cpu
{
void fp16_sve_scale(const ITensor      *src,
                    ITensor            *dst,
                    const ITensor      *offsets,
                    const ITensor      *dx,
                    const ITensor      *dy,
                    InterpolationPolicy policy,
                    BorderMode          border_mode,
                    PixelValue          constant_border_value,
                    float               sampling_offset,
                    bool                align_corners,
                    const Window       &window)
{
    ARM_COMPUTE_UNUSED(dx, dy, border_mode, constant_border_value);
    if (policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        fp16_sve_scale_nearest(src, dst, offsets, sampling_offset, align_corners, window);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}
} // namespace cpu
} // namespace arm_compute
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */
