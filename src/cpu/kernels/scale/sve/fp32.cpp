/*
 * Copyright (c) 2021 Arm Limited.
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
#if defined(ARM_COMPUTE_ENABLE_SVE)
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"

#include <cmath>
#include <cstddef>

#include <arm_sve.h>

namespace arm_compute
{
namespace
{
void fp32_sve_scale_nearest(const ITensor *src, ITensor *dst, const ITensor *offsets,
                            float sampling_offset, bool align_corners, const Window &window)
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

    execute_window_loop(win, [&](const Coordinates & id)
    {
        const int32_t offset     = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z()))) * in_stride_c;
        const auto    in_hi      = static_cast<int>(align_corners ? utils::rounding::round_half_away_from_zero((id.z() + sampling_offset) * hr) : std::floor((id.z() + sampling_offset) * hr));
        const int     offset_row = in_hi * in_stride_wc;
        const auto    in_ptr     = reinterpret_cast<const float *>(in_ptr_start + in_stride_bytes_hwc * id[3]);
        const auto    out_ptr    = reinterpret_cast<float *>(out.ptr());

        // Compute S elements per iteration
        int      x  = window_start_x;
        svbool_t pg = svwhilelt_b32(x, window_end_x);
        do
        {
            // Store results
            svst1_f32(pg, out_ptr + x, svld1_f32(pg, in_ptr + offset + offset_row + x));

            x += svcntw();
            pg = svwhilelt_b32(x, window_end_x);
        }
        while(svptest_any(svptrue_b32(), pg));
    },
    out);
}

void fp32_sve_scale_bilinear(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                             BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                             bool align_corners, const Window &window)
{
    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(2), dst->info()->dimension(2), align_corners);

    Iterator  out(dst, window);
    const int in_stride_c  = src->info()->dimension(0) + src->info()->padding().left + src->info()->padding().right;
    const int in_dim_w     = src->info()->dimension(1);
    const int in_dim_h     = src->info()->dimension(2);
    const int in_stride_wc = in_stride_c * (in_dim_w + src->info()->padding().top + src->info()->padding().bottom);

    // Don't increment in Y and Z direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimZ, Window::Dimension(0, 0, 0));
    Iterator in(src, win_in);

    if(border_mode == BorderMode::CONSTANT)
    {
        const float const_border_value = static_cast<float>(constant_border_value.get<float>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto    offset = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto    dx_val = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto    dy_val = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int32_t in_hi  = std::floor((id.z() + sampling_offset) * hr - sampling_offset);
            const float *in_ptr = reinterpret_cast<const float *>(in.ptr()) + offset * in_stride_c + in_hi * in_stride_wc;

            const auto a00 = (0 <= offset && offset < in_dim_w && 0 <= in_hi && in_hi < in_dim_h) ? *in_ptr : const_border_value;
            const auto a01 = (-1 <= offset && offset < in_dim_w - 1 && 0 <= in_hi && in_hi < in_dim_h) ? *(in_ptr + in_stride_c) : const_border_value;
            const auto a10 = (0 <= offset && offset < in_dim_w && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + in_stride_wc) : const_border_value;
            const auto a11 = (-1 <= offset && offset < in_dim_w - 1 && -1 <= in_hi && in_hi < in_dim_h - 1) ? *(in_ptr + in_stride_c + in_stride_wc) : const_border_value;

            *reinterpret_cast<float *>(out.ptr()) = static_cast<float>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else if(border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto offset = *reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dx_val = *reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id.y(), id.z())));
            const auto dy_val = *reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id.y(), id.z())));
            const int  in_hi  = std::floor((id.z() + sampling_offset) * hr - sampling_offset);

            auto clamped_w  = utility::clamp<int>(offset, 0, in_dim_w - 1);
            auto clamped_w1 = utility::clamp<int>(offset + 1, 0, in_dim_w - 1);
            auto clamped_h  = utility::clamp<int>(in_hi, 0, in_dim_h - 1);
            auto clamped_h1 = utility::clamp<int>(in_hi + 1, 0, in_dim_h - 1);

            const auto a00 = *(reinterpret_cast<const float *>(in.ptr()) + clamped_w * in_stride_c + clamped_h * in_stride_wc);
            const auto a01 = *(reinterpret_cast<const float *>(in.ptr()) + clamped_w1 * in_stride_c + clamped_h * in_stride_wc);
            const auto a10 = *(reinterpret_cast<const float *>(in.ptr()) + clamped_w * in_stride_c + clamped_h1 * in_stride_wc);
            const auto a11 = *(reinterpret_cast<const float *>(in.ptr()) + clamped_w1 * in_stride_c + clamped_h1 * in_stride_wc);

            *reinterpret_cast<float *>(out.ptr()) = static_cast<float>(scale_helpers::delta_bilinear(a00, a01, a10, a11, dx_val, dy_val));
        },
        in, out);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
}
}
namespace cpu
{
void fp32_sve_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                    InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                    bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        fp32_sve_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        fp32_sve_scale_nearest(src, dst, offsets, sampling_offset, align_corners, window);
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_ENABLE_SVE