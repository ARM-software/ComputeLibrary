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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/utils/ScaleUtils.h"
#include "support/Rounding.h"

#include <arm_sve.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace
{
void qasymm8_signed_sve_scale_nearest(const ITensor *src, ITensor *dst, const ITensor *offsets,
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
        const auto    in_ptr     = reinterpret_cast<const int8_t *>(in_ptr_start + in_stride_bytes_hwc * id[3]);
        const auto    out_ptr    = reinterpret_cast<int8_t *>(out.ptr());

        // Compute S elements per iteration
        int      x  = window_start_x;
        svbool_t pg = svwhilelt_b8(x, window_end_x);
        do
        {
            // Store results
            svst1_s8(pg, out_ptr + x, svld1_s8(pg, in_ptr + offset + offset_row + x));

            x += svcntw();
            pg = svwhilelt_b8(x, window_end_x);
        }
        while(svptest_any(svptrue_b8(), pg));
    },
    out);
}

void qasymm8_signed_sve_scale_bilinear(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                                       BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                                       bool align_corners, const Window &window)
{
    // Data layout is NHWC
    const int idx_width  = 1;
    const int idx_height = 2;

    // Compute the ratio between source height and destination height
    const auto hr = scale_utils::calculate_resize_ratio(src->info()->dimension(idx_height), dst->info()->dimension(idx_height), align_corners);
    Window     win_off;
    win_off.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_off.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(idx_width, Window::Dimension(0, 0, 0));
    win_in.set(idx_height, Window::Dimension(0, 0, 0));

    for(size_t d = Window::DimZ; d < offsets->info()->num_dimensions(); ++d)
    {
        win_off.set(d, Window::Dimension(0, 0, 0));
    }

    Iterator in(src, win_in);
    Iterator out(dst, window);

    const int32_t in_dim_w = src->info()->dimension(idx_width);
    const int32_t in_dim_h = src->info()->dimension(idx_height);
    const int32_t stride_w = src->info()->strides_in_bytes()[idx_width];
    const int32_t stride_h = src->info()->strides_in_bytes()[idx_height];

    const UniformQuantizationInfo iq_info = src->info()->quantization_info().uniform();
    const UniformQuantizationInfo oq_info = dst->info()->quantization_info().uniform();

    if(border_mode == BorderMode::CONSTANT)
    {
        const int8_t const_border_value = static_cast<int8_t>(constant_border_value.get<int8_t>());
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int32_t index_h       = std::floor((id[idx_height] + sampling_offset) * hr - sampling_offset);
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const int8_t *>(in.ptr());

            const auto a00 = (0 <= index_w && index_w < in_dim_w && 0 <= index_h && index_h < in_dim_h) ?
                             (*(pixel_row_ptr + index_w * stride_w + index_h * stride_h)) :
                             const_border_value;
            const auto a01 = (-1 <= index_w && index_w < in_dim_w - 1 && 0 <= index_h && index_h < in_dim_h) ?
                             (*(pixel_row_ptr + (index_w + 1) * stride_w + index_h * stride_h)) :
                             const_border_value;
            const auto a10 = (0 <= index_w && index_w < in_dim_w && -1 <= index_h && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + index_w * stride_w + (index_h + 1) * stride_h)) :
                             const_border_value;
            const auto a11 = (-1 <= index_w && index_w < in_dim_w - 1 && -1 <= index_h && index_h < in_dim_h - 1) ?
                             (*(pixel_row_ptr + (index_w + 1) * stride_w + (index_h + 1) * stride_h)) :
                             const_border_value;

            const float inp00                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a00, iq_info);
            const float inp01                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a01, iq_info);
            const float inp10                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a10, iq_info);
            const float inp11                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a11, iq_info);
            *reinterpret_cast<int8_t *>(out.ptr()) = Qasymm8QuantizationHelper<int8_t>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
        },
        in, out);
    }
    else if(border_mode == BorderMode::REPLICATE)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const int     index_h       = std::floor((id[idx_height] + sampling_offset) * hr - sampling_offset);
            const int32_t index_w       = *(reinterpret_cast<const int32_t *>(offsets->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dx_val        = *(reinterpret_cast<const float *>(dx->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    dy_val        = *(reinterpret_cast<const float *>(dy->ptr_to_element(Coordinates(id[idx_width], id[idx_height]))));
            const auto    pixel_row_ptr = reinterpret_cast<const int8_t *>(in.ptr());

            auto clamped_w  = utility::clamp<int>(index_w, 0, in_dim_w - 1);
            auto clamped_w1 = utility::clamp<int>(index_w + 1, 0, in_dim_w - 1);
            auto clamped_h  = utility::clamp<int>(index_h, 0, in_dim_h - 1);
            auto clamped_h1 = utility::clamp<int>(index_h + 1, 0, in_dim_h - 1);

            const auto a00 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h * stride_h);
            const auto a01 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h * stride_h);
            const auto a10 = *(pixel_row_ptr + clamped_w * stride_w + clamped_h1 * stride_h);
            const auto a11 = *(pixel_row_ptr + clamped_w1 * stride_w + clamped_h1 * stride_h);

            const float inp00                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a00, iq_info);
            const float inp01                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a01, iq_info);
            const float inp10                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a10, iq_info);
            const float inp11                      = Qasymm8QuantizationHelper<int8_t>::dequantize(a11, iq_info);
            *reinterpret_cast<int8_t *>(out.ptr()) = Qasymm8QuantizationHelper<int8_t>::quantize(scale_helpers::delta_bilinear(inp00, inp01, inp10, inp11, dx_val, dy_val), oq_info);
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
void qasymm8_signed_sve_scale(const ITensor *src, ITensor *dst, const ITensor *offsets, const ITensor *dx, const ITensor *dy,
                              InterpolationPolicy policy, BorderMode border_mode, PixelValue constant_border_value, float sampling_offset,
                              bool align_corners, const Window &window)
{
    if(policy == InterpolationPolicy::BILINEAR)
    {
        qasymm8_signed_sve_scale_bilinear(src, dst, offsets, dx, dy, border_mode, constant_border_value, sampling_offset, align_corners, window);
    }
    else if(policy == InterpolationPolicy::NEAREST_NEIGHBOR)
    {
        qasymm8_signed_sve_scale_nearest(src, dst, offsets, sampling_offset, align_corners, window);
    }
}
} // namespace cpu
} // namespace arm_compute
