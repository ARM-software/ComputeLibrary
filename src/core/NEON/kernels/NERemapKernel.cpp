/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NERemapKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/ScaleHelpers.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute::scale_helpers;

namespace arm_compute
{
class Coordinates;

namespace
{
inline int32_t num_out_of_tensor(const float *mapx_ptr, const float *mapy_ptr, const int32x4_t &width_1, const int32x4_t &height_1)
{
    const int32x4_t mapx_s32 = vcvtq_s32_f32(vld1q_f32(mapx_ptr));
    const int32x4_t mapy_s32 = vcvtq_s32_f32(vld1q_f32(mapy_ptr));

    const int32x4_t outbx_s32 = vminq_s32(vmaxq_s32(vminq_s32(vsubq_s32(width_1, mapx_s32), mapx_s32), vdupq_n_s32(-1)), vdupq_n_s32(0));  // Contains -1 if out of border in x, 0 otherwise
    const int32x4_t outby_s32 = vminq_s32(vmaxq_s32(vminq_s32(vsubq_s32(height_1, mapy_s32), mapy_s32), vdupq_n_s32(-1)), vdupq_n_s32(0)); // Contains -1 if out of border in y, 0 otherwise

    const int32x4_t out_of_tensor_v = vminq_s32(outbx_s32, outby_s32);
#if defined(__aarch64__)
    // only AArch64 supports vaddv
    return vaddvq_s32(out_of_tensor_v);
#else  // __aarch64__    
    return vgetq_lane_s32(out_of_tensor_v, 0) + vgetq_lane_s32(out_of_tensor_v, 1) + vgetq_lane_s32(out_of_tensor_v, 2)  + vgetq_lane_s32(out_of_tensor_v, 3);
#endif // __aarch64__
}

inline void serial_remap_nearest_interpolation(const uint8_t *in_ptr, const float *mapx_ptr, const float *mapy_ptr, uint8_t *out_ptr,
                                               int32_t width_val, int32_t height_val, int32_t in_stride_val, uint8_t constant_border_value)
{
    const auto x_s32 = static_cast<int32_t>(*mapx_ptr);
    const auto y_s32 = static_cast<int32_t>(*mapy_ptr);
    if(x_s32 < 0 || y_s32 < 0 || x_s32 >= width_val || y_s32 >= height_val)
    {
        *(out_ptr) = constant_border_value;
    }
    else
    {
        *(out_ptr) = in_ptr[x_s32 + y_s32 * in_stride_val];
    }
}

inline int32x4_t offset_nearest_interpolation(const float *mapx_ptr, const float *mapy_ptr, const int32x4_t &stride)
{
    const int32x4_t mapx_s32 = vcvtq_s32_f32(vld1q_f32(mapx_ptr));
    const int32x4_t mapy_s32 = vcvtq_s32_f32(vld1q_f32(mapy_ptr));
    return vmlaq_s32(mapx_s32, mapy_s32, stride);
}

inline uint8_t pixel_bilinear_c1_clamp(const uint8_t *pixel_ptr, int32_t stride, int32_t width, int32_t height, float x, float y, uint8_t constant_border_value)
{
    x = std::max(-1.f, std::min(x, static_cast<float>(width)));
    y = std::max(-1.f, std::min(y, static_cast<float>(height)));

    const int32_t xi = static_cast<int32_t>(std::floor(x));
    const int32_t yi = static_cast<int32_t>(std::floor(y));

    const float dx = x - static_cast<float>(xi);
    const float dy = y - static_cast<float>(yi);

    // Calculating the address won't trigger a segfault in case the value is outside the tensor
    // The ternary operator resolves the values in both conditions
    const uint8_t *a00 = (xi < 0 || xi >= width || yi < 0 || yi >= height) ? &constant_border_value : (pixel_ptr + xi + yi * stride);
    const uint8_t *a01 = (xi + 1 >= width || yi < 0 || yi >= height) ? &constant_border_value : (pixel_ptr + xi + 1 + yi * stride);
    const uint8_t *a10 = (xi < 0 || xi >= width || yi + 1 >= height) ? &constant_border_value : (pixel_ptr + xi + yi * stride + stride);
    const uint8_t *a11 = (xi + 1 >= width || yi + 1 >= height) ? &constant_border_value : (pixel_ptr + xi + 1 + yi * stride + stride);

    const float dx1 = 1.0f - dx;
    const float dy1 = 1.0f - dy;
    const float w1  = dx1 * dy1;
    const float w2  = dx * dy1;
    const float w3  = dx1 * dy;
    const float w4  = dx * dy;

    return static_cast<uint8_t>((*a00) * w1 + (*a01) * w2 + (*a10) * w3 + (*a11) * w4);
}
} // namespace

NERemapKernel::NERemapKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _map_x(nullptr), _map_y(nullptr), _border_mode(BorderMode::UNDEFINED), _constant_border_value(0)
{
}

void NERemapKernel::configure(const ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_x, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_y, 1, DataType::F32);

    _input                 = input;
    _output                = output;
    _map_x                 = map_x;
    _map_y                 = map_y;
    _border_mode           = border_mode;
    _constant_border_value = constant_border_value;

    switch(policy)
    {
        case InterpolationPolicy::NEAREST_NEIGHBOR:
        {
            _func = &NERemapKernel::remap_nearest;
            break;
        }
        case InterpolationPolicy::BILINEAR:
        {
            _func = &NERemapKernel::remap_bilinear;
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Unsupported interpolation mode");
            break;
    }

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps());
    INEKernel::configure(win);
}

void NERemapKernel::remap_nearest(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    const auto    window_start_x = static_cast<int32_t>(window.x().start());
    const auto    window_end_x   = static_cast<int32_t>(window.x().end());
    const int32_t window_step_x  = 8;

    // Don't increment in X direction for the output, mapx, mapy tensors
    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(_input, win_in);
    Iterator out(_output, win);
    Iterator mapx(_map_x, win);
    Iterator mapy(_map_y, win);

    const int32_t   width_val     = static_cast<int32_t>(_input->info()->dimension(0));
    const int32_t   height_val    = static_cast<int32_t>(_input->info()->dimension(1));
    const int32_t   in_stride_val = static_cast<int32_t>(_input->info()->strides_in_bytes()[1]);
    const int32x4_t width_1       = vdupq_n_s32(width_val - 1);
    const int32x4_t height_1      = vdupq_n_s32(height_val - 1);
    const int32x4_t in_stride     = vdupq_n_s32(in_stride_val);

    execute_window_loop(win, [&](const Coordinates &)
    {
        auto           mapx_ptr = reinterpret_cast<const float *>(mapx.ptr());
        auto           mapy_ptr = reinterpret_cast<const float *>(mapy.ptr());
        const uint8_t *in_ptr   = in.ptr();
        uint8_t       *out_ptr  = out.ptr();
        int32_t        x        = window_start_x;
        for(; x < window_end_x - window_step_x; x += window_step_x, mapx_ptr += window_step_x, mapy_ptr += window_step_x, out_ptr += window_step_x)
        {
            const int32_t out_of_tensor0 = num_out_of_tensor(mapx_ptr, mapy_ptr + 0, width_1, height_1);
            const int32_t out_of_tensor1 = num_out_of_tensor(mapx_ptr + 4, mapy_ptr + 4, width_1, height_1);
            const int32_t out_of_tensor  = out_of_tensor0 + out_of_tensor1;

            if(out_of_tensor == -8)
            {
                // All elements are out of xy plane
                uint8x8_t tmp = vdup_n_u8(_constant_border_value);
                vst1_u8(out_ptr, tmp);
            }
            else if(out_of_tensor < 0)
            {
                // Some elements are out of xy plane
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr, mapy_ptr, out_ptr, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 1, mapy_ptr + 1, out_ptr + 1, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 2, mapy_ptr + 2, out_ptr + 2, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 3, mapy_ptr + 3, out_ptr + 3, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 4, mapy_ptr + 4, out_ptr + 4, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 5, mapy_ptr + 5, out_ptr + 5, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 6, mapy_ptr + 6, out_ptr + 6, width_val, height_val, in_stride_val, _constant_border_value);
                serial_remap_nearest_interpolation(in_ptr, mapx_ptr + 7, mapy_ptr + 7, out_ptr + 7, width_val, height_val, in_stride_val, _constant_border_value);
            }
            else
            {
                // All elements are in xy plane
                uint8x8_t       tmp     = vdup_n_u8(0);
                const int32x4_t offset0 = offset_nearest_interpolation(mapx_ptr, mapy_ptr, in_stride);
                const int32x4_t offset1 = offset_nearest_interpolation(mapx_ptr + 4, mapy_ptr + 4, in_stride);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset0, 0)], tmp, 0);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset0, 1)], tmp, 1);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset0, 2)], tmp, 2);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset0, 3)], tmp, 3);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset1, 0)], tmp, 4);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset1, 1)], tmp, 5);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset1, 2)], tmp, 6);
                tmp                     = vset_lane_u8(in_ptr[vgetq_lane_s32(offset1, 3)], tmp, 7);
                vst1_u8(out_ptr, tmp);
            }
        }
        for(; x < window_end_x; ++x, ++mapx_ptr, ++mapy_ptr, ++out_ptr)
        {
            serial_remap_nearest_interpolation(in_ptr, mapx_ptr, mapy_ptr, out_ptr, width_val, height_val, in_stride_val, _constant_border_value);
        }
    },
    in, out, mapx, mapy);
}

void NERemapKernel::remap_bilinear(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    const auto    window_start_x = static_cast<int32_t>(window.x().start());
    const auto    window_end_x   = static_cast<int32_t>(window.x().end());
    const int32_t window_step_x  = 8;

    // Don't increment in X direction for the output, mapx, mapy tensors
    Window win(window);
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator in(_input, win_in);
    Iterator out(_output, win);
    Iterator mapx(_map_x, win);
    Iterator mapy(_map_y, win);

    const int32_t   width_val     = static_cast<int32_t>(_input->info()->dimension(0));
    const int32_t   height_val    = static_cast<int32_t>(_input->info()->dimension(1));
    const int32x4_t width_2       = vdupq_n_s32(width_val - 2);
    const int32x4_t height_2      = vdupq_n_s32(height_val - 2);
    const int32_t   in_stride_val = static_cast<int32_t>(_input->info()->strides_in_bytes()[1]);

    execute_window_loop(win, [&](const Coordinates &)
    {
        auto           mapx_ptr = reinterpret_cast<const float *>(mapx.ptr());
        auto           mapy_ptr = reinterpret_cast<const float *>(mapy.ptr());
        const uint8_t *in_ptr   = in.ptr();
        uint8_t       *out_ptr  = out.ptr();
        int32_t        x        = window_start_x;
        for(; x < window_end_x - window_step_x; x += window_step_x, mapx_ptr += window_step_x, mapy_ptr += window_step_x, out_ptr += window_step_x)
        {
            const int32_t out_of_tensor0 = num_out_of_tensor(mapx_ptr, mapy_ptr + 0, width_2, height_2);
            const int32_t out_of_tensor1 = num_out_of_tensor(mapx_ptr + 4, mapy_ptr + 4, width_2, height_2);
            const int32_t out_of_tensor  = out_of_tensor0 + out_of_tensor1;

            if(out_of_tensor < 0)
            {
                // Elements are out of xy plane
                *(out_ptr)     = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[0], mapy_ptr[0], _constant_border_value);
                *(out_ptr + 1) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[1], mapy_ptr[1], _constant_border_value);
                *(out_ptr + 2) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[2], mapy_ptr[2], _constant_border_value);
                *(out_ptr + 3) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[3], mapy_ptr[3], _constant_border_value);
                *(out_ptr + 4) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[4], mapy_ptr[4], _constant_border_value);
                *(out_ptr + 5) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[5], mapy_ptr[5], _constant_border_value);
                *(out_ptr + 6) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[6], mapy_ptr[6], _constant_border_value);
                *(out_ptr + 7) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[7], mapy_ptr[7], _constant_border_value);
            }
            else
            {
                // All elements are in xy plane
                uint8x8_t tmp = vdup_n_u8(0);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[0], mapy_ptr[0], _constant_border_value), tmp, 0);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[1], mapy_ptr[1], _constant_border_value), tmp, 1);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[2], mapy_ptr[2], _constant_border_value), tmp, 2);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[3], mapy_ptr[3], _constant_border_value), tmp, 3);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[4], mapy_ptr[4], _constant_border_value), tmp, 4);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[5], mapy_ptr[5], _constant_border_value), tmp, 5);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[6], mapy_ptr[6], _constant_border_value), tmp, 6);
                tmp           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[7], mapy_ptr[7], _constant_border_value), tmp, 7);
                vst1_u8(out_ptr, tmp);
            }
        }
        for(; x < window_end_x; ++x, ++mapx_ptr, ++mapy_ptr, ++out_ptr)
        {
            *(out_ptr) = pixel_bilinear_c1_clamp(in_ptr, in_stride_val, width_val, height_val, mapx_ptr[0], mapy_ptr[0], _constant_border_value);
        }
    },
    in, out, mapx, mapy);
}

void NERemapKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
} // namespace arm_compute