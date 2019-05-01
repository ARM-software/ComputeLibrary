/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NERemapKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
inline int32x4_t offset_nearest_interpolation(const float *mapx_ptr, const float *mapy_ptr, const float32x4_t &width, const float32x4_t &height, const int32x4_t &stride)
{
    const float32x4_t lowerxy = vdupq_n_f32(-1.f);

    float32x4_t x = vld1q_f32(mapx_ptr);
    float32x4_t y = vld1q_f32(mapy_ptr);

    // Clamp x coordinates
    x = vmaxq_f32(lowerxy, vminq_f32(x, width));
    y = vmaxq_f32(lowerxy, vminq_f32(y, height));

    const int32x4_t x_s32 = vcvtq_s32_f32(x);
    const int32x4_t y_s32 = vcvtq_s32_f32(y);

    return vmlaq_s32(x_s32, y_s32, stride);
}

} // namespace

NERemapKernel::NERemapKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _map_x(nullptr), _map_y(nullptr)
{
}

BorderSize NERemapKernel::border_size() const
{
    return BorderSize(1);
}

void NERemapKernel::configure(const ITensor *input, const ITensor *map_x, const ITensor *map_y, ITensor *output, InterpolationPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_x, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(map_y, 1, DataType::F32);

    _input  = input;
    _output = output;
    _map_x  = map_x;
    _map_y  = map_y;

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

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    const int total_right  = ceil_to_multiple(input->info()->dimension(0), num_elems_processed_per_iteration);
    const int access_right = total_right + (((total_right - input->info()->dimension(0)) == 0) ? border_size().right : 0);

    AccessWindowStatic input_access(input->info(), -border_size().left, -border_size().top, access_right, input->info()->dimension(1) + border_size().bottom);

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal mapx_access(map_x->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal mapy_access(map_y->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, mapx_access, mapy_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NERemapKernel::remap_nearest(const Window &window)
{
    // Don't increment in X and Y direction for the input tensor
    // A pointer to the start of this plane is needed as base for the precomputed offsets
    Window win_in(window);
    win_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_in.set(Window::DimY, Window::Dimension(0, 0, 0));

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator mapx(_map_x, window);
    Iterator mapy(_map_y, window);

    const float32x4_t width     = vdupq_n_f32(static_cast<float>(_input->info()->dimension(0)));
    const float32x4_t height    = vdupq_n_f32(static_cast<float>(_input->info()->dimension(1)));
    const int32x4_t   in_stride = vdupq_n_s32(static_cast<int32_t>(_input->info()->strides_in_bytes()[1]));

    execute_window_loop(window, [&](const Coordinates &)
    {
        const auto     mapx_ptr = reinterpret_cast<const float *>(mapx.ptr());
        const auto     mapy_ptr = reinterpret_cast<const float *>(mapy.ptr());
        const uint8_t *in_ptr   = in.ptr();

        const int32x4_t offset0 = offset_nearest_interpolation(mapx_ptr + 0, mapy_ptr + 0, width, height, in_stride);
        const int32x4_t offset1 = offset_nearest_interpolation(mapx_ptr + 4, mapy_ptr + 4, width, height, in_stride);
        const int32x4_t offset2 = offset_nearest_interpolation(mapx_ptr + 8, mapy_ptr + 8, width, height, in_stride);
        const int32x4_t offset3 = offset_nearest_interpolation(mapx_ptr + 12, mapy_ptr + 12, width, height, in_stride);

        uint8x16_t tmp = vdupq_n_u8(0);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset0, 0)], tmp, 0);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset0, 1)], tmp, 1);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset0, 2)], tmp, 2);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset0, 3)], tmp, 3);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset1, 0)], tmp, 4);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset1, 1)], tmp, 5);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset1, 2)], tmp, 6);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset1, 3)], tmp, 7);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset2, 0)], tmp, 8);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset2, 1)], tmp, 9);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset2, 2)], tmp, 10);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset2, 3)], tmp, 11);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset3, 0)], tmp, 12);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset3, 1)], tmp, 13);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset3, 2)], tmp, 14);
        tmp            = vsetq_lane_u8(in_ptr[vgetq_lane_s32(offset3, 3)], tmp, 15);
        vst1q_u8(out.ptr(), tmp);
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

    Iterator in(_input, win_in);
    Iterator out(_output, window);
    Iterator mapx(_map_x, window);
    Iterator mapy(_map_y, window);

    const size_t width     = _input->info()->dimension(0);
    const size_t height    = _input->info()->dimension(1);
    const size_t in_stride = _input->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates &)
    {
        const auto     mapx_ptr = reinterpret_cast<float *>(mapx.ptr());
        const auto     mapy_ptr = reinterpret_cast<float *>(mapy.ptr());
        const uint8_t *in_ptr   = in.ptr();

        uint8x8_t tmp0 = vdup_n_u8(0);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[0], mapy_ptr[0]), tmp0, 0);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[1], mapy_ptr[1]), tmp0, 1);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[2], mapy_ptr[2]), tmp0, 2);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[3], mapy_ptr[3]), tmp0, 3);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[4], mapy_ptr[4]), tmp0, 4);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[5], mapy_ptr[5]), tmp0, 5);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[6], mapy_ptr[6]), tmp0, 6);
        tmp0           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[7], mapy_ptr[7]), tmp0, 7);

        uint8x8_t tmp1 = vdup_n_u8(0);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[8], mapy_ptr[8]), tmp1, 0);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[9], mapy_ptr[9]), tmp1, 1);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[10], mapy_ptr[10]), tmp1, 2);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[11], mapy_ptr[11]), tmp1, 3);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[12], mapy_ptr[12]), tmp1, 4);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[13], mapy_ptr[13]), tmp1, 5);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[14], mapy_ptr[14]), tmp1, 6);
        tmp1           = vset_lane_u8(pixel_bilinear_c1_clamp(in_ptr, in_stride, width, height, mapx_ptr[15], mapy_ptr[15]), tmp1, 7);

        vst1q_u8(out.ptr(), vcombine_u8(tmp0, tmp1));
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
