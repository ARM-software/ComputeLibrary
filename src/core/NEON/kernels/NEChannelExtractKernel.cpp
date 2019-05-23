/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEChannelExtractKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEChannelExtractKernel::NEChannelExtractKernel()
    : _func(nullptr), _lut_index(0)
{
}

void NEChannelExtractKernel::configure(const ITensor *input, Channel channel, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON(input == output);

    set_format_if_unknown(*output->info(), Format::U8);

    // Check if input tensor has a valid format
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::RGB888, Format::RGBA8888, Format::UYVY422, Format::YUYV422);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::U8);

    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);

    // Check if channel is valid for given format
    const Format format = input->info()->format();
    ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(format, channel);

    unsigned int subsampling = 1;

    if(format == Format::YUYV422 || format == Format::UYVY422)
    {
        // Check if the width of the tensor shape is even for formats with subsampled channels (UYVY422 and YUYV422)
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(format, input);

        if(channel != Channel::Y)
        {
            subsampling = 2;
        }
    }

    TensorShape output_shape = calculate_subsampled_shape(input->info()->tensor_shape(), format, channel);
    set_shape_if_empty(*output->info(), output_shape);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output_shape, output->info()->tensor_shape());

    _input     = input;
    _output    = output;
    _lut_index = channel_idx_from_format(format, channel);

    unsigned int num_elems_processed_per_iteration = 16;

    if(format == Format::YUYV422 || format == Format::UYVY422)
    {
        _func = &NEChannelExtractKernel::extract_1C_from_2C_img;

        if(channel != Channel::Y) // Channel::U or Channel::V
        {
            num_elems_processed_per_iteration = 32;
            _func                             = &NEChannelExtractKernel::extract_YUYV_uv;
        }
    }
    else // Format::RGB888 or Format::RGBA8888
    {
        _func = &NEChannelExtractKernel::extract_1C_from_3C_img;

        if(format == Format::RGBA8888)
        {
            _func = &NEChannelExtractKernel::extract_1C_from_4C_img;
        }
    }

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  output_access(output->info(), 0, 0, num_elems_processed_per_iteration, 1, 1.f / subsampling, 1.f / subsampling);
    update_window_and_padding(win, input_access, output_access);

    ValidRegion input_valid_region = input->info()->valid_region();
    output_access.set_valid_region(win, ValidRegion(input_valid_region.anchor, output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEChannelExtractKernel::configure(const IMultiImage *input, Channel channel, IImage *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);

    set_format_if_unknown(*output->info(), Format::U8);

    const Format format = input->info()->format();
    ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(format, channel);

    // Get input plane
    const IImage *input_plane = input->plane(plane_idx_from_channel(format, channel));
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_plane);

    if(Channel::Y == channel && format != Format::YUV444)
    {
        // Check if the width of the tensor shape is even for formats with subsampled channels (UYVY422 and YUYV422)
        ARM_COMPUTE_ERROR_ON_TENSORS_NOT_EVEN(format, input_plane);
    }

    // Calculate 2x2 subsampled tensor shape
    TensorShape output_shape = calculate_subsampled_shape(input->plane(0)->info()->tensor_shape(), format, channel);
    set_shape_if_empty(*output->info(), output_shape);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output_shape, output->info()->tensor_shape());

    // Check if input tensor has a valid format
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::NV12, Format::NV21, Format::IYUV, Format::YUV444);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::U8);

    _input     = input_plane;
    _output    = output;
    _lut_index = channel_idx_from_format(format, channel);

    unsigned int num_elems_processed_per_iteration = 32;

    _func = &NEChannelExtractKernel::copy_plane;

    if((format == Format::NV12 || format == Format::NV21) && channel != Channel::Y)
    {
        num_elems_processed_per_iteration = 16;
        _func                             = &NEChannelExtractKernel::extract_1C_from_2C_img;
    }

    Window win = calculate_max_window(*_input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(_input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, _input->info()->valid_region());

    INEKernel::configure(win);
}

void NEChannelExtractKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEChannelExtractKernel::extract_1C_from_2C_img(const Window &win)
{
    Iterator in(_input, win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = static_cast<uint8_t *>(in.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());
        const auto pixels  = vld2q_u8(in_ptr);
        vst1q_u8(out_ptr, pixels.val[_lut_index]);
    },
    in, out);
}

void NEChannelExtractKernel::extract_1C_from_3C_img(const Window &win)
{
    Iterator in(_input, win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = static_cast<uint8_t *>(in.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());
        const auto pixels  = vld3q_u8(in_ptr);
        vst1q_u8(out_ptr, pixels.val[_lut_index]);
    },
    in, out);
}

void NEChannelExtractKernel::extract_1C_from_4C_img(const Window &win)
{
    Iterator in(_input, win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = static_cast<uint8_t *>(in.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());
        const auto pixels  = vld4q_u8(in_ptr);
        vst1q_u8(out_ptr, pixels.val[_lut_index]);
    },
    in, out);
}

void NEChannelExtractKernel::extract_YUYV_uv(const Window &win)
{
    ARM_COMPUTE_ERROR_ON(win.x().step() % 2);

    Window win_out(win);
    win_out.set_dimension_step(Window::DimX, win.x().step() / 2);

    Iterator in(_input, win);
    Iterator out(_output, win_out);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = static_cast<uint8_t *>(in.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());
        const auto pixels  = vld4q_u8(in_ptr);
        vst1q_u8(out_ptr, pixels.val[_lut_index]);
    },
    in, out);
}

void NEChannelExtractKernel::copy_plane(const Window &win)
{
    Iterator in(_input, win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates &)
    {
        const auto in_ptr  = static_cast<uint8_t *>(in.ptr());
        const auto out_ptr = static_cast<uint8_t *>(out.ptr());
        vst4_u8(out_ptr, vld4_u8(in_ptr));
    },
    in, out);
}
