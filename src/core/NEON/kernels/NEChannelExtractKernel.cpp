/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
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
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);
    ARM_COMPUTE_ERROR_ON(input == output);
    ARM_COMPUTE_ERROR_ON(Format::U8 != output->info()->format());

    unsigned int num_elems_processed_per_iteration = 8;

    // Check format and channel
    const Format       format = input->info()->format();
    const unsigned int subsampling(((Format::YUYV422 == format || Format::UYVY422 == format) && Channel::Y != channel) ? 2 : 1);

    switch(format)
    {
        case Format::RGB888:
        case Format::RGBA8888:
            num_elems_processed_per_iteration = 16;
            _func                             = (Format::RGB888 == format) ? &NEChannelExtractKernel::extract_1C_from_3C_img : &NEChannelExtractKernel::extract_1C_from_4C_img;
            switch(channel)
            {
                case Channel::R:
                    _lut_index = 0;
                    break;
                case Channel::G:
                    _lut_index = 1;
                    break;
                case Channel::B:
                    _lut_index = 2;
                    break;
                case Channel::A:
                    if(Format::RGBA8888 == format)
                    {
                        _lut_index = 3;
                        _func      = &NEChannelExtractKernel::extract_1C_from_4C_img;
                        break;
                    }
                default:
                    ARM_COMPUTE_ERROR("Not supported channel for this format.");
                    break;
            }
            break;
        case Format::YUYV422:
        case Format::UYVY422:
            switch(channel)
            {
                case Channel::Y:
                    num_elems_processed_per_iteration = 16;
                    _func                             = &NEChannelExtractKernel::extract_1C_from_2C_img;
                    _lut_index                        = (Format::YUYV422 == format) ? 0 : 1;
                    break;
                case Channel::U:
                    num_elems_processed_per_iteration = 32;
                    _func                             = &NEChannelExtractKernel::extract_YUYV_uv;
                    _lut_index                        = (Format::YUYV422 == format) ? 1 : 0;
                    break;
                case Channel::V:
                    num_elems_processed_per_iteration = 32;
                    _func                             = &NEChannelExtractKernel::extract_YUYV_uv;
                    _lut_index                        = (Format::YUYV422 == format) ? 3 : 2;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel for this format.");
                    break;
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported format.");
            break;
    }
    _input  = input;
    _output = output;

    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowRectangle output_access(input->info(), 0, 0, num_elems_processed_per_iteration, 1, 1.f / subsampling, 1.f / subsampling);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                              output_access);

    ValidRegion input_valid_region = input->info()->valid_region();

    output_access.set_valid_region(win, ValidRegion(std::move(input_valid_region.anchor), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEChannelExtractKernel::configure(const IMultiImage *input, Channel channel, IImage *output)
{
    ARM_COMPUTE_ERROR_ON(nullptr == input);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON(static_cast<const void *>(input) == static_cast<void *>(output));
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::U8);

    unsigned int num_elems_processed_per_iteration = 32;

    const Format &format = input->info()->format();

    switch(format)
    {
        case Format::NV12:
        case Format::NV21:
            switch(channel)
            {
                case Channel::Y:
                    _input = input->plane(0);
                    _func  = &NEChannelExtractKernel::copy_plane;
                    break;
                case Channel::U:
                    _input                            = input->plane(1);
                    num_elems_processed_per_iteration = 16;
                    _func                             = &NEChannelExtractKernel::extract_1C_from_2C_img;
                    _lut_index                        = (Format::NV12 == format) ? 0 : 1;
                    break;
                case Channel::V:
                    _input                            = input->plane(1);
                    num_elems_processed_per_iteration = 16;
                    _func                             = &NEChannelExtractKernel::extract_1C_from_2C_img;
                    _lut_index                        = (Format::NV12 == format) ? 1 : 0;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel for this format.");
                    break;
            }
            break;
        case Format::IYUV:
        case Format::YUV444:
            _func = &NEChannelExtractKernel::copy_plane;
            switch(channel)
            {
                case Channel::Y:
                    _input = input->plane(0);
                    break;
                case Channel::U:
                    _input = input->plane(1);
                    break;
                case Channel::V:
                    _input = input->plane(2);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not supported channel for this format.");
                    break;
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported format.");
            break;
    }

    _output                    = output;
    Window                 win = calculate_max_window(*_input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal input_access(_input->info(), 0, num_elems_processed_per_iteration);
    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, _input->info()->valid_region());

    INEKernel::configure(win);
}

void NEChannelExtractKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}

void NEChannelExtractKernel::extract_1C_from_2C_img(const Window &win)
{
    Iterator in(_input, win);
    Iterator out(_output, win);

    execute_window_loop(win, [&](const Coordinates & id)
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

    execute_window_loop(win, [&](const Coordinates & id)
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

    execute_window_loop(win, [&](const Coordinates & id)
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

    execute_window_loop(win, [&](const Coordinates & id)
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
