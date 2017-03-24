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
#include "arm_compute/core/CL/kernels/CLColorConvertKernel.h"

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLMultiImage.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <sstream>

using namespace arm_compute;

CLColorConvertKernel::CLColorConvertKernel()
    : _input(nullptr), _output(nullptr), _multi_input(nullptr), _multi_output(nullptr)
{
}

void CLColorConvertKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON(input == nullptr);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::RGBA8888:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                    num_elems_processed_per_iteration = 16;
                    break;
                default:
                    break;
            }
            break;
        }
        case Format::UYVY422:
        case Format::YUYV422:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                case Format::RGBA8888:
                    num_elems_processed_per_iteration = 8;
                    break;
                default:
                    break;
            }
            break;
        }
        case Format::RGB888:
        {
            switch(output->info()->format())
            {
                case Format::RGBA8888:
                    num_elems_processed_per_iteration = 16;
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(num_elems_processed_per_iteration == 0, "Conversion from %s to %s not supported",
                             string_from_format(input->info()->format()).c_str(),
                             string_from_format(output->info()->format()).c_str());

    std::stringstream kernel_name;

    kernel_name << string_from_format(input->info()->format());
    kernel_name << "_to_";
    kernel_name << string_from_format(output->info()->format());
    kernel_name << "_bt709";

    _input  = input;
    _output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str()));

    // Configure kernel window
    Window                  win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLColorConvertKernel::configure(const ICLMultiImage *input, ICLImage *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::NV12:
        case Format::NV21:
        case Format::IYUV:
        {
            switch(output->info()->format())
            {
                case Format::RGB888:
                case Format::RGBA8888:
                    num_elems_processed_per_iteration = 4;
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(num_elems_processed_per_iteration == 0, "Conversion from %s to %s not supported",
                             string_from_format(input->info()->format()).c_str(),
                             string_from_format(output->info()->format()).c_str());

    std::stringstream kernel_name;

    kernel_name << string_from_format(input->info()->format());
    kernel_name << "_to_";
    kernel_name << string_from_format(output->info()->format());
    kernel_name << "_bt709";

    _multi_input = input;
    _output      = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str()));

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));
    win.set_dimension_step(Window::DimY, 2);

    AccessWindowAutoPadding output_access(output->info());

    unsigned int input_plane_count = 3;

    if(input->info()->format() == Format::NV12 || input->info()->format() == Format::NV21)
    {
        input_plane_count = 2;
    }

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->cl_plane(0)->info()),
                              AccessWindowAutoPadding(input->cl_plane(1)->info()),
                              AccessWindowAutoPadding(input_plane_count == 2 ? nullptr : input->cl_plane(2)->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLColorConvertKernel::configure(const ICLImage *input, ICLMultiImage *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(output == nullptr);

    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::RGB888:
        case Format::RGBA8888:
        {
            switch(output->info()->format())
            {
                case Format::NV12:
                case Format::IYUV:
                    num_elems_processed_per_iteration = 2;
                    break;
                case Format::YUV444:
                    num_elems_processed_per_iteration = 4;
                    break;
                default:
                    break;
            }
            break;
        }
        case Format::UYVY422:
        case Format::YUYV422:
        {
            switch(output->info()->format())
            {
                case Format::IYUV:
                case Format::NV12:
                    num_elems_processed_per_iteration = 8;
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(num_elems_processed_per_iteration == 0, "Conversion from %s to %s not supported",
                             string_from_format(input->info()->format()).c_str(),
                             string_from_format(output->info()->format()).c_str());

    std::stringstream kernel_name;

    kernel_name << string_from_format(input->info()->format());
    kernel_name << "_to_";
    kernel_name << string_from_format(output->info()->format());
    kernel_name << "_bt709";

    _input        = input;
    _multi_output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str()));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    if((input->info()->format() != Format::RGB888 || output->info()->format() != Format::YUV444) && (input->info()->format() != Format::RGBA8888 || output->info()->format() != Format::YUV444))
    {
        win.set_dimension_step(Window::DimY, 2);
    }

    unsigned int output_plane_count = 3;

    if(output->info()->format() == Format::NV12 || output->info()->format() == Format::NV21)
    {
        output_plane_count = 2;
    }

    AccessWindowAutoPadding output0_access(output->cl_plane(0)->info());
    AccessWindowAutoPadding output1_access(output->cl_plane(1)->info());
    AccessWindowAutoPadding output2_access(output_plane_count == 2 ? nullptr : output->cl_plane(2)->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output0_access,
                              output1_access,
                              output2_access);

    output0_access.set_valid_region();
    output1_access.set_valid_region();
    output2_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLColorConvertKernel::configure(const ICLMultiImage *input, ICLMultiImage *output)
{
    unsigned int num_elems_processed_per_iteration = 0;

    switch(input->info()->format())
    {
        case Format::NV12:
        case Format::NV21:
        {
            switch(output->info()->format())
            {
                case Format::IYUV:
                case Format::YUV444:
                    num_elems_processed_per_iteration = 16;
                    break;
                default:
                    break;
            }
            break;
        }
        case Format::IYUV:
        {
            switch(output->info()->format())
            {
                case Format::YUV444:
                case Format::NV12:
                    num_elems_processed_per_iteration = 16;
                    break;
                default:
                    break;
            }
            break;
        }
        default:
            break;
    }

    ARM_COMPUTE_ERROR_ON_MSG(num_elems_processed_per_iteration == 0, "Conversion from %s to %s not supported",
                             string_from_format(input->info()->format()).c_str(),
                             string_from_format(output->info()->format()).c_str());

    std::stringstream kernel_name;

    kernel_name << string_from_format(input->info()->format());
    kernel_name << "_to_";
    kernel_name << string_from_format(output->info()->format());
    kernel_name << "_bt709";

    _multi_input  = input;
    _multi_output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str()));

    // Configure kernel window
    Window win = calculate_max_window(*input->cl_plane(0)->info(), Steps(num_elems_processed_per_iteration));
    win.set_dimension_step(Window::DimY, 2);

    unsigned int input_plane_count = 3;

    if(input->info()->format() == Format::NV12 || input->info()->format() == Format::NV21)
    {
        input_plane_count = 2;
    }

    unsigned int output_plane_count = 3;

    if(output->info()->format() == Format::NV12 || output->info()->format() == Format::NV21)
    {
        output_plane_count = 2;
    }

    AccessWindowAutoPadding output0_access(output->cl_plane(0)->info());
    AccessWindowAutoPadding output1_access(output->cl_plane(1)->info());
    AccessWindowAutoPadding output2_access(output_plane_count == 2 ? nullptr : output->cl_plane(2)->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->cl_plane(0)->info()),
                              AccessWindowAutoPadding(input->cl_plane(1)->info()),
                              AccessWindowAutoPadding(input_plane_count == 2 ? nullptr : input->cl_plane(2)->info()),
                              output0_access,
                              output1_access,
                              output2_access);

    output0_access.set_valid_region();
    output1_access.set_valid_region();
    output2_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLColorConvertKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    if(nullptr != _input && nullptr != _output)
    {
        do
        {
            unsigned int idx = 0;
            add_2D_tensor_argument(idx, _input, slice);
            add_2D_tensor_argument(idx, _output, slice);
            enqueue(queue, *this, slice);
        }
        while(window.slide_window_slice_2D(slice));
    }
    else if(nullptr != _input && nullptr != _multi_output)
    {
        Format format = _multi_output->info()->format();
        do
        {
            Window win_uv(slice);

            if((Format::NV12 == format) || (Format::NV21 == format) || (Format::IYUV == format))
            {
                win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
                win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
            }
            unsigned int idx = 0;
            add_2D_tensor_argument(idx, _input, slice);
            add_2D_tensor_argument(idx, _multi_output->cl_plane(0), slice);
            for(int i = 1; i < 3 && (0 != _multi_output->cl_plane(i)->info()->num_dimensions()); ++i)
            {
                add_2D_tensor_argument(idx, _multi_output->cl_plane(i), win_uv);
            }
            enqueue(queue, *this, slice);
        }
        while(window.slide_window_slice_2D(slice));
    }
    else if(nullptr != _multi_input && nullptr != _output)
    {
        Format format = _multi_input->info()->format();
        do
        {
            Window win_uv(slice);

            if((Format::NV12 == format) || (Format::NV21 == format) || (Format::IYUV == format))
            {
                win_uv.set(Window::DimX, Window::Dimension(win_uv.x().start() / 2, win_uv.x().end() / 2, win_uv.x().step() / 2));
                win_uv.set(Window::DimY, Window::Dimension(win_uv.y().start() / 2, win_uv.y().end() / 2, 1));
            }

            unsigned int idx = 0;
            add_2D_tensor_argument(idx, _multi_input->cl_plane(0), slice);

            for(int i = 1; i < 3 && (0 != _multi_input->cl_plane(i)->info()->num_dimensions()); ++i)
            {
                add_2D_tensor_argument(idx, _multi_input->cl_plane(i), win_uv);
            }
            add_2D_tensor_argument(idx, _output, slice);
            enqueue(queue, *this, slice);
        }
        while(window.slide_window_slice_2D(slice));
    }
    else if(nullptr != _multi_input && nullptr != _multi_output)
    {
        Format in_format  = _multi_input->info()->format();
        Format out_format = _multi_output->info()->format();
        do
        {
            Window win_in_uv(slice);
            if((Format::NV12 == in_format) || (Format::NV21 == in_format) || (Format::IYUV == in_format))
            {
                win_in_uv.set(Window::DimX, Window::Dimension(win_in_uv.x().start() / 2,
                                                              win_in_uv.x().end() / 2, win_in_uv.x().step() / 2));
                win_in_uv.set(Window::DimY, Window::Dimension(win_in_uv.y().start() / 2, win_in_uv.y().end() / 2, 1));
            }
            unsigned int idx = 0;
            add_2D_tensor_argument(idx, _multi_input->cl_plane(0), slice);
            for(int i = 1; i < 3 && (0 != _multi_input->cl_plane(i)->info()->num_dimensions()); ++i)
            {
                add_2D_tensor_argument(idx, _multi_input->cl_plane(i), win_in_uv);
            }

            Window win_out_uv(slice);
            if((Format::NV12 == out_format) || (Format::NV21 == out_format) || (Format::IYUV == out_format))
            {
                win_out_uv.set(Window::DimX, Window::Dimension(win_out_uv.x().start() / 2,
                                                               win_out_uv.x().end() / 2, win_out_uv.x().step() / 2));
                win_out_uv.set(Window::DimY, Window::Dimension(win_out_uv.y().start() / 2, win_out_uv.y().end() / 2, 1));
            }

            add_2D_tensor_argument(idx, _multi_output->cl_plane(0), slice);
            for(int i = 1; i < 3 && (0 != _multi_output->cl_plane(i)->info()->num_dimensions()); ++i)
            {
                add_2D_tensor_argument(idx, _multi_output->cl_plane(i), win_out_uv);
            }
            enqueue(queue, *this, slice);
        }
        while(window.slide_window_slice_2D(slice));
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
}
