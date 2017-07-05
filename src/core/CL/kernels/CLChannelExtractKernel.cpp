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
#include "arm_compute/core/CL/kernels/CLChannelExtractKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLMultiImage.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/MultiImageInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

CLChannelExtractKernel::CLChannelExtractKernel()
    : _input(nullptr), _output(nullptr), _num_elems_processed_per_iteration(8), _subsampling(1)
{
}

void CLChannelExtractKernel::configure(const ICLTensor *input, Channel channel, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::RGB888, Format::RGBA8888, Format::YUYV422, Format::UYVY422);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::U8);
    ARM_COMPUTE_ERROR_ON(static_cast<const void *>(input) == static_cast<void *>(output));

    _input  = input;
    _output = output;

    // Check format
    const Format format = input->info()->format();
    ARM_COMPUTE_ERROR_ON_CHANNEL_NOT_IN_KNOWN_FORMAT(format, channel);

    // Create kernel
    std::string           kernel_name = "channel_extract_" + string_from_format(format);
    std::set<std::string> build_opts  = { ("-DCHANNEL_" + string_from_channel(channel)) };
    _kernel                           = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Half the processed elements for U,V channels due to sub-sampling of 2
    _subsampling = ((Format::YUYV422 == format || Format::UYVY422 == format) && Channel::Y != channel) ? 2 : 1;

    // Configure window
    Window                 win = calculate_max_window(*input->info(), Steps(_num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input->info(), 0, _num_elems_processed_per_iteration);
    AccessWindowRectangle  output_access(output->info(), 0, 0, _num_elems_processed_per_iteration, 1, 1.f / _subsampling, 1.f / _subsampling);

    update_window_and_padding(win, input_access, output_access);

    ValidRegion input_valid_region = input->info()->valid_region();
    output_access.set_valid_region(win, ValidRegion(input_valid_region.anchor, output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLChannelExtractKernel::configure(const ICLMultiImage *input, Channel channel, ICLImage *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::NV12, Format::NV21, Format::IYUV, Format::YUV444);
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output, Format::U8);
    ARM_COMPUTE_ERROR_ON(static_cast<const void *>(input) == static_cast<void *>(output));

    // Get format
    const Format fmt = input->info()->format();

    // Get input plane
    const ICLImage *input_plane = input->cl_plane(plane_idx_from_channel(fmt, channel));
    ARM_COMPUTE_ERROR_ON(nullptr == input_plane);

    _output      = output;
    _input       = input_plane;
    _subsampling = 1;

    // Create kernel
    std::string           kernel_name;
    std::set<std::string> build_opts;
    if(Channel::Y == channel || Format::IYUV == fmt || Format::YUV444 == fmt)
    {
        kernel_name = "copy_plane";
    }
    else
    {
        kernel_name = "channel_extract_" + string_from_format(fmt);
        build_opts.insert(("-DCHANNEL_" + string_from_channel(channel)));
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts));

    // Configure window
    Window                 win = calculate_max_window(*input_plane->info(), Steps(_num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input_plane->info(), 0, _num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output->info(), 0, _num_elems_processed_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input_plane->info()->valid_region());

    ICLKernel::configure(win);
}

void CLChannelExtractKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        Window win_sub(slice);
        win_sub.set(Window::DimX, Window::Dimension(win_sub.x().start() / _subsampling, win_sub.x().end() / _subsampling, win_sub.x().step() / _subsampling));
        win_sub.set(Window::DimY, Window::Dimension(win_sub.y().start() / _subsampling, win_sub.y().end() / _subsampling, 1));

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, win_sub);
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
