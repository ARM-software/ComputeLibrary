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
#include "arm_compute/core/CL/kernels/CLHistogramKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLDistribution1D.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cstring>
#include <string>

using namespace arm_compute;

// each thread handle 16 pixels
constexpr signed int pixels_per_item = 16;

// local work group size in X dimension
constexpr unsigned int local_x_size = 16;

CLHistogramKernel::CLHistogramKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLHistogramKernel::configure(const ICLImage *input, ICLDistribution1D *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    // Check input size
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);

    // Check offset
    ARM_COMPUTE_ERROR_ON_MSG(0 > output->offset() || output->offset() > 256, "Offset is larger than the image value range.");

    // Check range
    ARM_COMPUTE_ERROR_ON_MSG(output->range() > 256 /* max range */, "Range larger than the image value range.");

    _input  = input;
    _output = output;

    if(_input->info()->dimension(0) < pixels_per_item)
    {
        return;
    }

    unsigned int num_bins    = _output->num_bins();
    unsigned int window_size = _output->window();
    unsigned int offset      = _output->offset();
    unsigned int range       = _output->range();
    unsigned int offrange    = offset + range;
    unsigned int bin_size    = _output->size();
    unsigned int buffer_size = bin_size + 1; // We need one extra place for pixels that don't meet the conditions

    // Create kernel
    bool        is_fixed_size = (256 == num_bins) && (1 == window_size) && (0 == offset) && (256 == offrange);
    std::string kernel_name   = is_fixed_size ? "hist_local_kernel_fixed" : "hist_local_kernel";
    _kernel                   = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, buffer_size, nullptr);
    _kernel.setArg(idx++, _output->cl_buffer());
    if(!is_fixed_size)
    {
        _kernel.setArg<cl_uint>(idx++, num_bins);
        _kernel.setArg<cl_uint>(idx++, offset);
        _kernel.setArg<cl_uint>(idx++, range);
        _kernel.setArg<cl_uint>(idx++, offrange);
    }

    // We only run histogram on Image, therefore only 2 dimensions here
    unsigned int end_position = (_input->info()->dimension(0) / pixels_per_item) * pixels_per_item;

    // Configure kernel window
    Window win;
    win.set(0, Window::Dimension(0, end_position, pixels_per_item));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));

    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, pixels_per_item));

    ICLKernel::configure(win);
}

void CLHistogramKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    _output->map(queue, true);
    ARM_COMPUTE_ERROR_ON(_output->buffer() == nullptr);
    memset(_output->buffer(), 0, _output->size());
    _output->unmap(queue);

    if(_input->info()->dimension(0) < pixels_per_item)
    {
        return;
    }

    Window             slice = window.first_slice_window_2D();
    const unsigned int gws_x = (window.x().end() - window.x().start()) / window.x().step();
    cl::NDRange        lws   = (local_x_size < gws_x) ? cl::NDRange(local_x_size, 1) : cl::NDRange(1, 1);

    do
    {
        /* Run the core part which has width can be divided by 16 */
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);

        enqueue(queue, *this, slice, lws);
    }
    while(window.slide_window_slice_2D(slice));
}

CLHistogramBorderKernel::CLHistogramBorderKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLHistogramBorderKernel::configure(const ICLImage *input, ICLDistribution1D *output)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON(nullptr == output);

    // Check input size
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);

    // Check offset
    ARM_COMPUTE_ERROR_ON_MSG(0 > output->offset() || output->offset() > 256, "Offset is larger than the image value range.");

    // Check range
    ARM_COMPUTE_ERROR_ON_MSG(output->range() > 256 /* max range */, "Range larger than the image value range.");

    // We only run histogram on Image, therefore only 2 dimensions here
    unsigned int start_position = (input->info()->dimension(0) / pixels_per_item) * pixels_per_item;

    if(start_position >= input->info()->dimension(0))
    {
        return; // no need to run histogram border kernel
    }

    _input  = input;
    _output = output;

    unsigned int num_bins    = _output->num_bins();
    unsigned int window_size = _output->window();
    unsigned int offset      = _output->offset();
    unsigned int range       = _output->range();
    unsigned int offrange    = offset + range;

    // Create kernel
    bool        is_fixed_size = (256 == num_bins) && (1 == window_size) && (0 == offset) && (256 == offrange);
    std::string kernel_name   = is_fixed_size ? "hist_border_kernel_fixed" : "hist_border_kernel";
    _kernel                   = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, _output->cl_buffer());
    if(!is_fixed_size)
    {
        _kernel.setArg<cl_uint>(idx++, num_bins);
        _kernel.setArg<cl_uint>(idx++, offset);
        _kernel.setArg<cl_uint>(idx++, range);
        _kernel.setArg<cl_uint>(idx++, offrange);
    }

    // Configure kernel window
    Window win;
    win.set(0, Window::Dimension(start_position, _input->info()->dimension(0)));
    win.set(1, Window::Dimension(0, _input->info()->dimension(1)));
    update_window_and_padding(win, AccessWindowHorizontal(input->info(), 0, 1));
    ICLKernel::configure(win);
}

void CLHistogramBorderKernel::run(const Window &window, cl::CommandQueue &queue)
{
    if(window.x().start() >= window.x().end())
    {
        return;
    }

    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);

    cl::NDRange lws = cl::NDRange(1, 1);

    Window slice = window.first_slice_window_2D();

    do
    {
        /* Run the border part which has width cannot be divided by 16 */
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);

        enqueue(queue, *this, slice, lws);
    }
    while(window.slide_window_slice_2D(slice));
}
