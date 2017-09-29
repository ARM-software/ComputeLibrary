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
#include "arm_compute/core/CL/kernels/CLMeanStdDevKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <cmath>
#include <set>
#include <string>

using namespace arm_compute;

CLMeanStdDevKernel::CLMeanStdDevKernel()
    : _input(nullptr), _mean(nullptr), _stddev(nullptr), _global_sum(nullptr), _global_sum_squared(nullptr), _border_size(0)
{
}

BorderSize CLMeanStdDevKernel::border_size() const
{
    return _border_size;
}

void CLMeanStdDevKernel::configure(const ICLImage *input, float *mean, cl::Buffer *global_sum, float *stddev, cl::Buffer *global_sum_squared)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(nullptr == mean);
    ARM_COMPUTE_ERROR_ON(nullptr == global_sum);
    ARM_COMPUTE_ERROR_ON(stddev && nullptr == global_sum_squared);

    _input              = input;
    _mean               = mean;
    _stddev             = stddev;
    _global_sum         = global_sum;
    _global_sum_squared = global_sum_squared;

    // Create kernel
    std::set<std::string> build_opts;

    if(_stddev != nullptr)
    {
        build_opts.insert("-DSTDDEV");
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("mean_stddev_accumulate", build_opts));

    // Set fixed arguments
    unsigned int idx = num_arguments_per_2D_tensor(); //Skip the input parameters

    _kernel.setArg(idx++, static_cast<cl_uint>(input->info()->dimension(1)));
    _kernel.setArg(idx++, *_global_sum);

    if(_stddev != nullptr)
    {
        _kernel.setArg(idx++, *_global_sum_squared);
    }

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration_x = 8;
    const unsigned int     num_elems_processed_per_iteration_y = input->info()->dimension(1);

    _border_size = BorderSize(ceil_to_multiple(input->info()->dimension(0), num_elems_processed_per_iteration_x) - input->info()->dimension(0));

    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    AccessWindowRectangle input_access(input->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    update_window_and_padding(win, input_access);

    ICLKernel::configure(win);
}

void CLMeanStdDevKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Clear sums
    static const cl_ulong zero = 0;
    queue.enqueueWriteBuffer(*_global_sum, CL_FALSE, 0, sizeof(cl_ulong), &zero);

    if(_stddev != nullptr)
    {
        queue.enqueueWriteBuffer(*_global_sum_squared, CL_FALSE, 0, sizeof(cl_ulong), &zero);
    }

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        // Set slice step equal to height to force gws[1] to 1,
        // as each thread calculates the sum across all rows and columns equal to the number of elements processed by each work-item
        slice.set_dimension_step(Window::DimY, _input->info()->dimension(1));
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));

    // Calculate mean and stddev
    cl_ulong    global_sum         = 0;
    cl_ulong    global_sum_squared = 0;
    const float num_pixels         = _input->info()->dimension(0) * _input->info()->dimension(1);

    queue.enqueueReadBuffer(*_global_sum, CL_TRUE, 0, sizeof(cl_ulong), static_cast<void *>(&global_sum));
    const float mean = global_sum / num_pixels;
    *_mean           = mean;

    if(_stddev != nullptr)
    {
        queue.enqueueReadBuffer(*_global_sum_squared, CL_TRUE, 0, sizeof(cl_ulong), static_cast<void *>(&global_sum_squared));
        *_stddev = std::sqrt((global_sum_squared / num_pixels) - (mean * mean));
    }
}
