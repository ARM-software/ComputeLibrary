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
#include "arm_compute/core/CL/kernels/CLDerivativeKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <set>
#include <string>

using namespace arm_compute;

CLDerivativeKernel::CLDerivativeKernel()
    : _input(nullptr), _output_x(nullptr), _output_y(nullptr), _run_derivative_x(false), _run_derivative_y(false)
{
}

BorderSize CLDerivativeKernel::border_size() const
{
    return BorderSize(1);
}

void CLDerivativeKernel::configure(const ICLTensor *input, ICLTensor *output_x, ICLTensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_derivative_x = output_x != nullptr;
    _run_derivative_y = output_y != nullptr;

    if(_run_derivative_x)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_x, 1, DataType::S16);
    }

    if(_run_derivative_y)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_y, 1, DataType::S16);
    }

    _input    = input;
    _output_x = output_x;
    _output_y = output_y;

    // Set build options
    std::set<std::string> build_opts;

    if(_run_derivative_x)
    {
        build_opts.insert("-DGRAD_X");
    }

    if(_run_derivative_y)
    {
        build_opts.insert("-DGRAD_Y");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("derivative", build_opts));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_read_rows_per_iteration       = 3;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle  input_access(input->info(), 0, 0, 0, 0);
    AccessWindowHorizontal output_x_access(output_x == nullptr ? nullptr : output_x->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_y_access(output_y == nullptr ? nullptr : output_y->info(), 0, num_elems_processed_per_iteration);
    if(_run_derivative_x && _run_derivative_y)
    {
        input_access = AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_processed_per_iteration + 2, num_read_rows_per_iteration);
    }
    else if(_run_derivative_x)
    {
        input_access = AccessWindowHorizontal(input->info(), -border_size().left, num_elems_processed_per_iteration + 2);
    }
    else if(_run_derivative_y)
    {
        input_access = AccessWindowRectangle(input->info(), 0, -border_size().top, num_elems_processed_per_iteration, num_read_rows_per_iteration);
    }

    update_window_and_padding(win,
                              input_access,
                              output_x_access,
                              output_y_access);

    output_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
    output_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure(win);
}

void CLDerivativeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);

        if(_run_derivative_x)
        {
            add_2D_tensor_argument(idx, _output_x, slice);
        }

        if(_run_derivative_y)
        {
            add_2D_tensor_argument(idx, _output_y, slice);
        }

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
