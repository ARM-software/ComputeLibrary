/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGaussianPyramidKernel.h"

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLGaussianPyramidHorKernel::CLGaussianPyramidHorKernel()
    : _input(nullptr), _output(nullptr)
{
}

BorderSize CLGaussianPyramidHorKernel::border_size() const
{
    return BorderSize(2);
}

void CLGaussianPyramidHorKernel::configure(const ICLTensor *input, ICLTensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != 2 * output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gaussian1x5_sub_x"));

    const unsigned int processed_elements = 8;

    // Configure kernel window
    Window                  win = calculate_max_window_horizontal(*input->info(), Steps(processed_elements), border_undefined, border_size());
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLGaussianPyramidHorKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    //The output is half the width of the input:
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(window.x().start() / 2, window.x().end() / 2, window.x().step() / 2));

    const int even_width = 1 - (_input->info()->dimension(0) % 2);
    Window    win_in(window);
    win_in.shift(Window::DimX, -2 + even_width);

    Window slice_in  = win_in.first_slice_window_2D();
    Window slice_out = win_out.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(win_in.slide_window_slice_2D(slice_in) && win_out.slide_window_slice_2D(slice_out));
}

CLGaussianPyramidVertKernel::CLGaussianPyramidVertKernel()
    : _input(nullptr), _output(nullptr)
{
}

BorderSize CLGaussianPyramidVertKernel::border_size() const
{
    return BorderSize(2, 0);
}

void CLGaussianPyramidVertKernel::configure(const ICLTensor *input, ICLTensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != 2 * output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gaussian5x1_sub_y"));

    const unsigned int processed_elements = 8;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(processed_elements), border_undefined, border_size());
    // In case of even height we shift the window to correctly reject even rows
    const int even_height = 1 - (_input->info()->dimension(1) % 2);
    win.set(Window::DimY, Window::Dimension(win.y().start() + even_height, win.y().end() + even_height, 2));
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
}

void CLGaussianPyramidVertKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() != 8);
    ARM_COMPUTE_ERROR_ON(window.y().step() % 2);

    Window win_in(window);
    win_in.set_dimension_step(Window::DimX, 8);

    Window win_out(window);
    win_out.set(Window::DimY, Window::Dimension(window.y().start() / 2, window.y().end() / 2, 1));

    Window slice_in  = win_in.first_slice_window_2D();
    Window slice_out = win_out.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(win_in.slide_window_slice_2D(slice_in) && win_out.slide_window_slice_2D(slice_out));
}
