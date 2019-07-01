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
#include "arm_compute/core/CL/kernels/CLGaussianPyramidKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLGaussianPyramidHorKernel::CLGaussianPyramidHorKernel()
    : _l2_load_offset(0)
{
}

BorderSize CLGaussianPyramidHorKernel::border_size() const
{
    return BorderSize{ 0, 2 };
}

void CLGaussianPyramidHorKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Create kernel
    const std::string kernel_name = std::string("gaussian1x5_sub_x");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_elems_read_per_iteration      = 20;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    const float            scale_x                           = static_cast<float>(output->info()->dimension(0)) / input->info()->dimension(0);

    Window                 win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration, scale_x);

    // Sub sampling selects odd pixels (1, 3, 5, ...) for images with even
    // width and even pixels (0, 2, 4, ...) for images with odd width. (Whether
    // a pixel is even or odd is determined based on the tensor shape not the
    // valid region!)
    // Thus the offset from which the first pixel (L2) for the convolution is
    // loaded depends on the anchor and shape of the valid region.
    // In the case of an even shape (= even image width) we need to load L2
    // from -2 if the anchor is odd and from -1 if the anchor is even. That
    // makes sure that L2 is always loaded from an odd pixel.
    // On the other hand, for an odd shape (= odd image width) we need to load
    // L2 from -1 if the anchor is odd and from -2 if the anchor is even to
    // achieve the opposite effect.
    // The condition can be simplified to checking whether anchor + shape is
    // odd (-2) or even (-1) as only adding an odd and an even number will have
    // an odd result.
    _l2_load_offset = -border_size().left;

    if((_input->info()->valid_region().anchor[0] + _input->info()->valid_region().shape[0]) % 2 == 0)
    {
        _l2_load_offset += 1;
    }

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), _l2_load_offset, num_elems_read_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

void CLGaussianPyramidHorKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window win_in(window);
    win_in.shift(Window::DimX, _l2_load_offset);

    //The output is half the width of the input:
    Window win_out(window);
    win_out.scale(Window::DimX, 0.5f);

    Window slice_in  = win_in.first_slice_window_2D();
    Window slice_out = win_out.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(win_in.slide_window_slice_2D(slice_in) && win_out.slide_window_slice_2D(slice_out));
}

CLGaussianPyramidVertKernel::CLGaussianPyramidVertKernel()
    : _t2_load_offset(0)
{
}

BorderSize CLGaussianPyramidVertKernel::border_size() const
{
    return BorderSize{ 2, 0 };
}

void CLGaussianPyramidVertKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(0));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Create kernel
    const std::string kernel_name = std::string("gaussian5x1_sub_y");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gaussian5x1_sub_y"));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_rows_processed_per_iteration  = 2;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 8;
    constexpr unsigned int num_rows_per_iteration            = 5;

    const float scale_y = static_cast<float>(output->info()->dimension(1)) / input->info()->dimension(1);

    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration, num_rows_processed_per_iteration));
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, num_rows_per_iteration, 1.f, scale_y);

    // Determine whether we need to load even or odd rows. See above for a
    // detailed explanation.
    _t2_load_offset = -border_size().top;

    if((_input->info()->valid_region().anchor[1] + _input->info()->valid_region().shape[1]) % 2 == 0)
    {
        _t2_load_offset += 1;
    }

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, _t2_load_offset, num_elems_read_per_iteration, num_rows_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

void CLGaussianPyramidVertKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() != 8);
    ARM_COMPUTE_ERROR_ON(window.y().step() % 2);

    Window win_in(window);
    win_in.shift(Window::DimY, _t2_load_offset);

    Window win_out(window);
    win_out.scale(Window::DimY, 0.5f);

    Window slice_in  = win_in.first_slice_window_2D();
    Window slice_out = win_out.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(win_in.slide_window_slice_2D(slice_in) && win_out.slide_window_slice_2D(slice_out));
}
