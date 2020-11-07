/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLIntegralImageKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <cstddef>

using namespace arm_compute;

void CLIntegralImageHorKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output);
}

void CLIntegralImageHorKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32);

    _input  = input;
    _output = output;

    // Create kernel
    const std::string kernel_name = std::string("integral_horizontal");
    _kernel                       = create_kernel(compile_context, kernel_name);

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = input->info()->dimension(0);
    const unsigned int num_elems_accessed_per_iteration  = ceil_to_multiple(num_elems_processed_per_iteration, 16);

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_accessed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_accessed_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region());

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

CLIntegralImageVertKernel::CLIntegralImageVertKernel()
    : _in_out(nullptr)
{
}

void CLIntegralImageVertKernel::configure(ICLTensor *in_out)
{
    configure(CLKernelLibrary::get().get_compile_context(), in_out);
}

void CLIntegralImageVertKernel::configure(const CLCompileContext &compile_context, ICLTensor *in_out)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(in_out, 1, DataType::U32);

    _in_out = in_out;

    // Create kernel
    const std::string kernel_name = std::string("integral_vertical");
    _kernel                       = create_kernel(compile_context, kernel_name);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration_x = 8;
    const unsigned int     num_elems_processed_per_iteration_y = in_out->info()->dimension(Window::DimY);

    Window win = calculate_max_window(*in_out->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowRectangle in_out_access(in_out->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

    update_window_and_padding(win, in_out_access);

    in_out_access.set_valid_region(win, in_out->info()->valid_region());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(in_out->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(in_out->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(in_out->info()->dimension(1));
}

void CLIntegralImageVertKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const size_t height = _in_out->info()->dimension(1);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _in_out, slice);
        _kernel.setArg<cl_uint>(idx++, height);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
