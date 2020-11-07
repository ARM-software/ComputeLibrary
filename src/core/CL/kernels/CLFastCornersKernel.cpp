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
#include "src/core/CL/kernels/CLFastCornersKernel.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

#include <set>
#include <string>

using namespace arm_compute;

CLFastCornersKernel::CLFastCornersKernel()
    : ICLKernel(), _input(nullptr), _output(nullptr)
{
}

BorderSize CLFastCornersKernel::border_size() const
{
    return BorderSize(3);
}

void CLFastCornersKernel::configure(const ICLImage *input, ICLImage *output, float threshold, bool non_max_suppression, BorderMode border_mode)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, threshold, non_max_suppression, border_mode);
}

void CLFastCornersKernel::configure(const CLCompileContext &compile_context, const ICLImage *input, ICLImage *output, float threshold, bool non_max_suppression, BorderMode border_mode)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MSG(border_mode != BorderMode::UNDEFINED, "Not implemented");

    _input  = input;
    _output = output;

    // Create build options
    std::set<std::string> build_opts;

    if(non_max_suppression)
    {
        build_opts.emplace("-DUSE_MAXSUPPRESSION");
    }

    // Create kernel
    const std::string kernel_name = std::string("fast_corners");
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts);

    // Set static kernel arguments
    unsigned int idx = 2 * num_arguments_per_2D_tensor(); // Skip the input and output parameters
    _kernel.setArg<cl_float>(idx, static_cast<float>(threshold));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;
    constexpr unsigned int num_elems_read_per_iteration      = 7;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_mode == BorderMode::UNDEFINED, BorderSize(3));

    AccessWindowHorizontal output_access(output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_mode == BorderMode::UNDEFINED, border_size());

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
    _config_id += "_";
    _config_id += support::cpp11::to_string(non_max_suppression);
    _config_id += "_";
    _config_id += lower_string(string_from_border_mode(border_mode));
}

void CLFastCornersKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}

CLCopyToArrayKernel::CLCopyToArrayKernel()
    : ICLKernel(), _input(nullptr), _corners(nullptr), _num_buffer(nullptr)
{
}

void CLCopyToArrayKernel::configure(const ICLImage *input, bool update_number, ICLKeyPointArray *corners, cl::Buffer *num_buffers)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, update_number, corners, num_buffers);
}

void CLCopyToArrayKernel::configure(const CLCompileContext &compile_context, const ICLImage *input, bool update_number, ICLKeyPointArray *corners, cl::Buffer *num_buffers)
{
    ARM_COMPUTE_ERROR_ON_TENSOR_NOT_2D(input);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(corners == nullptr);
    ARM_COMPUTE_ERROR_ON(num_buffers == nullptr);

    _input      = input;
    _corners    = corners;
    _num_buffer = num_buffers;

    std::set<std::string> build_opts;

    if(update_number)
    {
        build_opts.emplace("-DUPDATE_NUMBER");
    }

    // Create kernel
    const std::string kernel_name = std::string("copy_to_keypoint");
    _kernel                       = create_kernel(compile_context, kernel_name, build_opts);

    //Get how many pixels skipped in the x dimension in the previous stages
    unsigned int offset = _input->info()->valid_region().anchor.x();

    // Set static kernel arguments
    unsigned int idx = num_arguments_per_2D_tensor(); // Skip the input and output parameters
    _kernel.setArg<unsigned int>(idx++, _corners->max_num_values());
    _kernel.setArg<cl_uint>(idx++, offset);
    _kernel.setArg(idx++, *_num_buffer);
    _kernel.setArg(idx++, _corners->cl_buffer());

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;
    Window                 win                               = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration));
    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
}

void CLCopyToArrayKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    //Initialise the _num_buffer as it used as both input and output
    static const unsigned int zero_init = 0;
    queue.enqueueWriteBuffer(*_num_buffer, CL_FALSE, 0, sizeof(unsigned int), &zero_init);

    Window slice = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
