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
#include "arm_compute/core/CL/kernels/CLCannyEdgeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLGradientKernel::CLGradientKernel()
    : _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr)
{
}

void CLGradientKernel::configure(const ICLTensor *gx, const ICLTensor *gy, ICLTensor *magnitude, ICLTensor *phase, int32_t norm_type)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gx, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(gy, 1, DataType::S16, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::U16, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MSG(data_size_from_type(gx->info()->data_type()) != data_size_from_type(gy->info()->data_type()),
                             "Gx and Gy must have the same pixel size");
    ARM_COMPUTE_ERROR_ON_MSG(data_size_from_type(gx->info()->data_type()) != data_size_from_type(magnitude->info()->data_type()),
                             "Mag must have the same pixel size as Gx and Gy");

    _gx        = gx;
    _gy        = gy;
    _magnitude = magnitude;
    _phase     = phase;

    // Create build opts
    std::set<std::string> built_opts;
    built_opts.emplace("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(gx->info()->data_type()));
    built_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(gx->info()->data_type()));

    // Create kernel
    const std::string kernel_name = (norm_type == 1) ? std::string("combine_gradients_L1") : std::string("combine_gradients_L2");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, built_opts));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 4;

    Window win = calculate_max_window(*_gx->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal gx_access(_gx->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal gy_access(_gy->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal mag_access(_magnitude->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal phase_access(_phase->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, gx_access, gy_access, mag_access, phase_access);

    mag_access.set_valid_region(win, _gx->info()->valid_region());
    phase_access.set_valid_region(win, _gx->info()->valid_region());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(gx->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(gx->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(gx->info()->dimension(1));
}

void CLGradientKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _gx, slice);
        add_2D_tensor_argument(idx, _gy, slice);
        add_2D_tensor_argument(idx, _magnitude, slice);
        add_2D_tensor_argument(idx, _phase, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}

CLEdgeNonMaxSuppressionKernel::CLEdgeNonMaxSuppressionKernel()
    : _magnitude(nullptr), _phase(nullptr), _output(nullptr)
{
}

BorderSize CLEdgeNonMaxSuppressionKernel::border_size() const
{
    return BorderSize(1);
}

void CLEdgeNonMaxSuppressionKernel::configure(const ICLTensor *magnitude, const ICLTensor *phase, ICLTensor *output, int32_t lower_thr, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(magnitude, 1, DataType::U16, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(phase, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U16, DataType::U32);

    _magnitude = magnitude;
    _phase     = phase;
    _output    = output;

    // Create build opts
    std::set<std::string> built_opts;
    built_opts.emplace("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(magnitude->info()->data_type()));
    built_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));

    // Create kernel
    const std::string kernel_name = std::string("suppress_non_maximum");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, built_opts));

    // Set minimum threshold argument
    unsigned int idx = 3 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, lower_thr);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration    = 1;
    constexpr unsigned int num_elems_read_written_per_iteration = 3;

    Window win = calculate_max_window(*_magnitude->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowRectangle mag_access(_magnitude->info(), -border_size().left, -border_size().top,
                                     num_elems_read_written_per_iteration, num_elems_read_written_per_iteration);
    AccessWindowHorizontal phase_access(_phase->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(_output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, mag_access, phase_access, output_access);

    output_access.set_valid_region(win, _magnitude->info()->valid_region(), border_undefined, border_size());

    ICLKernel::configure_internal(win);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(output->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_undefined);
}

void CLEdgeNonMaxSuppressionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _magnitude, slice);
        add_2D_tensor_argument(idx, _phase, slice);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}

CLEdgeTraceKernel::CLEdgeTraceKernel()
    : _input(nullptr), _output(nullptr), _lower_thr(0), _upper_thr(0), _visited(nullptr), _recorded(nullptr), _l1_stack(nullptr), _l1_stack_counter(nullptr)
{
}

void CLEdgeTraceKernel::configure(const ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr,
                                  ICLTensor *visited, ICLTensor *recorded, ICLTensor *l1_stack, ICLTensor *l1_stack_counter)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(visited, 1, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(recorded, 1, DataType::U32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(l1_stack, 1, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(l1_stack_counter, 1, DataType::U8);

    _input            = input;
    _output           = output;
    _lower_thr        = lower_thr;
    _upper_thr        = upper_thr;
    _visited          = visited;
    _recorded         = recorded;
    _l1_stack         = l1_stack;
    _l1_stack_counter = l1_stack_counter;

    // Create build opts
    std::set<std::string> built_opts;
    built_opts.emplace("-DDATA_TYPE_IN=" + get_cl_type_from_data_type(input->info()->data_type()));
    built_opts.emplace("-DDATA_TYPE_OUT=" + get_cl_type_from_data_type(output->info()->data_type()));

    // Create kernel
    const std::string kernel_name = std::string("hysteresis");
    _kernel                       = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, built_opts));

    // Set constant kernel args
    unsigned int width  = _input->info()->dimension(0);
    unsigned int height = _input->info()->dimension(1);
    unsigned int idx    = 6 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, static_cast<cl_uint>(_lower_thr));
    _kernel.setArg(idx++, static_cast<cl_uint>(_upper_thr));
    _kernel.setArg(idx++, static_cast<cl_uint>(width));
    _kernel.setArg(idx++, static_cast<cl_uint>(height));

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 1;
    Window                 win                               = calculate_max_window(*_input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal output_access(_output->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal visited_access(_visited->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal recorded_access(_recorded->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal l1_stack_access(_l1_stack->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal l1_stack_counter_access(_l1_stack_counter->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(_input->info(), 0, num_elems_processed_per_iteration),
                              output_access,
                              visited_access,
                              recorded_access,
                              l1_stack_access,
                              l1_stack_counter_access);

    output_access.set_valid_region(win, _input->info()->valid_region());
    visited_access.set_valid_region(win, _input->info()->valid_region());
    recorded_access.set_valid_region(win, _input->info()->valid_region());
    l1_stack_access.set_valid_region(win, _input->info()->valid_region());
    l1_stack_counter_access.set_valid_region(win, _input->info()->valid_region());

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
    _config_id += lower_string(string_from_format(output->info()->format()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

void CLEdgeTraceKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice = window.first_slice_window_2D();
    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice);
        add_2D_tensor_argument(idx, _output, slice);
        add_2D_tensor_argument(idx, _visited, slice);
        add_2D_tensor_argument(idx, _recorded, slice);
        add_2D_tensor_argument(idx, _l1_stack, slice);
        add_2D_tensor_argument(idx, _l1_stack_counter, slice);

        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_2D(slice));
}
