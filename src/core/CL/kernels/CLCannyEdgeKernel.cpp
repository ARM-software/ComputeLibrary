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
#include "arm_compute/core/CL/kernels/CLCannyEdgeKernel.h"

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"

using namespace arm_compute;

CLGradientKernel::CLGradientKernel()
    : _gx(nullptr), _gy(nullptr), _magnitude(nullptr), _phase(nullptr), _pixels_to_skip(0)
{
}

BorderSize CLGradientKernel::border_size() const
{
    return BorderSize(_pixels_to_skip);
}

void CLGradientKernel::configure(const ICLTensor *gx, const ICLTensor *gy, ICLTensor *magnitude, ICLTensor *phase, int32_t norm_type, int32_t num_pixel_to_skip_prev, bool border_undefined)
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

    // Skip pixels around from previous stage
    _pixels_to_skip = num_pixel_to_skip_prev;

    const unsigned int processed_elements = 4;

    // Configure kernel window
    Window                  win = calculate_max_window(*_gx->info(), Steps(processed_elements), border_undefined, border_size());
    AccessWindowAutoPadding magnitude_access(magnitude->info());
    AccessWindowAutoPadding phase_access(phase->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(gx->info()),
                              AccessWindowAutoPadding(gy->info()),
                              magnitude_access,
                              phase_access);

    magnitude_access.set_valid_region();
    phase_access.set_valid_region();

    ICLKernel::configure(win);
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
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}

CLEdgeNonMaxSuppressionKernel::CLEdgeNonMaxSuppressionKernel()
    : _magnitude(nullptr), _phase(nullptr), _output(nullptr), _pixels_to_skip(0)
{
}

BorderSize CLEdgeNonMaxSuppressionKernel::border_size() const
{
    return BorderSize(_pixels_to_skip);
}

void CLEdgeNonMaxSuppressionKernel::configure(const ICLTensor *magnitude, const ICLTensor *phase, ICLTensor *output, int32_t lower_thr, int32_t num_pixel_to_skip_prev, bool border_undefined)
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
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("suppress_non_maximum", built_opts));

    // Pixels to skip
    _pixels_to_skip = num_pixel_to_skip_prev;

    // Set minimum threshold argument

    unsigned int idx = 3 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, lower_thr);

    const unsigned int processed_elements = 1;

    // Configure kernel window
    Window                  win = calculate_max_window(*magnitude->info(), Steps(processed_elements), border_undefined, border_size());
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(magnitude->info()),
                              AccessWindowAutoPadding(phase->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
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
        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}

CLEdgeTraceKernel::CLEdgeTraceKernel()
    : _input(nullptr), _output(nullptr), _lower_thr(0), _upper_thr(0), _visited(nullptr), _recorded(nullptr), _l1_stack(nullptr), _l1_stack_counter(nullptr), _pixels_to_skip(0)
{
}

BorderSize CLEdgeTraceKernel::border_size() const
{
    return BorderSize(_pixels_to_skip);
}

void CLEdgeTraceKernel::configure(const ICLTensor *input, ICLTensor *output, int32_t upper_thr, int32_t lower_thr,
                                  ICLTensor *visited, ICLTensor *recorded, ICLTensor *l1_stack, ICLTensor *l1_stack_counter,
                                  int32_t num_pixel_to_skip_prev, bool border_undefined)
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
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("hysteresis", built_opts));

    // Set constant kernel args
    unsigned int width  = _input->info()->dimension(0);
    unsigned int height = _input->info()->dimension(1);
    unsigned int idx    = 6 * num_arguments_per_2D_tensor(); //Skip the input and output parameters
    _kernel.setArg(idx++, static_cast<cl_uint>(_lower_thr));
    _kernel.setArg(idx++, static_cast<cl_uint>(_upper_thr));
    _kernel.setArg(idx++, static_cast<cl_uint>(width));
    _kernel.setArg(idx++, static_cast<cl_uint>(height));

    // Pixels to skip
    _pixels_to_skip = num_pixel_to_skip_prev;

    const unsigned int processed_elements = 1;

    // Configure kernel window
    Window                  win = calculate_max_window(*input->info(), Steps(processed_elements), border_undefined, border_size());
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    ICLKernel::configure(win);
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

        enqueue(queue, *this, slice);
    }
    while(window.slide_window_slice_2D(slice));
}
