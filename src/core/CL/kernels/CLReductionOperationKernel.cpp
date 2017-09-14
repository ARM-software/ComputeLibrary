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
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLReductionOperationKernel::CLReductionOperationKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::SUM_SQUARE), _border_size()
{
}

BorderSize CLReductionOperationKernel::border_size() const
{
    return _border_size;
}

void CLReductionOperationKernel::configure(const ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);

    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(axis, 1);
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_ERROR_ON_MSG(axis > 0, "Unsupported reduction axis, Supported axis is 0");

    const unsigned int num_elems_processed_per_iteration = 16;
    const unsigned int border_width                      = ((input->info()->dimension(0) % 128) != 0) ? 128 - input->info()->dimension(0) % 128 : 0;

    _input          = input;
    _output         = output;
    _reduction_axis = axis;
    _op             = op;
    _lws_hint       = cl::NDRange(8);
    _border_size    = BorderSize(0, border_width, 0, 0);

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));
    if(is_data_type_fixed_point(input->info()->data_type()))
    {
        build_opts.emplace("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position()));
    }

    switch(op)
    {
        case ReductionOperation::SUM_SQUARE:
            build_opts.emplace(("-DOPERATION=square_sum"));
            break;
        case ReductionOperation::SUM:
            build_opts.emplace(("-DOPERATION=sum"));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reduction_operation", build_opts));

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input->info(), 0, 0, input->info()->dimension(0) + border_width, 1);
    AccessWindowHorizontal output_access(output->info(), 0, 1);

    update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->info()->valid_region());

    ICLKernel::configure(win);
}

void CLReductionOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Set out window
    Window out_window(window);
    out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

    // Get first input and output slices
    Window in_slice  = window.first_slice_window_1D();
    Window out_slice = out_window.first_slice_window_1D();

    // Reshape window
    const unsigned int border_width = ((in_slice.x().end() % 128) != 0) ? 128 - in_slice.x().end() % 128 : 0;
    in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start(), in_slice.x().end() + border_width, in_slice.x().step()));

    // Set local sums buffer
    unsigned int local_sum_size = _lws_hint[0] * _input->info()->element_size();
    _kernel.setArg(num_arguments_per_1D_tensor() * 2, local_sum_size, nullptr);

    do
    {
        unsigned int idx = 0;
        add_1D_tensor_argument(idx, _input, in_slice);
        add_1D_tensor_argument(idx, _output, out_slice);
        enqueue(queue, *this, in_slice, _lws_hint);
    }
    while(window.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(out_slice));
}
