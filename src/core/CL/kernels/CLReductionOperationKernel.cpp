/*
 * Copyright (c) 2017-2018 ARM Limited.
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

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_UNUSED(op);

    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 0, "Unsupported reduction axis, Supported axis is 0");

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(output->data_layout() != DataLayout::NCHW);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int axis)
{
    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(axis, 1);
    auto_init_if_empty(*output, output_shape, 1, input->data_type(), input->fixed_point_position());

    const unsigned int num_elems_processed_per_iteration = 16;

    Window             win          = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    const unsigned int border_width = ((input->dimension(0) % 128) != 0) ? 128 - input->dimension(0) % 128 : 0;

    AccessWindowStatic     input_access(input, 0, 0, input->dimension(0) + border_width, 1);
    AccessWindowHorizontal output_access(output, 0, 1);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, output->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};

    return std::make_tuple(err, win);
}
} // namespace

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
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

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
    auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis);

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure(std::get<1>(win_config));
}

Status CLReductionOperationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), axis)));

    return Status{};
}

void CLReductionOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Set out window
    Window out_window(window);
    out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

    // Get first input and output slices
    Window in_slice  = window.first_slice_window_2D();
    Window out_slice = out_window.first_slice_window_2D();

    // Reshape window
    const unsigned int border_width = ((in_slice.x().end() % 128) != 0) ? 128 - in_slice.x().end() % 128 : 0;
    in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start(), in_slice.x().end() + border_width, in_slice.x().step()));

    // Set local sums buffer
    unsigned int local_sum_size = _lws_hint[0] * _input->info()->element_size();
    _kernel.setArg(num_arguments_per_2D_tensor() * 2, local_sum_size, nullptr);

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, in_slice);
        add_2D_tensor_argument(idx, _output, out_slice);
        enqueue(queue, *this, in_slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
}
