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
#include "arm_compute/core/CL/kernels/CLReductionOperationKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
// OpenCL kernel requires input width to be a power of 2 for x-axis.
constexpr unsigned int border_val = 64;

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op, unsigned int width)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    if(input->num_channels() == 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::S32, DataType::F16, DataType::F32);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F32);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op == ReductionOperation::SUM_SQUARE && input->data_type() == DataType::QASYMM8, "Not supported reduction operation for QASYMM8");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");
    ARM_COMPUTE_RETURN_ERROR_ON(op == ReductionOperation::MEAN_SUM && axis == 0 && width == 0 && input->data_type() != DataType::QASYMM8);

    if(output->total_size() != 0)
    {
        if(op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_type() == DataType::QASYMM8, "Not supported operation for QASYMM8");
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        }
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(axis, 1);
    const bool is_arg_min_max   = (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX);
    DataType   output_data_type = is_arg_min_max ? DataType::U32 : input->data_type();
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_data_type(output_data_type).reset_padding().set_is_resizable(true));

    const unsigned int num_elems_processed_per_iteration = (is_data_type_quantized(input->data_type()) && (axis == 0)) ? 1 : 16;
    Window             win                               = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool               window_changed                    = false;
    const bool         is_serial_op                      = (op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::MIN
                                                            || op == ReductionOperation::MAX || is_data_type_quantized(input->data_type()));

    switch(axis)
    {
        case 0:
        {
            if(is_serial_op)
            {
                AccessWindowHorizontal input_access(input, 0, input->dimension(0));
                AccessWindowHorizontal output_access(output, 0, 1);
                window_changed = update_window_and_padding(win, input_access, output_access);
                output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
            }
            else
            {
                const unsigned int     border_width = ((input->dimension(0) % border_val) != 0) ? border_val - input->dimension(0) % border_val : 0;
                AccessWindowStatic     input_access(input, 0, 0, input->dimension(0) + border_width, 1);
                AccessWindowHorizontal output_access(output, 0, 1);
                window_changed = update_window_and_padding(win, input_access, output_access);
                output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
            }
        }
        break;
        case 1:
        case 2:
        case 3:
        {
            AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
            window_changed = update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }

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

void CLReductionOperationKernel::configure(const ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op, unsigned int width)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op, width));

    _input          = input;
    _output         = output;
    _reduction_axis = axis;
    _op             = op;

    // Set build options
    CLBuildOptions build_opts;
    std::string    data_type_promoted = get_cl_type_from_data_type(input->info()->data_type());
    if(is_data_type_quantized(input->info()->data_type()))
    {
        data_type_promoted = "uint";
    }

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_PROMOTED=" + data_type_promoted);
    build_opts.add_option_if(is_data_type_float(input->info()->data_type()), "-DFLOAT_DATA_TYPE");
    build_opts.add_option_if(op == ReductionOperation::SUM_SQUARE, "-DSUM_SQUARE");
    build_opts.add_option_if(op == ReductionOperation::MEAN_SUM, "-DMEAN");
    build_opts.add_option_if(op == ReductionOperation::ARG_IDX_MAX, "-DARG_MAX");
    build_opts.add_option_if(op == ReductionOperation::ARG_IDX_MIN, "-DARG_MIN");
    build_opts.add_option_if(op == ReductionOperation::PROD, "-DPROD");
    build_opts.add_option_if(op == ReductionOperation::MIN, "-DMIN");
    build_opts.add_option_if(op == ReductionOperation::MAX, "-DMAX");
    build_opts.add_option_if(input->info()->num_channels() == 2, "-DCOMPLEX");

    switch(op)
    {
        case ReductionOperation::SUM_SQUARE:
            build_opts.add_option(("-DOPERATION=square_sum"));
            break;
        case ReductionOperation::SUM:
        case ReductionOperation::MEAN_SUM:
            build_opts.add_option(("-DOPERATION=sum"));
            break;
        case ReductionOperation::ARG_IDX_MAX:
        case ReductionOperation::ARG_IDX_MIN:
        case ReductionOperation::MIN:
        case ReductionOperation::MAX:
            break;
        case ReductionOperation::PROD:
            build_opts.add_option(("-DOPERATION=product"));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation");
    }

    // Create kernel
    cl::NDRange lws_hint = CLKernelLibrary::get().default_ndrange();
    std::string kernel_axis_name;
    const bool  is_serial_op = (op == ReductionOperation::ARG_IDX_MAX || op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::MIN || op == ReductionOperation::MAX
                                || is_data_type_quantized(input->info()->data_type()));
    switch(axis)
    {
        case 0:
        {
            if(is_serial_op)
            {
                build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
                build_opts.add_option_if_else(_input->info()->data_type() == DataType::F16, "-DCOND_DATA_TYPE=short", "-DCOND_DATA_TYPE=int");
                kernel_axis_name = "non_parallel_x";
            }
            else
            {
                build_opts.add_option_if(op == ReductionOperation::MEAN_SUM, "-DWIDTH=" + support::cpp11::to_string(width));
                const unsigned int width_leftover = input->info()->dimension(0) % border_val;
                const unsigned int border_width   = (width_leftover != 0) ? border_val - width_leftover : 0;
                const unsigned int num_of_threads = ((input->info()->dimension(0) + border_width) / 16);
                kernel_axis_name                  = "x";

                // Set the number of WG based on the input size. If input width is < 128
                // we can use fewer threads than 8.
                lws_hint     = cl::NDRange(std::min(8U, num_of_threads));
                _border_size = BorderSize(0, border_width, 0, 0);
            }
        }
        break;
        case 1:
            build_opts.add_option("-DHEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));
            kernel_axis_name = "y";
            break;
        case 2:
            build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));
            kernel_axis_name = "z";
            break;
        case 3:
            build_opts.add_option("-DDEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));
            build_opts.add_option("-DBATCH=" + support::cpp11::to_string(input->info()->dimension(3)));
            kernel_axis_name = "w";
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("reduction_operation_" + kernel_axis_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info(), axis, op);

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config), lws_hint);
}

Status CLReductionOperationKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op, unsigned int width)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op, width));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), axis, op)));

    return Status{};
}

void CLReductionOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    const bool is_serial_op = (_op == ReductionOperation::ARG_IDX_MAX || _op == ReductionOperation::ARG_IDX_MIN || _op == ReductionOperation::MIN || _op == ReductionOperation::MAX
                               || is_data_type_quantized(_input->info()->data_type()));
    switch(_reduction_axis)
    {
        case 0:
        {
            // We use parallel reduction only in non quantized types
            if(is_serial_op)
            {
                // Get first input and output slices
                Window window_in{ window };
                window_in.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));

                Window in_slice  = window.first_slice_window_1D();
                Window out_slice = window.first_slice_window_1D();

                do
                {
                    unsigned int idx = 0;
                    add_1D_tensor_argument(idx, _input, in_slice);
                    add_1D_tensor_argument(idx, _output, out_slice);
                    enqueue(queue, *this, in_slice);
                }
                while(window_in.slide_window_slice_1D(in_slice) && window.slide_window_slice_1D(out_slice));
            }
            else
            {
                // Set out window
                Window out_window(window);
                out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

                // Get first input and output slices
                Window in_slice  = window.first_slice_window_2D();
                Window out_slice = out_window.first_slice_window_2D();

                // Reshape window
                const unsigned int border_width = ((in_slice.x().end() % border_val) != 0) ? border_val - in_slice.x().end() % border_val : 0;
                in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start(), in_slice.x().end() + border_width, in_slice.x().step()));

                // Set local sums buffer
                unsigned int local_res_size = lws_hint()[0] * _input->info()->element_size();
                _kernel.setArg(num_arguments_per_2D_tensor() * 2, local_res_size, nullptr);

                do
                {
                    unsigned int idx = 0;
                    add_2D_tensor_argument(idx, _input, in_slice);
                    add_2D_tensor_argument(idx, _output, out_slice);
                    enqueue(queue, *this, in_slice, lws_hint());
                }
                while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
            }
        }
        break;
        case 1:
        {
            // Get first input and output slices
            Window window_in{ window };
            window_in.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), _input->info()->dimension(1)));
            Window in_slice  = window_in.first_slice_window_2D();
            Window out_slice = window.first_slice_window_2D();

            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                add_2D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            }
            while(window_in.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
        }
        break;
        case 2:
        {
            // Get first input and output slices
            Window window_in{ window };
            window_in.set(Window::DimZ, Window::Dimension(0, _input->info()->dimension(2), _input->info()->dimension(2)));
            Window in_slice  = window_in.first_slice_window_3D();
            Window out_slice = window.first_slice_window_3D();

            do
            {
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, in_slice);
                add_3D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            }
            while(window_in.slide_window_slice_3D(in_slice) && window.slide_window_slice_3D(out_slice));
        }
        break;
        case 3:
        {
            // Get first input and output slices
            Window window_in{ window };
            window_in.set(3, Window::Dimension(0, 1, 1));
            Window in_slice  = window_in.first_slice_window_4D();
            Window out_slice = window.first_slice_window_4D();

            do
            {
                unsigned int idx = 0;
                add_4D_tensor_argument(idx, _input, in_slice);
                add_4D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            }
            while(window_in.slide_window_slice_4D(in_slice) && window.slide_window_slice_4D(out_slice));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
