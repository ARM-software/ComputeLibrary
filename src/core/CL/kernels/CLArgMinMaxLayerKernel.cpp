/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLArgMinMaxLayerKernel.h"

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

#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
constexpr unsigned int vector_size = 16;

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *prev_output, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op != ReductionOperation::ARG_IDX_MAX && op != ReductionOperation::ARG_IDX_MIN, "Only ARG_IDX_MAX and ARG_IDX_MIN are supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32, DataType::S32);
    }
    if(prev_output != nullptr && prev_output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(prev_output, 1, DataType::U32, DataType::S32);
        if(output->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(prev_output, output);
        }
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *prev_output, ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_UNUSED(op);
    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape{ input->tensor_shape() };
    output_shape.set(axis, 1);
    DataType output_data_type = DataType::S32;
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_data_type(output_data_type).reset_padding().set_is_resizable(true));

    Window win            = calculate_max_window((prev_output != nullptr) ? (*prev_output) : (*input), Steps(vector_size));
    bool   window_changed = false;

    switch(axis)
    {
        case 0:
        {
            ITensorInfo           *input_tensor_access = prev_output != nullptr ? prev_output : input;
            AccessWindowStatic     input_access(input_tensor_access, 0, 0, static_cast<int>(input_tensor_access->dimension(0)), 1);
            AccessWindowHorizontal output_access(output, 0, 1);
            window_changed = update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        }
        break;
        case 1:
        case 2:
        case 3:
        {
            AccessWindowHorizontal input_access(input, 0, vector_size);
            AccessWindowHorizontal output_access(output, 0, vector_size);
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

CLArgMinMaxLayerKernel::CLArgMinMaxLayerKernel()
    : _input(nullptr), _prev_output(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::ARG_IDX_MAX)
{
}

void CLArgMinMaxLayerKernel::configure(const ICLTensor *input, const ICLTensor *prev_output, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, prev_output, output, axis, op);
}

void CLArgMinMaxLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *prev_output, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (prev_output != nullptr) ? prev_output->info() : nullptr, output->info(), axis, op));
    auto win_config = validate_and_configure_window(input->info(), (prev_output != nullptr) ? prev_output->info() : nullptr, output->info(), axis, op);
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    _input          = input;
    _prev_output    = prev_output;
    _output         = output;
    _reduction_axis = axis;
    _op             = op;

    // Set build options
    CLBuildOptions build_opts;

    build_opts.add_option_if(_prev_output != nullptr, "-DPREV_OUTPUT");
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(is_data_type_float(input->info()->data_type()), "-DFLOAT_DATA_TYPE");
    build_opts.add_option_if_else(op == ReductionOperation::ARG_IDX_MAX, "-DARG_MAX", "-DARG_MIN");
    build_opts.add_option("-DDATA_TYPE_OUTPUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    build_opts.add_option("-DDATA_TYPE_SELECT=" + get_cl_signed_type_from_element_size(input->info()->element_size()));

    // Create kernel
    cl::NDRange lws_hint = CLKernelLibrary::get().default_ndrange();
    std::string kernel_axis_name;
    switch(axis)
    {
        case 0:
        {
            const ICLTensor *input_for_width = prev_output != nullptr ? _prev_output : _input;
            build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(input_for_width->info()->dimension(0)));

            kernel_axis_name = "x";
            lws_hint         = create_lws_hint_parallel_implementations(input_for_width->info()->dimension(0), vector_size);
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
    _kernel = create_kernel(compile_context, "arg_min_max_" + kernel_axis_name, build_opts.options());

    // Configure kernel window
    ICLKernel::configure_internal(std::get<1>(win_config), lws_hint);
}

Status CLArgMinMaxLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *prev_output, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, prev_output, output, axis, op));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), (prev_output != nullptr) ? prev_output->clone().get() : nullptr, output->clone().get(), axis, op)));
    return Status{};
}

void CLArgMinMaxLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    switch(_reduction_axis)
    {
        case 0:
        {
            // Set out window
            Window out_window(window);
            out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

            // Get first input and output slices
            Window in_slice  = window.first_slice_window_2D();
            Window out_slice = out_window.first_slice_window_2D();

            // Reshape window
            const unsigned int num_tensors = _prev_output != nullptr ? 3 : 2;

            // Set local sums buffer
            unsigned int local_res_size = lws_hint()[0] * _output->info()->element_size();
            _kernel.setArg(num_arguments_per_2D_tensor() * num_tensors, local_res_size, nullptr);
            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                if(_prev_output != nullptr)
                {
                    add_2D_tensor_argument(idx, _prev_output, in_slice);
                }
                add_2D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
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
                enqueue(queue, *this, in_slice, lws_hint());
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
                enqueue(queue, *this, in_slice, lws_hint());
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
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(window_in.slide_window_slice_4D(in_slice) && window.slide_window_slice_4D(out_slice));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
} // namespace arm_compute
