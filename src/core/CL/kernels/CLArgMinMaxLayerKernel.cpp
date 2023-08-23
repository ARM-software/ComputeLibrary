/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLArgMinMaxLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32, DataType::S64);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op != ReductionOperation::ARG_IDX_MAX && op != ReductionOperation::ARG_IDX_MIN, "Only ARG_IDX_MAX and ARG_IDX_MIN are supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions, "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32, DataType::S32, DataType::S64, DataType::U64);
    }

    return Status{};
}
} // namespace

CLArgMinMaxLayerKernel::CLArgMinMaxLayerKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::ARG_IDX_MAX)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLArgMinMaxLayerKernel::configure(const ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, axis, op);
}

void CLArgMinMaxLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    TensorShape output_shape{ input->info()->tensor_shape() };
    output_shape.set(axis, 1);
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(output_shape).set_data_type(DataType::S32).reset_padding().set_is_resizable(true));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

    auto padding_info = get_padding_info({ input, output });

    _input          = input;
    _output         = output;
    _reduction_axis = axis;
    _op             = op;

    // Set build options
    const auto adjusted_vector_size = adjust_vec_size(16U, input->info()->dimension(0));
    const auto vector_size          = (adjusted_vector_size == 3U && axis == 0U) ? 2U : adjusted_vector_size; // the opencl kernel only supports sizes 2, 4, 8 and 16.

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(input->info()->dimension(0) % vector_size));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vector_size));
    build_opts.add_option_if(is_data_type_float(input->info()->data_type()), "-DFLOAT_DATA_TYPE");
    build_opts.add_option_if_else(op == ReductionOperation::ARG_IDX_MAX, "-DARG_MAX", "-DARG_MIN");
    build_opts.add_option("-DDATA_TYPE_OUTPUT=" + get_cl_type_from_data_type(output->info()->data_type()));
    build_opts.add_option("-DCOND_DATA_TYPE=" + get_cl_select_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DUNROLL_WITH_PRAGMA=1");

    // Create kernel
    std::string kernel_axis_name;
    switch(axis)
    {
        case 0:
            build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
            kernel_axis_name = "x";
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
    Window win = calculate_max_window(*input->info(), Steps(vector_size));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLArgMinMaxLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int axis, ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
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
            Window in_window(window);
            out_window.set(Window::DimX, Window::Dimension(0, 0, 0));
            in_window.set(Window::DimX, Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));
            in_window.set(Window::DimY, Window::Dimension(0, _input->info()->dimension(1), 1u));

            // Get first input and output slices
            Window in_slice  = in_window.first_slice_window_2D();
            Window out_slice = out_window.first_slice_window_2D();
            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                add_2D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(in_window.slide_window_slice_2D(in_slice) && out_window.slide_window_slice_2D(out_slice));
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
