/*
 * Copyright (c) 2017-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLReductionOperationKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/Validate.h"

#include "src/core/AccessWindowStatic.h"
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
    if (input->num_channels() == 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                             DataType::S32, DataType::F16, DataType::F32);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 2, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON(axis == 0);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(op == ReductionOperation::SUM_SQUARE && input->data_type() == DataType::QASYMM8,
                                    "Not supported reduction operation for QASYMM8");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis >= TensorShape::num_max_dimensions,
                                    "Reduction axis greater than max number of dimensions");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(axis > 3, "Unsupported reduction axis");
    ARM_COMPUTE_RETURN_ERROR_ON((op == ReductionOperation::MEAN_SUM) && (axis == 0) && (input->dimension(0) == 0) &&
                                (input->data_type() != DataType::QASYMM8) &&
                                (input->data_type() != DataType::QASYMM8_SIGNED));
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((op == ReductionOperation::ARG_IDX_MAX) || (op == ReductionOperation::ARG_IDX_MIN),
                                    "Not supported reduction operation, use CLArgMinMaxLayer");

    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    }

    return Status{};
}
} // namespace

CLReductionOperationKernel::CLReductionOperationKernel()
    : _input(nullptr), _output(nullptr), _reduction_axis(0), _op(ReductionOperation::SUM_SQUARE)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLReductionOperationKernel::configure(const ICLTensor   *input,
                                           ICLTensor         *output,
                                           unsigned int       axis,
                                           ReductionOperation op)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, axis, op);
}

void CLReductionOperationKernel::configure(const CLCompileContext &compile_context,
                                           const ICLTensor        *input,
                                           ICLTensor              *output,
                                           unsigned int            axis,
                                           ReductionOperation      op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), axis, op));

    auto padding_info = get_padding_info({input, output});

    _input          = input;
    _output         = output;
    _reduction_axis = axis;
    _op             = op;

    const TensorShape output_shape =
        arm_compute::misc::shape_calculator::compute_reduced_shape(input->info()->tensor_shape(), axis, true);
    auto_init_if_empty(*output->info(),
                       input->info()->clone()->set_tensor_shape(output_shape).reset_padding().set_is_resizable(true));

    // Set build options
    CLBuildOptions build_opts;
    DataType       data_type = input->info()->data_type();
    std::string    data_type_promoted{};

    if (is_data_type_quantized(data_type))
    {
        data_type_promoted = "int";
    }
    else
    {
        data_type_promoted = get_cl_type_from_data_type(data_type);
    }

    const unsigned int width             = input->info()->dimension(0) * input->info()->num_channels();
    unsigned int       vec_size          = (is_data_type_quantized(input->info()->data_type()) && (axis == 0)) ? 1 : 16;
    vec_size                             = adjust_vec_size(vec_size, width);
    const unsigned int vec_size_leftover = width % vec_size;

    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DDATA_TYPE_PROMOTED=" + data_type_promoted);
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size));
    build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_leftover));
    build_opts.add_option_if(is_data_type_float(data_type), "-DFLOAT_DATA_TYPE");
    build_opts.add_option_if(op == ReductionOperation::SUM_SQUARE, "-DSUM_SQUARE");
    build_opts.add_option_if(op == ReductionOperation::MEAN_SUM, "-DMEAN");
    build_opts.add_option_if(op == ReductionOperation::SUM, "-DSUM");
    build_opts.add_option_if(op == ReductionOperation::PROD, "-DPROD");
    build_opts.add_option_if(op == ReductionOperation::MIN, "-DMIN");
    build_opts.add_option_if(op == ReductionOperation::MAX, "-DMAX");
    build_opts.add_option_if(is_data_type_quantized(data_type),
                             "-DOFFSET=" +
                                 support::cpp11::to_string(input->info()->quantization_info().uniform().offset));
    build_opts.add_option_if(
        is_data_type_quantized(data_type),
        "-DSCALE=" + float_to_string_with_full_precision(input->info()->quantization_info().uniform().scale));

    switch (op)
    {
        case ReductionOperation::SUM_SQUARE:
            build_opts.add_option(("-DOPERATION=square_sum"));
            break;
        case ReductionOperation::SUM:
        case ReductionOperation::MEAN_SUM:
            build_opts.add_option(("-DOPERATION=sum"));
            break;
        case ReductionOperation::MIN:
            build_opts.add_option(("-DOPERATION=min_"));
            break;
        case ReductionOperation::MAX:
            build_opts.add_option(("-DOPERATION=max_"));
            break;
        case ReductionOperation::PROD:
            build_opts.add_option(("-DOPERATION=product"));
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation");
    }

    // Create kernel
    std::string kernel_axis_name;
    const bool  is_serial_op = needs_serialized_reduction(_op, _input->info()->data_type(), _reduction_axis);

    switch (axis)
    {
        case 0:
        {
            build_opts.add_option("-DWIDTH=" + support::cpp11::to_string(width));
            kernel_axis_name = ((is_serial_op) ? "non_parallel_x" : "x");
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
    _kernel = create_kernel(compile_context, "reduction_operation_" + kernel_axis_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(vec_size));
    win.set(Window::DimX,
            Window::Dimension(win.x().start(), win.x().end() * _input->info()->num_channels(), win.x().step()));
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLReductionOperationKernel::validate(const ITensorInfo *input,
                                            const ITensorInfo *output,
                                            unsigned int       axis,
                                            ReductionOperation op)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, axis, op));
    return Status{};
}

void CLReductionOperationKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    const bool is_serial_op = needs_serialized_reduction(_op, _input->info()->data_type(), _reduction_axis);
    switch (_reduction_axis)
    {
        case 0:
        {
            // We use parallel reduction only in non quantized types
            if (is_serial_op)
            {
                // Get first input and output slices
                Window window_in{window};
                window_in.set(Window::DimX,
                              Window::Dimension(0, _input->info()->dimension(0), _input->info()->dimension(0)));

                Window out_window{window};
                out_window.set(Window::DimX, Window::Dimension(0, 0, 0));

                Window in_slice  = window_in.first_slice_window_1D();
                Window out_slice = out_window.first_slice_window_1D();

                do
                {
                    unsigned int idx = 0;
                    add_1D_tensor_argument(idx, _input, in_slice);
                    add_1D_tensor_argument(idx, _output, out_slice);
                    enqueue(queue, *this, in_slice);
                } while (window_in.slide_window_slice_1D(in_slice) && out_window.slide_window_slice_1D(out_slice));
            }
            else
            {
                // Set out window
                bool   has_collapsed = true;
                Window window_in     = window.collapse_if_possible(window, 2, &has_collapsed);
                ARM_COMPUTE_ERROR_ON(!has_collapsed);

                Window window_out = window_in;
                window_out.set(0, Window::Dimension());

                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, window_in);
                add_3D_tensor_argument(idx, _output, window_out);
                enqueue(queue, *this, window_in);
            }
        }
        break;
        case 1:
        {
            // Get first input and output slices
            Window window_in{window};
            window_in.set(Window::DimY,
                          Window::Dimension(0, _input->info()->dimension(1), _input->info()->dimension(1)));
            Window in_slice  = window_in.first_slice_window_2D();
            Window out_slice = window.first_slice_window_2D();

            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                add_2D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            } while (window_in.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(out_slice));
        }
        break;
        case 2:
        {
            // Get first input and output slices
            Window window_in{window};
            window_in.set(Window::DimZ,
                          Window::Dimension(0, _input->info()->dimension(2), _input->info()->dimension(2)));
            Window in_slice  = window_in.first_slice_window_3D();
            Window out_slice = window.first_slice_window_3D();

            do
            {
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, in_slice);
                add_3D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            } while (window_in.slide_window_slice_3D(in_slice) && window.slide_window_slice_3D(out_slice));
        }
        break;
        case 3:
        {
            // Get first input and output slices
            Window window_in{window};
            window_in.set(3, Window::Dimension(0, 1, 1));
            Window in_slice  = window_in.first_slice_window_4D();
            Window out_slice = window.first_slice_window_4D();

            do
            {
                unsigned int idx = 0;
                add_4D_tensor_argument(idx, _input, in_slice);
                add_4D_tensor_argument(idx, _output, out_slice);
                enqueue(queue, *this, in_slice);
            } while (window_in.slide_window_slice_4D(in_slice) && window.slide_window_slice_4D(out_slice));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
} // namespace arm_compute
