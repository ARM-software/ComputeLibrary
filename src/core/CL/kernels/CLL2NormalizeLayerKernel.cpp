/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLL2NormalizeLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
constexpr int max_input_tensor_dim = 3;

constexpr unsigned int num_elems_processed_per_iteration = 16;

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);

    const uint32_t actual_axis = wrap_around(axis, max_input_tensor_dim);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, sum);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis > 2, "Actual axis greater than 2 is not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(actual_axis >= TensorShape::num_max_dimensions, "Actual normalization axis greater than max number of dimensions");

    // Reduce shape on axis
    TensorShape sum_shape = input->tensor_shape();
    sum_shape.set(actual_axis, 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(sum->tensor_shape(), sum_shape);

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(input->tensor_shape(), output->tensor_shape());
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    Window win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};

    return std::make_tuple(err, win);
}
} // namespace

CLL2NormalizeLayerKernel::CLL2NormalizeLayerKernel()
    : _input(nullptr), _sum(nullptr), _output(nullptr), _actual_axis(0), _epsilon(1e-12)
{
}

void CLL2NormalizeLayerKernel::configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, int axis, float epsilon)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, sum, output, axis, epsilon);
}

void CLL2NormalizeLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, int axis, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, sum, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), sum->info(), output->info(), axis, epsilon));

    _input       = input;
    _sum         = sum;
    _output      = output;
    _actual_axis = wrap_around(axis, max_input_tensor_dim);
    _epsilon     = epsilon;

    // Set build options
    std::set<std::string> build_opts;
    build_opts.emplace(("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type())));
    build_opts.emplace(("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_processed_per_iteration)));

    // Create kernel
    std::string  kernel_name;
    unsigned int idx = 0;
    switch(_actual_axis)
    {
        case 0:
            kernel_name = "x";
            idx         = num_arguments_per_2D_tensor() * 3;
            break;
        case 1:
            kernel_name = "y";
            idx         = num_arguments_per_2D_tensor() * 3;
            break;
        case 2:
            kernel_name = "z";
            idx         = num_arguments_per_3D_tensor() * 3;
            break;
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
    _kernel = create_kernel(compile_context, "l2_normalize_" + kernel_name, build_opts);

    // Set epsilon argument
    if(input->info()->data_type() == DataType::F32)
    {
        _kernel.setArg<cl_float>(idx, _epsilon);
    }
    else
    {
        _kernel.setArg<cl_half>(idx, _epsilon);
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));
}

Status CLL2NormalizeLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, int axis, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, sum, output, axis, epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

    return Status{};
}

void CLL2NormalizeLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_sum(window);

    switch(_actual_axis)
    {
        case 0:
        {
            window_sum.set(Window::DimX, Window::Dimension(0, 0, 0));
            Window in_slice  = window.first_slice_window_2D();
            Window sum_slice = window_sum.first_slice_window_2D();
            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                add_2D_tensor_argument(idx, _sum, sum_slice);
                add_2D_tensor_argument(idx, _output, in_slice);
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(sum_slice));
        }
        break;
        case 1:
        {
            window_sum.set(Window::DimY, Window::Dimension(0, 0, 0));
            Window in_slice  = window.first_slice_window_2D();
            Window sum_slice = window_sum.first_slice_window_2D();
            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, in_slice);
                add_2D_tensor_argument(idx, _sum, sum_slice);
                add_2D_tensor_argument(idx, _output, in_slice);
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(window.slide_window_slice_2D(in_slice) && window.slide_window_slice_2D(sum_slice));
        }
        break;
        case 2:
        {
            window_sum.set(Window::DimZ, Window::Dimension(0, 0, 0));
            Window in_slice  = window.first_slice_window_3D();
            Window sum_slice = window_sum.first_slice_window_3D();
            do
            {
                unsigned int idx = 0;
                add_3D_tensor_argument(idx, _input, in_slice);
                add_3D_tensor_argument(idx, _sum, sum_slice);
                add_3D_tensor_argument(idx, _output, in_slice);
                enqueue(queue, *this, in_slice, lws_hint());
            }
            while(window.slide_window_slice_3D(in_slice) && window.slide_window_slice_3D(sum_slice));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
} // namespace arm_compute
