/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32);

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::F16, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &output);
    }

    return Status{};
}
} // namespace

void CLElementWiseUnaryLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const ElementWiseUnary &op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input->info(), *output->info()));

    // Configure kernel window
    _input  = input;
    _output = output;

    const std::string kernel_name    = "elementwise_unary";
    const int         vec_size_x     = 16 / output->info()->element_size();
    const int         output_width_x = output->info()->tensor_shape().x();
    const bool        multi_access_x = (output_width_x / vec_size_x > 0);

    Window win = calculate_max_window(*output->info());
    if(multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    // Set kernel build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if(multi_access_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(output_width_x - vec_size_x, 0)));
    switch(op)
    {
        case ElementWiseUnary::RSQRT:
            build_opts.add_option("-DOPERATION=rsqrt_op");
            break;
        case ElementWiseUnary::EXP:
            build_opts.add_option("-DOPERATION=exp_op");
            break;
        case ElementWiseUnary::NEG:
            build_opts.add_option("-DOPERATION=neg_op");
            break;
        case ElementWiseUnary::SIN:
            build_opts.add_option("-DOPERATION=sin_op");
            break;
        case ElementWiseUnary::ABS:
            build_opts.add_option("-DOPERATION=fabs_op");
            break;
        case ElementWiseUnary::LOG:
            build_opts.add_option("-DOPERATION=natural_log_op");
            break;
        case ElementWiseUnary::ROUND:
            build_opts.add_option("-DOPERATION=round_op");
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));
}

Status CLElementWiseUnaryLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ElementWiseUnary &op)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input, *output));

    return Status{};
}

void CLElementWiseUnaryLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimX);

    do
    {
        unsigned int idx = 0;
        add_1D_tensor_argument(idx, _input, collapsed);
        add_1D_tensor_argument(idx, _output, collapsed);
        enqueue(queue, *this, collapsed);
    }
    while(window.slide_window_slice_1D(collapsed));
}
