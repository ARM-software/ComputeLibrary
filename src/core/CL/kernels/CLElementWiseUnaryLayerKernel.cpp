/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLElementWiseUnaryLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo &input, const ITensorInfo &output, const ElementWiseUnary op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&input);
    if(op == ElementWiseUnary::LOGICAL_NOT)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::U8);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32);
    }

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&input, &output);
    }

    return Status{};
}
} // namespace

void CLElementWiseUnaryLayerKernel::configure(const ITensorInfo *input, ITensorInfo *output, const ElementWiseUnary &op)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, op);
}

void CLElementWiseUnaryLayerKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *input, ITensorInfo *output, const ElementWiseUnary &op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    auto padding_info = get_padding_info({ input, output });

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input, *output, op));

    const std::string kernel_name    = "elementwise_unary";
    const int         vec_size_x     = 16 / output->element_size();
    const int         output_width_x = output->tensor_shape().x();
    const bool        multi_access_x = (output_width_x / vec_size_x > 0);

    // Set kernel build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->data_type()));
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
        case ElementWiseUnary::LOGICAL_NOT:
            build_opts.add_option("-DOPERATION=logical_not_op");
            break;
        default:
            ARM_COMPUTE_ERROR("Not implemented");
    }

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    Window win = calculate_max_window(*output);
    if(multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status CLElementWiseUnaryLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ElementWiseUnary &op)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input, *output, op));

    return Status{};
}

void CLElementWiseUnaryLayerKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, src, slice);
        add_3D_tensor_argument(idx, dst, slice);
        enqueue(queue, *this, slice, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice));
}
} // namespace arm_compute
