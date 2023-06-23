/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClElementwiseUnaryKernel.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
constexpr unsigned int vector_size_byte_opencl = 16;

Status validate_arguments(const ITensorInfo &src, const ITensorInfo &dst, const ElementWiseUnary op)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&src);
    if(op == ElementWiseUnary::LOGICAL_NOT)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::U8);
    }
    else if(op == ElementWiseUnary::NEG)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32, DataType::S32);
    }
    else if(op == ElementWiseUnary::RSQRT) // Allow quantized types for only RSQRT.
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32);
    }

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(&dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
    }

    return Status{};
}
} // namespace

ClElementWiseUnaryKernel::ClElementWiseUnaryKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClElementWiseUnaryKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const ElementWiseUnary &op)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    auto padding_info = get_padding_info({ src, dst });

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src, *dst, op));
    const unsigned int num_elems_processed_per_iteration = adjust_vec_size(vector_size_byte_opencl / dst->element_size(), dst->dimension(0));

    std::string kernel_name    = "elementwise_unary";
    const int   vec_size_x     = num_elems_processed_per_iteration;
    const int   dst_width_x    = dst->dimension(0);
    if(is_data_type_quantized(src->data_type()))
    {
        kernel_name += "_quantized";
    }
    // Set kernel build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option("-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(dst_width_x - vec_size_x, 0)));
    if(is_data_type_quantized(src->data_type()))
    {
        const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();
        build_opts.add_option("-DOFFSET_IN=" + support::cpp11::to_string(iqinfo.offset));
        build_opts.add_option("-DOFFSET_OUT=" + support::cpp11::to_string(oqinfo.offset));
        build_opts.add_option("-DSCALE_IN=" + float_to_string_with_full_precision(iqinfo.scale));
        build_opts.add_option("-DSCALE_OUT=" + float_to_string_with_full_precision(oqinfo.scale));
    }
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
    Window win = calculate_max_window(*dst);
    win.set(Window::DimX, Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));

    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClElementWiseUnaryKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const ElementWiseUnary &op)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src, *dst, op));

    return Status{};
}

void ClElementWiseUnaryKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
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
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
