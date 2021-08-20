/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClReshapeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

#include <string>

/** [ClReshapeKernel Kernel] **/
namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);

    if(dst->tensor_shape().total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().total_size() != dst->tensor_shape().total_size());
    }

    return Status{};
}
} // namespace

ClReshapeKernel::ClReshapeKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClReshapeKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst));

    auto padding_info = get_padding_info({ src, dst });

    // Create kernel
    std::set<std::string> build_opts = { "-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(src->element_size()) };
    _kernel                          = create_kernel(compile_context, "reshape_layer", build_opts);

    // Add static arguments
    const cl_int2 src_shape =
    {
        {
            static_cast<cl_int>(src->tensor_shape()[0]),
            static_cast<cl_int>(src->tensor_shape()[1])
        }
    };
    const cl_int2 dst_shape =
    {
        {
            static_cast<cl_int>(dst->tensor_shape()[0]),
            static_cast<cl_int>(dst->tensor_shape()[1])
        }
    };
    unsigned int idx = 2 * num_arguments_per_3D_tensor(); // Skip the src and dst parameters
    _kernel.setArg<cl_int2>(idx++, src_shape);
    _kernel.setArg<cl_int2>(idx++, dst_shape);

    // Configure kernel window
    Window win = calculate_max_window(*src);
    ICLKernel::configure_internal(win);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClReshapeKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst));

    return Status{};
}

void ClReshapeKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice            = window_collapsed.first_slice_window_3D();

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    // Set srcs
    unsigned int idx = 0;
    add_3D_tensor_argument(idx, src, window_collapsed);
    add_3D_tensor_argument(idx, dst, window_collapsed);
    enqueue(queue, *this, slice, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
/** [ClReshapeKernel Kernel] **/
