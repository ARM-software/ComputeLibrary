/*
 * Copyright (c) 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClMatMulLowpNativeMMULKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/helpers/MatMulKernelHelpers.h"
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
// Block size dimensions for the MMUL extension
constexpr int mmul_m0 = 4;
constexpr int mmul_n0 = 4;
constexpr int mmul_k0 = 16;

Status validate_matmul_kernel_info(const MatMulKernelInfo &matmul_kernel_info)
{
    ARM_COMPUTE_UNUSED(matmul_kernel_info);
    // TODO: Validate MatMulKernelInfo
    return Status{};
}
} // namespace

ClMatMulLowpNativeMMULKernel::ClMatMulLowpNativeMMULKernel()
{
    _type = CLKernelType::GEMM;
}

Status ClMatMulLowpNativeMMULKernel::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *bias, const ITensorInfo *dst, const MatMulKernelInfo &matmul_kernel_info,
                                              const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, rhs);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_matmul_kernel_info(matmul_kernel_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_matmul_input_shapes(lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info));
    // TODO: Check MMUL block sizes against the tensor shapes
    ARM_COMPUTE_UNUSED(mmul_k0);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((act_info.activation() != ActivationFunction::IDENTITY && act_info.activation() != ActivationFunction::RELU
                                     && act_info.activation() != ActivationFunction::LU_BOUNDED_RELU && act_info.activation() != ActivationFunction::BOUNDED_RELU),
                                    "Activation Function specified is unsupported.");
    const TensorShape expected_output_shape = misc::shape_calculator::compute_matmul_shape(lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info);

    if(dst->total_size() != 0)
    {
        const TensorInfo tensor_info_output = dst->clone()->set_tensor_shape(expected_output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(lhs, dst);
    }

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(expected_output_shape[0] != bias->dimension(0));
    }

    return Status{};
}

void ClMatMulLowpNativeMMULKernel::configure(const ClCompileContext &compile_context, ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *bias, ITensorInfo *dst,
                                             const MatMulKernelInfo    &matmul_kernel_info,
                                             const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, bias, dst, matmul_kernel_info, act_info);
    ARM_COMPUTE_ERROR_THROW_ON(validate(lhs, rhs, bias, dst, matmul_kernel_info));

    // dst tensor auto initialization if not yet initialized
    auto_init_if_empty(*dst, lhs->clone()->set_tensor_shape(misc::shape_calculator::compute_matmul_shape(lhs->tensor_shape(), rhs->tensor_shape(), matmul_kernel_info)));

    ARM_COMPUTE_UNUSED(compile_context, lhs, rhs, bias, matmul_kernel_info, act_info);

    // Configure kernel window
    const auto win_config = validate_and_configure_window_for_mmul_kernels(lhs, rhs, dst, matmul_kernel_info, mmul_m0, mmul_n0);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    IClKernel::configure_internal(win_config.second);

    CLBuildOptions build_opts;
    // TODO: Build options & configuration

    std::string kernel_name("mat_mul_native_quantized_mmul");
    kernel_name += matmul_kernel_info.adj_lhs ? "_t" : "_nt";
    kernel_name += matmul_kernel_info.adj_rhs ? "_t" : "_nt";

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // TODO: Tuner configuration
}

void ClMatMulLowpNativeMMULKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto *lhs  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto *rhs  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto *bias = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2)); // nullptr if bias is not present
    auto       *dst  = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, bias, dst);

    unsigned int idx = 0;
    add_3d_tensor_nhw_argument(idx, lhs);
    add_3d_tensor_nhw_argument(idx, rhs);

    if(bias != nullptr)
    {
        add_3d_tensor_nhw_argument(idx, bias);
    }
    add_3d_tensor_nhw_argument(idx, dst);

    // LWS_x should be multiple of 16 at least. (32, 2) has been chosen to have more work-items on a single core
    // LWS also enforces the order of execution of the work items which improves cache utilization
    enqueue(queue, *this, window, cl::NDRange(32, 2), false);
}

} // namespace kernels
} // namespace opencl
} // namespace arm_compute
