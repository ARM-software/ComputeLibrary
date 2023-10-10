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
#include "src/gpu/cl/operators/ClMatMul.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClMatMulLowpNativeKernel.h"
#include "src/gpu/cl/kernels/ClMatMulLowpNativeMMULKernel.h"
#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"
#include "src/gpu/cl/kernels/ClMatMulNativeMMULKernel.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeDefaultConfigValhall.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"

using namespace arm_compute::cl_matmul;

namespace arm_compute
{
namespace opencl
{
namespace
{
enum class MatMulKernelType
{
    /** Native matrix multiplication for FP types */
    NATIVE_FP,

    /** Native matrix multiplication for quantized types */
    NATIVE_QUANTIZED,

    /** Native matrix multiplication using MMUL extension for FP types */
    NATIVE_MMUL_FP,

    /** Native matrix multiplication using MMUL extension for Quantized types */
    NATIVE_MMUL_QUANTIZED
};

MatMulKernelType get_matmul_kernel(const ITensorInfo         *lhs,
                                   const ITensorInfo         *rhs,
                                   const MatMulInfo          &matmul_info,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(lhs, rhs, matmul_info, act_info);

    const bool is_quantized      = is_data_type_quantized_asymmetric(lhs->data_type());
    const bool is_mmul_supported = arm_matrix_multiply_supported(CLKernelLibrary::get().get_device());

    const int k = matmul_info.adj_lhs() ? lhs->tensor_shape().y() : lhs->tensor_shape().x();

    if (is_quantized)
    {
        // MMUL kernel works only when K is a multiple of 16
        if (is_mmul_supported && !act_info.enabled() && k % 16 == 0)
        {
            return MatMulKernelType::NATIVE_MMUL_QUANTIZED;
        }

        return MatMulKernelType::NATIVE_QUANTIZED;
    }
    else
    {
        // MMUL kernel works only when K is a multiple of 4
        if (is_mmul_supported && !act_info.enabled() && k % 4 == 0)
        {
            return MatMulKernelType::NATIVE_MMUL_FP;
        }

        return MatMulKernelType::NATIVE_FP;
    }

    return is_quantized ? MatMulKernelType::NATIVE_QUANTIZED : MatMulKernelType::NATIVE_FP;
}
} // namespace
using namespace arm_compute::opencl::kernels;

ClMatMul::ClMatMul()
{
}

Status ClMatMul::validate(const ITensorInfo         *lhs,
                          const ITensorInfo         *rhs,
                          const ITensorInfo         *dst,
                          const MatMulInfo          &matmul_info,
                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rhs, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);

    const GPUTarget gpu_target = CLScheduler::get().target();

    std::unique_ptr<IClMatMulNativeKernelConfig> t = ClMatMulNativeKernelConfigurationFactory::create(gpu_target);

    const MatMulKernelInfo kernel_info = t->configure(lhs, rhs, matmul_info);

    switch (get_matmul_kernel(lhs, rhs, matmul_info, act_info))
    {
        case MatMulKernelType::NATIVE_FP:
            return ClMatMulNativeKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
        case MatMulKernelType::NATIVE_MMUL_FP:
            return ClMatMulNativeMMULKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info);
        case MatMulKernelType::NATIVE_QUANTIZED:
            return ClMatMulLowpNativeKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
        case MatMulKernelType::NATIVE_MMUL_QUANTIZED:
            return ClMatMulLowpNativeMMULKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
        default:
            ARM_COMPUTE_ERROR("Unsupported MatMul Kernel!");
    }
}

void ClMatMul::configure(const CLCompileContext    &compile_context,
                         ITensorInfo               *lhs,
                         ITensorInfo               *rhs,
                         ITensorInfo               *dst,
                         const MatMulInfo          &matmul_info,
                         const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, dst, matmul_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(lhs, rhs, dst, matmul_info));

    const GPUTarget        gpu_target    = CLScheduler::get().target();
    const auto             kernel_config = ClMatMulNativeKernelConfigurationFactory::create(gpu_target);
    const MatMulKernelInfo kernel_info   = kernel_config->configure(lhs, rhs, matmul_info);

    switch (get_matmul_kernel(lhs, rhs, matmul_info, act_info))
    {
        case MatMulKernelType::NATIVE_FP:
        {
            auto kernel = std::make_unique<ClMatMulNativeKernel>();
            kernel->set_target(gpu_target);

            kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
            _matmul_kernel = std::move(kernel);
        }
        break;
        case MatMulKernelType::NATIVE_MMUL_FP:
        {
            auto kernel = std::make_unique<ClMatMulNativeMMULKernel>();
            kernel->set_target(gpu_target);

            kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info);
            _matmul_kernel = std::move(kernel);
        }
        break;
        case MatMulKernelType::NATIVE_QUANTIZED:
        {
            auto kernel = std::make_unique<ClMatMulLowpNativeKernel>();
            kernel->set_target(gpu_target);

            kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
            _matmul_kernel = std::move(kernel);
        }
        break;
        case MatMulKernelType::NATIVE_MMUL_QUANTIZED:
        {
            auto kernel = std::make_unique<ClMatMulLowpNativeMMULKernel>();
            kernel->set_target(gpu_target);

            kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
            _matmul_kernel = std::move(kernel);
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported MatMul Kernel!");
    }
}

void ClMatMul::run(ITensorPack &tensors)
{
    CLScheduler::get().enqueue_op(*_matmul_kernel, tensors, /* flush */ true);
}

} // namespace opencl
} // namespace arm_compute
