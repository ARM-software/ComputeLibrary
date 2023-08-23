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
#include "src/gpu/cl/kernels/ClMatMulNativeKernel.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeDefaultConfigValhall.h"
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeKernelConfig.h"
#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelConfig.h"

using namespace arm_compute::cl_matmul;

namespace arm_compute
{
namespace opencl
{
using namespace arm_compute::opencl::kernels;

ClMatMul::ClMatMul()
    : _matmul_native_kernel(std::make_unique<ClMatMulNativeKernel>()),
      _matmul_lowp_native_kernel(std::make_unique<ClMatMulLowpNativeKernel>())
{
}

Status ClMatMul::validate(const ITensorInfo *lhs, const ITensorInfo *rhs, const ITensorInfo *dst, const MatMulInfo &matmul_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(lhs, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(rhs, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    const GPUTarget gpu_target = CLScheduler::get().target();

    std::unique_ptr<IClMatMulNativeKernelConfig> t = ClMatMulNativeKernelConfigurationFactory::create(gpu_target);

    const MatMulKernelInfo kernel_info = t->configure(lhs, rhs, matmul_info);

    const bool is_quantized = is_data_type_quantized_asymmetric(lhs->data_type());

    return is_quantized ? ClMatMulLowpNativeKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info) :
           ClMatMulNativeKernel::validate(lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
}

void ClMatMul::configure(const CLCompileContext &compile_context, ITensorInfo *lhs, ITensorInfo *rhs, ITensorInfo *dst, const MatMulInfo &matmul_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(lhs, rhs, dst);
    ARM_COMPUTE_LOG_PARAMS(lhs, rhs, dst, matmul_info);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate(lhs, rhs, dst, matmul_info));

    _is_quantized = is_data_type_quantized_asymmetric(lhs->data_type());

    const GPUTarget gpu_target = CLScheduler::get().target();

    std::unique_ptr<IClMatMulNativeKernelConfig> t = ClMatMulNativeKernelConfigurationFactory::create(gpu_target);

    MatMulKernelInfo kernel_info = t->configure(lhs, rhs, matmul_info);

    if(_is_quantized)
    {
        _matmul_lowp_native_kernel->set_target(gpu_target);

        // Configure the low-precision native matrix multiply kernel
        _matmul_lowp_native_kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
    }
    else
    {
        _matmul_native_kernel->set_target(gpu_target);

        // Configure the native matrix multiply kernel
        _matmul_native_kernel->configure(compile_context, lhs, rhs, nullptr /* bias */, dst, kernel_info, act_info);
    }
}

void ClMatMul::run(ITensorPack &tensors)
{
    if(_is_quantized)
    {
        CLScheduler::get().enqueue_op(*_matmul_lowp_native_kernel, tensors, true);
    }
    else
    {
        CLScheduler::get().enqueue_op(*_matmul_native_kernel, tensors, true);
    }
}

} // namespace opencl
} // namespace arm_compute
