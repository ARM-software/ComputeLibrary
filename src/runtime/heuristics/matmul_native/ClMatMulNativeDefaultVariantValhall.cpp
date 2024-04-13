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
#include "src/runtime/heuristics/matmul_native/ClMatMulNativeDefaultVariantValhall.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/TensorInfo.h"

namespace arm_compute
{
namespace cl_matmul
{
ClMatMulNativeDefaultVariantValhall::ClMatMulNativeDefaultVariantValhall(GPUTarget gpu)
    : IClMatMulNativeKernelVariant(gpu)
{
}

MatMulKernelType ClMatMulNativeDefaultVariantValhall::select_kernel(const ITensorInfo         *lhs,
                                                                    const ITensorInfo         *rhs,
                                                                    const MatMulInfo          &info,
                                                                    const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(rhs);

    using VariantFunctionExecutorPtr =
        MatMulKernelType (ClMatMulNativeDefaultVariantValhall::*)(int k, bool act_enabled);

    ClMatMulNativeVariantArray<VariantFunctionExecutorPtr> configs_G715(
        &ClMatMulNativeDefaultVariantValhall::configure_G715_float,
        &ClMatMulNativeDefaultVariantValhall::configure_G715_quantized);

    ClMatMulNativeVariantArray<VariantFunctionExecutorPtr> configs_default(
        &ClMatMulNativeDefaultVariantValhall::configure_default_float,
        &ClMatMulNativeDefaultVariantValhall::configure_default_quantized);

    VariantFunctionExecutorPtr func = nullptr;
    switch (_target)
    {
        case GPUTarget::G715:
        case GPUTarget::G615:
            func = configs_G715.get_function(lhs->data_type());
            break;
        default:
            func = configs_default.get_function(lhs->data_type());
            break;
    }

    const int  k           = info.adj_lhs() ? lhs->tensor_shape().y() : lhs->tensor_shape().x();
    const bool act_enabled = act_info.enabled();

    ARM_COMPUTE_ERROR_ON_MSG(func == nullptr, "Data type not supported for matmul native");
    return (this->*func)(k, act_enabled);
}

MatMulKernelType ClMatMulNativeDefaultVariantValhall::configure_G715_float(int k, bool act_enabled)
{
    // MMUL kernel works only when K is a multiple of 4
    if (!act_enabled && k % 4 == 0)
    {
        return MatMulKernelType::NATIVE_MMUL_FP;
    }

    return MatMulKernelType::NATIVE_FP;
}

MatMulKernelType ClMatMulNativeDefaultVariantValhall::configure_G715_quantized(int k, bool act_enabled)
{
    // MMUL kernel works only when K is a multiple of 16
    if (!act_enabled && k % 16 == 0)
    {
        return MatMulKernelType::NATIVE_MMUL_QUANTIZED;
    }

    return MatMulKernelType::NATIVE_QUANTIZED;
}

MatMulKernelType ClMatMulNativeDefaultVariantValhall::configure_default_float(int k, bool act_enabled)
{
    ARM_COMPUTE_UNUSED(k, act_enabled);

    return MatMulKernelType::NATIVE_FP;
}

MatMulKernelType ClMatMulNativeDefaultVariantValhall::configure_default_quantized(int k, bool act_enabled)
{
    ARM_COMPUTE_UNUSED(k, act_enabled);

    return MatMulKernelType::NATIVE_QUANTIZED;
}

} // namespace cl_matmul
} // namespace arm_compute
