/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuActivation.h"

#include "src/common/IOperator.h"
#include "src/common/utils/LegacySupport.h"
#include "src/core/cpu/kernels/CpuActivationKernel.h"
#include "src/cpu/CpuContext.h"

namespace arm_compute
{
namespace cpu
{
void CpuActivation::configure(const ITensorInfo *input, ITensorInfo *output, const ActivationLayerInfo &activation_info)
{
    auto k = std::make_unique<kernels::CpuActivationKernel>();
    k->configure(input, output, activation_info);
    _kernel = std::move(k);
}

Status CpuActivation::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &activation_info)
{
    return kernels::CpuActivationKernel::validate(input, output, activation_info);
}

std::tuple<IOperator *, StatusCode> CpuContext::create_activation(const AclTensorDescriptor &src, const AclTensorDescriptor &dst, const AclActivationDescriptor &act, bool is_validate)
{
    TensorInfo src_info = detail::convert_to_legacy_tensor_info(src);
    TensorInfo dst_info = detail::convert_to_legacy_tensor_info(dst);
    auto       info     = detail::convert_to_activation_info(act);

    if(is_validate && !bool(CpuActivation::validate(&src_info.set_is_resizable(false), &dst_info.set_is_resizable(false), info)))
    {
        return std::make_tuple(nullptr, StatusCode::UnsupportedConfig);
    }

    auto act_op = std::make_unique<cpu::CpuActivation>();
    act_op->configure(&src_info, &dst_info, info);

    auto op = new arm_compute::IOperator(static_cast<IContext *>(this));
    if(op == nullptr)
    {
        ARM_COMPUTE_LOG_ERROR_ACL("Couldn't allocate internal resources");
        return std::make_tuple(nullptr, StatusCode::OutOfMemory);
    }
    op->set_internal_operator(std::move(act_op));

    return std::make_tuple(op, StatusCode::Success);
}
} // namespace cpu
} // namespace arm_compute
