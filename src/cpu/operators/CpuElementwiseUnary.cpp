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
#include "src/cpu/operators/CpuElementwiseUnary.h"
#include "src/common/utils/Log.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/CpuElementwiseUnaryKernel.h"

namespace arm_compute
{
namespace cpu
{
using KernelType = kernels::CpuElementwiseUnaryKernel;

void CpuElementwiseUnary::configure(ElementWiseUnary op, const ITensorInfo &src, ITensorInfo &dst)
{
    ARM_COMPUTE_LOG_PARAMS(op, src, dst);
    auto k = std::make_unique<KernelType>();
    k->configure(op, src, dst);
    _kernel = std::move(k);
}

Status CpuElementwiseUnary::validate(ElementWiseUnary op, const ITensorInfo &src, const ITensorInfo &dst)
{
    return KernelType::validate(op, src, dst);
}

void CpuElementwiseUnary::run(ITensorPack &tensors)
{
    if(_kernel->is_window_configured())
    {
        ICpuOperator::run(tensors);
        return;
    }

    auto src_info = tensors.get_const_tensor(TensorType::ACL_SRC)->info();
    ICpuOperator::run(tensors, compute_output_shape_and_window(src_info->tensor_shape()).second);
}
} // namespace cpu
} // namespace arm_compute