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
#include "src/gpu/cl/operators/ClPRelu.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"

namespace arm_compute
{
namespace opencl
{
using KernelType = kernels::ClArithmeticKernel;
void ClPRelu::configure(const CLCompileContext &compile_context,
                        ITensorInfo            *input,
                        ITensorInfo            *alpha,
                        ITensorInfo            *output)
{
    ARM_COMPUTE_LOG_PARAMS(input, alpha, output);
    auto k = std::make_unique<KernelType>();
    k->configure(compile_context, ArithmeticOperation::PRELU, input, alpha, (output == nullptr ? input : output));
    _kernel = std::move(k);
}

Status ClPRelu::validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output)
{
    return KernelType::validate(ArithmeticOperation::PRELU, input, alpha, (output == nullptr ? input : output));
}

void ClPRelu::run(ITensorPack &tensors)
{
    // Output tensor can be given as nullptr for in-place computation.
    // In this case, get the input tensor and use it as the output tensor.
    if (tensors.get_tensor(TensorType::ACL_DST) == nullptr)
    {
        auto src_tensor = const_cast<ITensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
        ARM_COMPUTE_ERROR_ON_MSG(src_tensor == nullptr, "invalid source tensor is given for in-place computation");
        tensors.add_tensor(TensorType::ACL_DST, src_tensor);
    }
    IClOperator::run(tensors);
}
} // namespace opencl
} // namespace arm_compute
