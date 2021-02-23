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
#include "src/runtime/gpu/cl/operators/ClSub.h"

#include "src/core/gpu/cl/ClCompileContext.h"
#include "src/core/gpu/cl/kernels/ClElementwiseKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClSub::configure(const ClCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst,
                      ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    auto k = std::make_unique<kernels::ClSaturatedArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::SUB, src1, src2, dst, policy, act_info);
    _kernel = std::move(k);
}

Status ClSub::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst,
                       ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return kernels::ClSaturatedArithmeticKernel::validate(ArithmeticOperation::SUB, src1, src2, dst, policy, act_info);
}
} // namespace opencl
} // namespace arm_compute
