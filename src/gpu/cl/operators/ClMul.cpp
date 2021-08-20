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
#include "src/gpu/cl/operators/ClMul.h"

#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClMulKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClMul::configure(const CLCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float scale,
                      ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    auto k = std::make_unique<kernels::ClMulKernel>();
    k->configure(compile_context, src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
    _kernel = std::move(k);
}

Status ClMul::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float scale,
                       ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    return kernels::ClMulKernel::validate(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);
}

void ClComplexMul::configure(const CLCompileContext &compile_context, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    auto k = std::make_unique<kernels::ClComplexMulKernel>();
    k->configure(compile_context, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClComplexMul::validate(const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    return kernels::ClComplexMulKernel::validate(src1, src2, dst, act_info);
}
} // namespace opencl
} // namespace arm_compute