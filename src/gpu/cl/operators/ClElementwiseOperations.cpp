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
#include "src/gpu/cl/operators/ClElementwiseOperations.h"

#include "src/common/utils/Log.h"
#include "src/gpu/cl/kernels/ClElementwiseKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClElementwiseDivision::configure(const ClCompileContext    &compile_context,
                                      ITensorInfo               *src1,
                                      ITensorInfo               *src2,
                                      ITensorInfo               *dst,
                                      const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);
    auto k = std::make_unique<kernels::ClArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::DIV, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClElementwiseDivision::validate(const ITensorInfo         *src1,
                                       const ITensorInfo         *src2,
                                       const ITensorInfo         *dst,
                                       const ActivationLayerInfo &act_info)
{
    return kernels::ClArithmeticKernel::validate(ArithmeticOperation::DIV, src1, src2, dst, act_info);
}

void ClElementwiseMax::configure(const ClCompileContext    &compile_context,
                                 ITensorInfo               *src1,
                                 ITensorInfo               *src2,
                                 ITensorInfo               *dst,
                                 const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);
    auto k = std::make_unique<kernels::ClArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::MAX, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClElementwiseMax::validate(const ITensorInfo         *src1,
                                  const ITensorInfo         *src2,
                                  const ITensorInfo         *dst,
                                  const ActivationLayerInfo &act_info)
{
    return kernels::ClArithmeticKernel::validate(ArithmeticOperation::MAX, src1, src2, dst, act_info);
}

void ClElementwiseMin::configure(const ClCompileContext    &compile_context,
                                 ITensorInfo               *src1,
                                 ITensorInfo               *src2,
                                 ITensorInfo               *dst,
                                 const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);
    auto k = std::make_unique<kernels::ClArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::MIN, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClElementwiseMin::validate(const ITensorInfo         *src1,
                                  const ITensorInfo         *src2,
                                  const ITensorInfo         *dst,
                                  const ActivationLayerInfo &act_info)
{
    return kernels::ClArithmeticKernel::validate(ArithmeticOperation::MIN, src1, src2, dst, act_info);
}

void ClElementwiseSquaredDiff::configure(const ClCompileContext    &compile_context,
                                         ITensorInfo               *src1,
                                         ITensorInfo               *src2,
                                         ITensorInfo               *dst,
                                         const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);
    auto k = std::make_unique<kernels::ClArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::SQUARED_DIFF, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClElementwiseSquaredDiff::validate(const ITensorInfo         *src1,
                                          const ITensorInfo         *src2,
                                          const ITensorInfo         *dst,
                                          const ActivationLayerInfo &act_info)
{
    return kernels::ClArithmeticKernel::validate(ArithmeticOperation::SQUARED_DIFF, src1, src2, dst, act_info);
}

void ClElementwisePower::configure(const ClCompileContext    &compile_context,
                                   ITensorInfo               *src1,
                                   ITensorInfo               *src2,
                                   ITensorInfo               *dst,
                                   const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);
    auto k = std::make_unique<kernels::ClArithmeticKernel>();
    k->configure(compile_context, ArithmeticOperation::POWER, src1, src2, dst, act_info);
    _kernel = std::move(k);
}

Status ClElementwisePower::validate(const ITensorInfo         *src1,
                                    const ITensorInfo         *src2,
                                    const ITensorInfo         *dst,
                                    const ActivationLayerInfo &act_info)
{
    return kernels::ClArithmeticKernel::validate(ArithmeticOperation::POWER, src1, src2, dst, act_info);
}
} // namespace opencl
} // namespace arm_compute
