/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
namespace
{
void configure_border_handler(const CLCompileContext &compile_context, CLFillBorderKernel &border_handler, BorderSize border_size, ICLTensor *input1, ICLTensor *input2, const ICLTensor *output)
{
    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            border_handler.configure(compile_context, broadcasted_info, border_size, BorderMode::REPLICATE);
        }
    }
}
} // namespace

void CLArithmeticAddition::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, policy, act_info);
}

void CLArithmeticAddition::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLSaturatedArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::ADD, input1, input2, output, policy, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLArithmeticAddition::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    return CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::ADD, input1, input2, output, policy, act_info);
}

void CLArithmeticSubtraction::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, policy, act_info);
}

void CLArithmeticSubtraction::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLSaturatedArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::SUB, input1, input2, output, policy, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLArithmeticSubtraction::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(policy);
    return CLSaturatedArithmeticOperationKernel::validate(ArithmeticOperation::SUB, input1, input2, output, policy, act_info);
}

void CLArithmeticDivision::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLArithmeticDivision::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::DIV, input1, input2, output, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLArithmeticDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::DIV, input1, input2, output, act_info);
}

void CLElementwiseMax::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseMax::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::MAX, input1, input2, output, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::MAX, input1, input2, output, act_info);
}

void CLElementwiseMin::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseMin::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::MIN, input1, input2, output, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::MIN, input1, input2, output, act_info);
}

void CLElementwiseSquaredDiff::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwiseSquaredDiff::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::SQUARED_DIFF, input1, input2, output, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::SQUARED_DIFF, input1, input2, output, act_info);
}

void CLElementwisePower::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLElementwisePower::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(compile_context, ArithmeticOperation::POWER, input1, input2, output, act_info);
    _kernel = std::move(k);
    configure_border_handler(compile_context, _border_handler, _kernel->border_size(), input1, input2, output);
}

Status CLElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::POWER, input1, input2, output, act_info);
}

} // namespace arm_compute
