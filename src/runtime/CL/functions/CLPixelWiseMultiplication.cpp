/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLPixelWiseMultiplicationKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void CLPixelWiseMultiplication::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void CLPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, float scale,
                                          ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLPixelWiseMultiplicationKernel>();
    k->configure(compile_context, input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
    _kernel = std::move(k);

    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            _border_handler.configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status CLPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale,
                                           ConvertPolicy overflow_policy, RoundingPolicy rounding_policy, const ActivationLayerInfo &act_info)
{
    return CLPixelWiseMultiplicationKernel::validate(input1, input2, output, scale, overflow_policy, rounding_policy, act_info);
}

void CLComplexPixelWiseMultiplication::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input1, input2, output, act_info);
}

void CLComplexPixelWiseMultiplication::configure(const CLCompileContext &compile_context, ICLTensor *input1, ICLTensor *input2, ICLTensor *output, const ActivationLayerInfo &act_info)
{
    auto k = arm_compute::support::cpp14::make_unique<CLComplexPixelWiseMultiplicationKernel>();
    k->configure(compile_context, input1, input2, output, act_info);
    _kernel = std::move(k);

    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            _border_handler.configure(compile_context, broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status CLComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    return CLComplexPixelWiseMultiplicationKernel::validate(input1, input2, output, act_info);
}
} // namespace arm_compute
