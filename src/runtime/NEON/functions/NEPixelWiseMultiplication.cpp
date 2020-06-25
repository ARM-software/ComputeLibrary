/*
 * Copyright (c) 2016-2020 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEPixelWiseMultiplicationKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void NEPixelWiseMultiplication::configure(ITensor *input1, ITensor *input2, ITensor *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEPixelWiseMultiplicationKernel>();
    k->configure(input1, input2, output, scale, overflow_policy, rounding_policy);
    _kernel = std::move(k);
}
Status NEPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                           const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEPixelWiseMultiplicationKernel::validate(input1, input2, output, scale, overflow_policy, rounding_policy);
}

void NEComplexPixelWiseMultiplication::configure(ITensor *input1, ITensor *input2, ITensor *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEComplexPixelWiseMultiplicationKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

Status NEComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEComplexPixelWiseMultiplicationKernel::validate(input1, input2, output);
}

} // namespace arm_compute
