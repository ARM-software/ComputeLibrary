/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#include "src/runtime/cpu/operators/CpuPixelWiseMultiplication.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/core/cpu/kernels/CpuPixelWiseMultiplicationKernel.h"

namespace arm_compute
{
namespace cpu
{
Status CpuPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                            const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return kernels::CpuPixelWiseMultiplicationKernel::validate(input1, input2, output, scale, overflow_policy, rounding_policy);
}

void CpuPixelWiseMultiplication::configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, float scale, ConvertPolicy overflow_policy, RoundingPolicy rounding_policy,
                                           const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = std::make_unique<kernels::CpuPixelWiseMultiplicationKernel>();
    k->configure(input1, input2, output, scale, overflow_policy, rounding_policy);
    _kernel = std::move(k);
}

void CpuPixelWiseMultiplication::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}

Status CpuComplexPixelWiseMultiplication::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return kernels::CpuComplexPixelWiseMultiplicationKernel::validate(input1, input2, output);
}

void CpuComplexPixelWiseMultiplication::configure(ITensorInfo *input1, ITensorInfo *input2, ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = std::make_unique<kernels::CpuComplexPixelWiseMultiplicationKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

void CpuComplexPixelWiseMultiplication::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute