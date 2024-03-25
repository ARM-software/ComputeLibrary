/*
 * Copyright (c) 2016-2022 Arm Limited.
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
#include "src/cpu/operators/CpuMul.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuMulKernel.h"

namespace arm_compute
{
namespace cpu
{
Status CpuMul::validate(const ITensorInfo         *src1,
                        const ITensorInfo         *src2,
                        const ITensorInfo         *dst,
                        float                      scale,
                        ConvertPolicy              overflow_policy,
                        RoundingPolicy             rounding_policy,
                        const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return kernels::CpuMulKernel::validate(src1, src2, dst, scale, overflow_policy, rounding_policy);
}

void CpuMul::configure(ITensorInfo               *src1,
                       ITensorInfo               *src2,
                       ITensorInfo               *dst,
                       float                      scale,
                       ConvertPolicy              overflow_policy,
                       RoundingPolicy             rounding_policy,
                       const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, scale, overflow_policy, rounding_policy, act_info);

    auto k = std::make_unique<kernels::CpuMulKernel>();
    k->configure(src1, src2, dst, scale, overflow_policy, rounding_policy);
    _kernel = std::move(k);
}

void CpuMul::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    auto split_dimension = static_cast<kernels::CpuMulKernel *>(_kernel.get())->get_split_dimension_hint();
    NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}

Status CpuComplexMul::validate(const ITensorInfo         *src1,
                               const ITensorInfo         *src2,
                               const ITensorInfo         *dst,
                               const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return kernels::CpuComplexMulKernel::validate(src1, src2, dst);
}

void CpuComplexMul::configure(ITensorInfo               *src1,
                              ITensorInfo               *src2,
                              ITensorInfo               *dst,
                              const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_LOG_PARAMS(src1, src2, dst, act_info);

    auto k = std::make_unique<kernels::CpuComplexMulKernel>();
    k->configure(src1, src2, dst);
    _kernel = std::move(k);
}

void CpuComplexMul::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
