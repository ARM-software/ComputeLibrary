/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/cpu/operators/CpuAdd.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuAddKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuAdd::configure(const ITensorInfo         *src0,
                       const ITensorInfo         *src1,
                       ITensorInfo               *dst,
                       ConvertPolicy              policy,
                       const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst, policy, act_info);
    auto k = std::make_unique<kernels::CpuAddKernel>();
    k->configure(src0, src1, dst, policy);
    _kernel = std::move(k);
}

Status CpuAdd::validate(const ITensorInfo         *src0,
                        const ITensorInfo         *src1,
                        const ITensorInfo         *dst,
                        ConvertPolicy              policy,
                        const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return kernels::CpuAddKernel::validate(src0, src1, dst, policy);
}

void CpuAdd::run(ITensorPack &tensors)
{
    const auto split_dimension = static_cast<kernels::CpuAddKernel *>(_kernel.get())->get_split_dimension();

    NEScheduler::get().schedule_op(_kernel.get(), split_dimension, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
