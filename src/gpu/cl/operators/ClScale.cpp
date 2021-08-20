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
#include "src/gpu/cl/operators/ClScale.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClScaleKernel.h"

namespace arm_compute
{
namespace opencl
{
void ClScale::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const ScaleKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    // Configure Scale kernel
    auto k = std::make_unique<kernels::ClScaleKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, src, dst, info);
    _kernel = std::move(k);

    // Tune kernel
    CLScheduler::get().tune_kernel_static(*_kernel);
}

Status ClScale::validate(const ITensorInfo *src, const ITensorInfo *dst, const ScaleKernelInfo &info)
{
    return kernels::ClScaleKernel::validate(src, dst, info);
}

void ClScale::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    CLScheduler::get().enqueue_op(*_kernel.get(), tensors);
}
} // namespace opencl
} // namespace arm_compute