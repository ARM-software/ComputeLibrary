/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/gpu/cl/operators/ClPool3d.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/kernels/ClPool3dKernel.h"

#include "src/common/utils/Log.h"

namespace arm_compute
{
namespace opencl
{
void ClPool3d::configure(const ClCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);
    ARM_COMPUTE_LOG_PARAMS(src, dst, info);

    // Configure pooling kernel
    auto k = std::make_unique<kernels::ClPool3dKernel>();
    k->set_target(CLScheduler::get().target());
    k->configure(compile_context, src, dst, info);
    _kernel = std::move(k);

    // Tune kernels
    CLScheduler::get().tune_kernel_static(*_kernel);
}

Status ClPool3d::validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &info)
{
    return kernels::ClPool3dKernel::validate(src, dst, info);
}
} // namespace opencl
} // namespace arm_compute
