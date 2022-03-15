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
#include "src/cpu/operators/CpuPool3d.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/Scheduler.h"
#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuPool3dKernel.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{
CpuPool3d::CpuPool3d()
    : _aux_mem(1)
{
}

CpuPool3d::~CpuPool3d() = default;

void CpuPool3d::configure(const ITensorInfo *src, ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    ARM_COMPUTE_LOG_PARAMS(src, dst, pool_info);

    // Configure pooling kernel
    auto k = std::make_unique<kernels::CpuPool3dKernel>();
    k->configure(src, dst, pool_info);
    _kernel = std::move(k);
}

Status CpuPool3d::validate(const ITensorInfo *src, const ITensorInfo *dst, const Pooling3dLayerInfo &pool_info)
{
    return kernels::CpuPool3dKernel::validate(src, dst, pool_info);
}

void CpuPool3d::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No tensors provided");

    Scheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}

experimental::MemoryRequirements CpuPool3d::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute