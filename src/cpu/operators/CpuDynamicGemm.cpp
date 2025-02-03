/*
 * Copyright (c) 2024-2025 Arm Limited.
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
#include "src/cpu/operators/CpuDynamicGemm.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"

using namespace arm_compute::experimental;

namespace arm_compute
{
namespace cpu
{

void CpuDynamicGemm::configure(const ITensorInfo *a,
                               const ITensorInfo *b,
                               const ITensorInfo *c,
                               ITensorInfo       *d,
                               float              alpha,
                               float              beta,
                               const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_ERROR_THROW_ON(CpuDynamicGemm::validate(a, b, c, d, alpha, beta, gemm_info));
    ARM_COMPUTE_LOG_PARAMS(a, b, c, d, alpha, beta, gemm_info);

    _kernel = std::make_unique<kernels::CpuDynamicGemmKernel>();
    _kernel->configure(a, b, c, d, alpha, beta, Count, gemm_info);
}

Status CpuDynamicGemm::validate(const ITensorInfo *a,
                                const ITensorInfo *b,
                                const ITensorInfo *c,
                                const ITensorInfo *d,
                                float              alpha,
                                float              beta,
                                const GEMMInfo    &gemm_info)
{
    return kernels::CpuDynamicGemmKernel::validate(a, b, c, d, alpha, beta, gemm_info);
}

void CpuDynamicGemm::run(ITensorPack &tensors)
{
    ARM_COMPUTE_EXIT_ON_MSG(tensors.empty(), "No inputs provided");

    Window window = calculate_max_window(*tensors.get_const_tensor(ACL_DST)->info(), Steps());
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimX, window, tensors);
}

const experimental::MemoryRequirements &CpuDynamicGemm::workspace_dynamic(const ITensorPack &tensors) const
{
    // Update memory requirements with those from the kernel.
    _dynamic_workspace.reserve(Count + kernels::CpuDynamicGemmKernel::max_workspace_count());
    _dynamic_workspace.resize(Count);
    for (MemoryInfo mi : _kernel->workspace(tensors))
    {
        _dynamic_workspace.push_back(mi);
    }

    return _dynamic_workspace;
}

} // namespace cpu
} // namespace arm_compute
