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

    _reshape_b_and_c_only_on_first_run = b->are_values_constant() && c->are_values_constant();
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

    kernels::CpuDynamicGemmKernel *dynamic_gemm = _kernel.get();
    dynamic_gemm->prepare(tensors, _reuse_b);

    if (_reshape_b_and_c_only_on_first_run)
    {
        _reuse_b = true;
    }

    Window window           = dynamic_gemm->window();
    auto   split_dimensions = dynamic_gemm->get_split_dimension_hint();

    NEScheduler::get().schedule_op(_kernel.get(), split_dimensions, window, tensors);
}

const experimental::MemoryRequirements &CpuDynamicGemm::workspace_dynamic(const ITensorPack &tensors) const
{
    ARM_COMPUTE_ERROR_ON(tensors.empty());
    // Update memory requirements with those from the kernel.
    _aux_mem.reserve(Count + kernels::CpuDynamicGemmKernel::max_workspace_count());
    _aux_mem.resize(Count);

    for (MemoryInfo mi : _kernel->workspace(tensors))
    {
        _aux_mem.push_back(mi);
    }

    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
