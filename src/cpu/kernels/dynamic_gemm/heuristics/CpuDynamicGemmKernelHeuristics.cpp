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

#include "src/cpu/kernels/dynamic_gemm/heuristics/CpuDynamicGemmKernelHeuristics.h"

#include "src/common/utils/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/cpu/kernels/CpuDynamicGemmKernel.h"
#include "src/cpu/kernels/dynamic_gemm/generic/impl.h"

#include <algorithm>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{

const CpuDynamicGemmKernelHeuristics::KernelList CpuDynamicGemmKernelHeuristics::fp32_kernels
{
#if defined(__aarch64__)
    {"neon_fp32_dynamic_gemm",
     [](const DataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_run),
     REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_pack_rhs),
     REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_size_of_packed_rhs),
     REGISTER_FP32_NEON(neon_fp32_dynamic_gemm_window)},
#endif /* __aarch64__ */
};

const CpuDynamicGemmKernelHeuristics::KernelMap CpuDynamicGemmKernelHeuristics::kernels{
    {DataType::F32, fp32_kernels},
};

void CpuDynamicGemmKernelHeuristics::choose_kernel(const DataTypeISASelectorData &selector)
{
    const auto &klist = kernels.find(selector.dt);
    ARM_COMPUTE_ERROR_ON(klist == kernels.end());

    for (const auto &uk : klist->second)
    {
        if (uk.is_selected(selector))
        {
            _kernel = &uk;
            return;
        }
    }
}

CpuDynamicGemmKernelHeuristics::CpuDynamicGemmKernelHeuristics(const ITensorInfo *a,
                                                               const ITensorInfo *b,
                                                               const ITensorInfo *c,
                                                               ITensorInfo       *d,
                                                               float              alpha,
                                                               float              beta,
                                                               const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);

    const DataTypeISASelectorData selector{a->data_type(), CPUInfo::get().get_isa()};
    choose_kernel(selector);
}

size_t CpuDynamicGemmKernelHeuristics::mws() const
{
    return _mws;
}

CpuDynamicGemmKernelHeuristics::KernelPtr CpuDynamicGemmKernelHeuristics::kernel() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->ukernel;
}

CpuDynamicGemmKernelHeuristics::PackRhsPtr CpuDynamicGemmKernelHeuristics::pack_rhs() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->pack_rhs;
}

CpuDynamicGemmKernelHeuristics::SizeOfPackedRhsPtr CpuDynamicGemmKernelHeuristics::size_of_packed_rhs() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->size_of_packed_rhs;
}

CpuDynamicGemmKernelHeuristics::GetWindowPtr CpuDynamicGemmKernelHeuristics::get_window() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->get_window;
}

const char *CpuDynamicGemmKernelHeuristics::name() const
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(_kernel);
    return _kernel->name;
}

const IScheduler::Hints &CpuDynamicGemmKernelHeuristics::scheduler_hint() const
{
    return _hint;
}

} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
