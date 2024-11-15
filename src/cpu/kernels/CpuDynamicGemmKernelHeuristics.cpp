/*
 * Copyright (c) 2024 Arm Limited.
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

#include "src/cpu/kernels/CpuDynamicGemmKernelHeuristics.h"

#include <map>
#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{
namespace
{

using KernelList = std::vector<CpuDynamicGemmKernelHeuristics::DynamicGemmKernel>;
using KernelMap  = std::map<DataType, KernelList>;

static const KernelMap kernels = {};

} // namespace

CpuDynamicGemmKernelHeuristics::CpuDynamicGemmKernelHeuristics(const ITensorInfo *a,
                                                               const ITensorInfo *b,
                                                               const ITensorInfo *c,
                                                               ITensorInfo       *d,
                                                               float              alpha,
                                                               float              beta,
                                                               const GEMMInfo    &gemm_info)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    ARM_COMPUTE_UNUSED(d);
    ARM_COMPUTE_UNUSED(alpha);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_UNUSED(gemm_info);
}

/** Return minimum workload size
 *
 * @return Minimum workload size for requested configuration.
 */
size_t CpuDynamicGemmKernelHeuristics::mws() const
{
    return _mws;
}

/** Return kernel's execution window
 *
 * @return The execution window
 */
const Window &CpuDynamicGemmKernelHeuristics::window() const
{
    return _window;
}

/** Return the kernel to run
 *
 * @return The function pointer to the chosen kernel
 */
const CpuDynamicGemmKernelHeuristics::DynamicGemmKernel *CpuDynamicGemmKernelHeuristics::kernel()
{
    return _kernel;
}

/** Return the scheduling hint e.g. dimension(s) to split
 *
 * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
 */
const IScheduler::Hints &CpuDynamicGemmKernelHeuristics::scheduler_hint() const
{
    return _hint;
}
} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
