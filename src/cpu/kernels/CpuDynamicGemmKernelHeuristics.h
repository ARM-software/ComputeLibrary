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

#ifndef ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNELHEURISTICS_H
#define ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNELHEURISTICS_H

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/IScheduler.h"

#include "src/core/common/Macros.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{

class CpuDynamicGemmKernelHeuristics
{
public:
    using KernelPtr = std::add_pointer<void(
        const ITensor *, const ITensor *, const ITensor *, ITensor *, const Window &, float, float)>::type;

    struct DynamicGemmKernel
    {
        const char *name{nullptr};
        KernelPtr   ukernel{nullptr};
    };

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuDynamicGemmKernelHeuristics);

    // Default constructor and destructor
    CpuDynamicGemmKernelHeuristics() noexcept {};
    ~CpuDynamicGemmKernelHeuristics() = default;

    /** Similar to @ref CpuDynamicGemmKernel::configure() */
    CpuDynamicGemmKernelHeuristics(const ITensorInfo *a,
                                   const ITensorInfo *b,
                                   const ITensorInfo *c,
                                   ITensorInfo       *d,
                                   float              alpha,
                                   float              beta,
                                   const GEMMInfo    &gemm_info = GEMMInfo());

    /** Return minimum workload size
     *
     * @return Minimum workload size for requested configuration in size_t
     */
    size_t mws() const;

    /** Return kernel's execution window
     *
     * @return a reference to the kernel execution window of type @ref Window
     */
    const Window &window() const;

    /** Return the kernel to run
     *
     * @return The function pointer to the chosen kernel
     */
    const DynamicGemmKernel *kernel();

    /** Return the scheduling hint e.g. dimension(s) to split
     *
     * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
     */
    const IScheduler::Hints &scheduler_hint() const;

private:
    size_t                   _mws{ICPPKernel::default_mws};
    Window                   _window{};
    const DynamicGemmKernel *_kernel{nullptr};
    IScheduler::Hints        _hint{Window::DimY};
};

} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_CPUDYNAMICGEMMKERNELHEURISTICS_H
