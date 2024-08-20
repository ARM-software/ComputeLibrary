/*
 * Copyright (c) 2017-2024 Arm Limited.
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

#ifndef ACL_SRC_CPU_KERNELS_ACTIVATION_HEURISTICS_CPUACTIVATIONKERNELHEURISTICS_H
#define ACL_SRC_CPU_KERNELS_ACTIVATION_HEURISTICS_CPUACTIVATIONKERNELHEURISTICS_H

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/runtime/IScheduler.h"

#include "src/core/common/Macros.h"
#include "src/cpu/kernels/CpuKernelSelectionTypes.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{

class CpuActivationKernelHeuristics
{
public:
    using KernelPtr =
        std::add_pointer<void(const ITensor *, ITensor *, const ActivationLayerInfo &, const Window &)>::type;

    struct ActivationKernel
    {
        const char                                *name{nullptr};
        const ActivationDataTypeISASelectorDataPtr is_selected{nullptr};
        KernelPtr                                  ukernel{nullptr};
    };

    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuActivationKernelHeuristics);

    // Default constructor and destructor
    CpuActivationKernelHeuristics() noexcept {};
    ~CpuActivationKernelHeuristics() = default;

    /** Similar to @ref CpuActivationKernel::configure() */
    CpuActivationKernelHeuristics(const ITensorInfo         *src,
                                  const ITensorInfo         *dst,
                                  const ActivationLayerInfo &activation_info);

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
    const ActivationKernel *kernel();

    /** Return the scheduling hint e.g. dimension(s) to split
     *
     * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
     */
    const IScheduler::Hints &scheduler_hint() const;

private:
    /** Chooses a kernel to run and saves it into _kernel data member
     *
     * @param[in] selector Selector object based on input and device configuration
     */
    void choose_kernel(ActivationDataTypeISASelectorData &selector);

private:
    size_t                  _mws{ICPPKernel::default_mws};
    Window                  _window{};
    const ActivationKernel *_kernel{nullptr};
    IScheduler::Hints       _hint{Window::DimY};
};

} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_ACTIVATION_HEURISTICS_CPUACTIVATIONKERNELHEURISTICS_H
