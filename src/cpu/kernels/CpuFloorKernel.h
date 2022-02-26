/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_FLOOR_KERNEL_H
#define ARM_COMPUTE_CPU_FLOOR_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Cpu accelarated kernel to perform a floor operation */
class CpuFloorKernel : public ICpuKernel<CpuFloorKernel>
{
private:
    using FloorKernelPtr = std::add_pointer<void(const void *, void *, int)>::type;

public:
    CpuFloorKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuFloorKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]  src Source tensor. Data type supported: F16/F32.
     * @param[out] dst Destination tensor. Same as @p src
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuFloorKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);
    /** Infer execution window
     *
     * @param[in] src Source tensor info. Data type supported: F16/F32.
     * @param[in] dst Destination tensor info. Same as @p src
     *
     * @return an execution Window
     */
    Window infer_window(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    struct FloorKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        FloorKernelPtr               ukernel;
    };

    static const std::vector<FloorKernel> &get_available_kernels();

private:
    FloorKernelPtr _run_method{ nullptr };
    std::string    _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_FLOOR_KERNEL_H */
