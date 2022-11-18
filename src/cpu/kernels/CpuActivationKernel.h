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
#ifndef ARM_COMPUTE_CPU_ACTIVATION_KERNEL_H
#define ARM_COMPUTE_CPU_ACTIVATION_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the activation kernel */
class CpuActivationKernel : public ICpuKernel<CpuActivationKernel>
{
private:
    using ActivationKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const ActivationLayerInfo &, const Window &)>::type;

public:
    CpuActivationKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuActivationKernel);
    /** Configure kernel for a given list of arguments
     *
     * @note If the output tensor is a nullptr, the activation function will be performed in-place
     *
     * @param[in, out] src             Source tensor info. In case of @p dst tensor = nullptr, this tensor will store the result
     *                                 of the activation function. Data types supported: QASYMM8/QASYMM8_SIGNED/QSYMM16/F16/F32.
     * @param[out]     dst             Destination tensor info. Data type supported: same as @p src
     * @param[in]      activation_info Activation layer information.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, ActivationLayerInfo activation_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuActivationKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info);

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
     *
     * @return The split dimension hint.
     */
    size_t get_split_dimension_hint() const
    {
        return _split_dimension;
    }

    struct ActivationKernel
    {
        const char                                *name;
        const ActivationDataTypeISASelectorDataPtr is_selected;
        ActivationKernelPtr                        ukernel;
    };

    static const std::vector<ActivationKernel> &get_available_kernels();

private:
    ActivationLayerInfo _act_info{};
    ActivationKernelPtr _run_method{ nullptr };
    size_t              _split_dimension{ Window::DimY };
    std::string         _name{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_ACTIVATION_KERNEL_H */
