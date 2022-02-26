/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPU_POOL2D_ASSEMBLY_WRAPPER_KERNEL_H
#define ARM_COMPUTE_CPU_POOL2D_ASSEMBLY_WRAPPER_KERNEL_H

#include "arm_compute/core/Types.h"
#include "src/core/NEON/kernels/assembly/pooling.hpp"
#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"
#include "src/cpu/kernels/CpuKernelSelectionTypes.h"

#include "pool_common.hpp"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** This class is a wrapper for the assembly kernels.
  *
  * Some kernels were written in assembly and highly optimised for specific
  * CPUs like A53 or A55. The arm compute library creates an instance of
  * CpuPool2dAssemblyWrapperKernel and other auxiliary data structures to
  * execute a single assembly kernel in the context of an NEFunction.
  *
  */
class CpuPool2dAssemblyWrapperKernel final : public ICpuKernel<CpuPool2dAssemblyWrapperKernel>
{
public:
    /** Constructor
     */
    CpuPool2dAssemblyWrapperKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuPool2dAssemblyWrapperKernel);

    const char *name() const override
    {
        return "CpuPool2dAssemblyWrapperKernel";
    }

    /** Initialise the kernel's src and dst.
     *
     * @param[in]  src      Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] dst      Destination tensor info to store the result of pooling. Data types supported: same as @p src.
     * @param[in]  info     Pooling meta-data.
     * @param[in]  cpu_info CPU information needed to select the most appropriate kernel.
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to CpuPool2dAssemblyWrapperKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const PoolingLayerInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    /** Get size of the workspace needed by the assembly kernel.
     *
     * @param[in] num_threads Maximum number of threads that are going to be spawned.
     *
     * @return size of workspace
     */
    size_t get_working_size(unsigned int num_threads) const;

    /** Was the asm kernel successfully configured?
     *
     * @return True if the asm kernel is configured and ready to run
     */
    bool is_configured() const;

private:
    /** Helper function to create the assembly kernel.
     *
     * @param[in] src  Source tensor info.
     * @param[in] dst  Destination tensor info.
     * @param[in] info Pooling layer meta-data.
     */
    template <typename Typesrc, typename Typedst>
    void create_arm_pooling(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info);

    /** Helper function to create the assembly kernel with requantization support
     *
     * @param[in] src  Source tensor info.
     * @param[in] dst  Destination tensor info.
     * @param[in] info Pooling layer meta-data.
     */
    template <typename Typesrc, typename Typedst>
    void create_arm_pooling_requant(const ITensorInfo *src, ITensorInfo *dst, const PoolingLayerInfo &info, const CPUInfo &cpu_info);

    std::unique_ptr<arm_conv::pooling::IPoolingCommon> _kernel_asm{ nullptr };

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /* ARM_COMPUTE_CPU_POOL2D_ASSEMBLY_WRAPPER_KERNEL_H */
