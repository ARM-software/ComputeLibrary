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
#ifndef ACL_SRC_CPU_KERNELS_CPURESHAPEKERNEL_H
#define ACL_SRC_CPU_KERNELS_CPURESHAPEKERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the kernel to perform tensor reshaping */
class CpuReshapeKernel : public ICpuKernel<CpuReshapeKernel>
{
public:
    CpuReshapeKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuReshapeKernel);
    /** Configure kernel for a given list of arguments
     *
     * @param[in]  src Source tensor info. Data type supported: All
     * @param[out] dst Destination tensor info. Data type supported: Same as @p input
     */
    void configure(const ITensorInfo *src, ITensorInfo *dst);

    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref CpuReshapeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst);

    // Inherited methods overridden:
    void        run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;
    const char *name() const override;

    /** Prepare the reshape kernel for execution (Only executed once) by calculating max or squashed window and selecting the _reshape_tensor_fn based on the presence of holes
     *
     * @param[in] tensors Pack of input and output tensors
     *
     */
    void prepare(ITensorPack &tensors);

    /** Return minimum workload size of the relevant kernel
     *
     * @param[in] platform     The CPU platform used to create the context.
     * @param[in] thread_count Number of threads in the execution.
     *
     * @return[out] small_network_mws          Minimum workload size for requsted configuration.
     */
    size_t get_mws(const CPUInfo &platform, size_t thread_count) const override;

    /** Get the preferred dimension in which the scheduler splits the work into multiple jobs.
      *
      * @return The split dimension.
      */
    size_t get_split_dimension() const
    {
        return _split_dimension;
    }

private:
    size_t _split_dimension{Window::DimY};

    static constexpr std::size_t _reshape_mws = 10'000;

    std::function<void(const Window &window, const ITensor *src, ITensor *dst)> _reshape_tensor_fn{};
};
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif // ACL_SRC_CPU_KERNELS_CPURESHAPEKERNEL_H
