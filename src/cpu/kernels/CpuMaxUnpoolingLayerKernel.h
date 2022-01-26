/*
 * Copyright (c) 2020-2022 Arm Limited.
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
#ifndef ARM_COMPUTE_CPUMAXUNPOOLINGLAYERKERNEL_H
#define ARM_COMPUTE_CPUMAXUNPOOLINGLAYERKERNEL_H

#include "src/core/common/Macros.h"
#include "src/cpu/ICpuKernel.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
/** Interface for the pooling layer kernel */
class CpuMaxUnpoolingLayerKernel : public ICpuKernel<CpuMaxUnpoolingLayerKernel>
{
private:
    using MaxUnpoolingUKernelPtr = std::add_pointer<void(const ITensor *input, const ITensor *indices, ITensor *output, const Window &window)>::type;

public:
    /** Default constructor */
    CpuMaxUnpoolingLayerKernel() = default;
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(CpuMaxUnpoolingLayerKernel);

    /** Configure kernel for a given list of arguments
     *
     * @note Dst shape must be equal to the shape of the original src to pool.
     *
     * @param[in]  src       Source tensor to permute. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  indices   Tensor containing the offset to store the src elements in the dst tensor.
     *                       @ref CpuMaxUnpooling with indices should precede this function in order to
     *                       properly reconstruct the output tensor.
     *                       The tensor shape of this tensor has to be equal to the src tensor shape. Data type supported: U32.
     * @param[out] dst       Destination tensor. Data types supported: Same as @p src
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(const ITensorInfo *src, const ITensorInfo *indices, ITensorInfo *dst, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CpuMaxUnpoolingLayerKernel
     *
     * @param[in]  src       Source tensor to permute. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  indices   Tensor info of the indices of the maximal values. Data type supported: U32.
     * @param[out] dst       Destination tensor. Data types supported: Same as @p src
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *indices, const ITensorInfo *dst, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info) override;

    struct MaxUnpoolingKernel
    {
        const char                  *name;
        const DataTypeISASelectorPtr is_selected;
        MaxUnpoolingUKernelPtr       ukernel;
    };

    static const std::vector<MaxUnpoolingKernel> &get_available_kernels();

    const char *name() const override;

private:
    MaxUnpoolingUKernelPtr _run_method{ nullptr };
};

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
#endif /*ARM_COMPUTE_CPUMAXUNPOOLINGLAYERKERNEL_H */
