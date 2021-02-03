/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEMAXUNPOOLINGLAYERKERNEL_H
#define ARM_COMPUTE_NEMAXUNPOOLINGLAYERKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the pooling layer kernel */
class NEMaxUnpoolingLayerKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEMaxUnpoolingLayerKernel";
    }
    /** Default constructor */
    NEMaxUnpoolingLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMaxUnpoolingLayerKernel(const NEMaxUnpoolingLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMaxUnpoolingLayerKernel &operator=(const NEMaxUnpoolingLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMaxUnpoolingLayerKernel(NEMaxUnpoolingLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMaxUnpoolingLayerKernel &operator=(NEMaxUnpoolingLayerKernel &&) = default;
    /** Default destructor */
    ~NEMaxUnpoolingLayerKernel() = default;
    /** Set the input and output tensors.
     *
     * @note Output shape must be equal to the shape of the original input to pool.
     *
     * @param[in]  input     Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  indices   Tensor containing the offset to store the input elements in the output tensor.
     *                       @ref cpu::kernels::CpuPoolingKernel with indices should precede this function in order to
     *                       properly reconstruct the output tensor.
     *                       The tensor shape of this tensor has to be equal to the input tensor shape. Data type supported: U32.
     * @param[out] output    Destination tensor. Data types supported: Same as @p input.
     * @param[in]  pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     */
    void configure(const ITensor *input, const ITensor *indices, ITensor *output, const PoolingLayerInfo &pool_info);
    /** Static function to check if given info will lead to a valid configuration of @ref NEMaxUnpoolingLayerKernel
     *
     * @param[in] input     Source tensor info. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output    Destination tensor info. Data types supported: Same as @p input.
     * @param[in] indices   Tensor info of the indices of the maximal values. Data type supported: U32.
     * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Function to perform 2x2 pooling and compute the pooling indices. The indices can be used for max unpool.
     *
     * @param[in] window_input Input region on which to execute the kernel.
     */
    template <typename T>
    void unpooling2(const Window &window_input);

    using UnpoolingFunction = void (NEMaxUnpoolingLayerKernel::*)(const Window &window);

private:
    UnpoolingFunction _func;
    const ITensor    *_input;
    ITensor          *_output;
    const ITensor    *_indices;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEMAXUNPOOLINGLAYERKERNEL_H */
