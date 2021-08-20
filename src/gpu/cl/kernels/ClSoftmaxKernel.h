/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_SOFTMAX_KERNEL_H
#define ARM_COMPUTE_CL_SOFTMAX_KERNEL_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** Interface for max, shifting, exponentiating and summing the logits */
class ClLogits1DMaxShiftExpSumKernel : public IClKernel
{
    /**< Grid size (obtained through auto-tuning) */
    static const unsigned int _grid_size;
    /**< Vector size in the serial case (obtained through auto-tuning) */
    static const unsigned int _serial_vector_size;
    /**< Vector size in the parallel case (obtained through auto-tuning, enables the best memory access pattern for Bifrost) .*/
    static const unsigned int _parallel_vector_size;

public:
    /** Info for whether a parallel reduction will be run and the vector size of the execution. */
    using ParallelReductionInfo = std::tuple<bool, unsigned int>;

    ClLogits1DMaxShiftExpSumKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClLogits1DMaxShiftExpSumKernel);
    /** Configure the kernel using the given information about tensors
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in]     src             Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in,out] max             Max values tensor. Data types supported: same as @p src
     * @param[out]    dst             Destination tensor. Data types supported: same as @p src
     * @param[out]    sum             Sum of 1D logits tensor. Data types supported: same as @p src
     * @param[in]     info            Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo &src, ITensorInfo &max, ITensorInfo &dst, ITensorInfo &sum, const SoftmaxKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClLogits1DMaxShiftExpSumKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &max, const ITensorInfo &dst, const ITensorInfo &sum);
    /** Checks if the given size is eligible for parallel reduction
     *
     * @note  Serial reduction is launched for width < (_grid_size * _serial_vector_size).
     * @note  Parallel reduction is launched for width >= (_grid_size * _serial_vector_size) and vector_size is forced to 4.
     *
     * @param[in] size Size to check
     *
     * @return A two-element tuple where the first element is a boolean specifying if a parallel reduction will be run,
     *         while the second element is the vector size of the execution.
     */
    static ParallelReductionInfo is_parallel_reduction(size_t size);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;
};

/** Interface for calculating the final step of the Softmax Layer where each logit value is multiplied by the inverse of the sum of the logits. */
class ClLogits1DNormKernel : public IClKernel
{
public:
    ClLogits1DNormKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClLogits1DNormKernel);

    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src             Source tensor. Data types supported: S32/F16/F32. If this kernel is used for log softmax, only F32/F16 is supported.
     * @param[in]  sum             Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[out] dst             Destination tensor. Data types supported: QASYMM8/QASYMM8_SIGNED for S32 @p input, or same as @p input
     * @param[in]  info            Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo &src, const ITensorInfo &sum, ITensorInfo &dst, const SoftmaxKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClLogits1DNormKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo &src, const ITensorInfo &sum, const ITensorInfo &dst, const SoftmaxKernelInfo &info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, ::cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_SOFTMAX_KERNEL_H */
