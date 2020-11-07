/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H
#define ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/CL/ICLSimple3DKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for max, shifting, exponentiating and summing the logits */
class CLLogits1DMaxShiftExpSumKernel : public ICLKernel
{
public:
    /** Info for whether a parallel reduction will be run and the vector size of the execution. */
    using ParallelReductionInfo = std::tuple<bool, unsigned int>;

public:
    /** Default constructor */
    CLLogits1DMaxShiftExpSumKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DMaxShiftExpSumKernel(const CLLogits1DMaxShiftExpSumKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DMaxShiftExpSumKernel &operator=(const CLLogits1DMaxShiftExpSumKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLogits1DMaxShiftExpSumKernel(CLLogits1DMaxShiftExpSumKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLogits1DMaxShiftExpSumKernel &operator=(CLLogits1DMaxShiftExpSumKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]     input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in,out] max    Max values tensor. Data types supported: same as @p input
     * @param[out]    output Destination tensor. Data types supported: same as @p input
     * @param[out]    sum    Sum of 1D logits tensor. Data types supported: same as @p input
     * @param[in]     info   Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const ICLTensor *input, ICLTensor *max, ICLTensor *output, ICLTensor *sum, const SoftmaxKernelInfo &info);
    /** Set the input and output tensors.
     *
     * @param[in]     compile_context The compile context to be used.
     * @param[in]     input           Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in,out] max             Max values tensor. Data types supported: same as @p input
     * @param[out]    output          Destination tensor. Data types supported: same as @p input
     * @param[out]    sum             Sum of 1D logits tensor. Data types supported: same as @p input
     * @param[in]     info            Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *max, ICLTensor *output, ICLTensor *sum, const SoftmaxKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLLogits1DMaxShiftExpSumKernel
     *
     * @param[in] input  Source tensor. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
     * @param[in] max    Max values tensor. Data types supported: same as @p input
     * @param[in] output Destination tensor. Data types supported: same as @p input
     * @param[in] sum    Sum of 1D logits tensor. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum);
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
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_max;
    ICLTensor       *_output;
    ICLTensor       *_sum;

private:
    static const unsigned int _grid_size;
    static const unsigned int _serial_vector_size;
    static const unsigned int _parallel_vector_size;
};
/** Interface for calculating the final step of the Softmax Layer where each logit value is multiplied by the inverse of the sum of the logits. */
class CLLogits1DNormKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLLogits1DNormKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DNormKernel(const CLLogits1DNormKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DNormKernel &operator=(const CLLogits1DNormKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLogits1DNormKernel(CLLogits1DNormKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLogits1DNormKernel &operator=(CLLogits1DNormKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: S32/F16/F32. If this kernel is used for log softmax, only F32/F16 is supported.
     * @param[in]  sum    Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[out] output Destination tensor. Data types supported: QASYMM8/QASYMM8_SIGNED for S32 @p input, or same as @p input
     * @param[in]  info   Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, const SoftmaxKernelInfo &info);
    /** Set the input and output tensors.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           Source tensor. Data types supported: S32/F16/F32. If this kernel is used for log softmax, only F32/F16 is supported.
     * @param[in]  sum             Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[out] output          Destination tensor. Data types supported: QASYMM8/QASYMM8_SIGNED for S32 @p input, or same as @p input
     * @param[in]  info            Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *sum, ICLTensor *output, const SoftmaxKernelInfo &info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLLogits1DNormKernel
     *
     * @param[in] input  Source tensor. Data types supported: S32/F16/F32. If this kernel is used for log softmax, only F32/F16 is supported.
     * @param[in] sum    Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[in] output Destination tensor. Data types supported: QASYMM8 for S32 @p input, or same as @p input
     * @param[in] info   Contains information consumed by kernels for softmax described in @ref SoftmaxKernelInfo.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output, const SoftmaxKernelInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_sum;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H */
