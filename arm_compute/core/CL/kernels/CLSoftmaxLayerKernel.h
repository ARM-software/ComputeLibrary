/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H__
#define __ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H__

#include "arm_compute/core/CL/ICLSimple2DKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the identifying the max value of 1D Logits */
class CLLogits1DMaxKernel : public ICLSimple2DKernel
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, ICLTensor *output);
};

/** Interface for shifting the logits values around the max value and exponentiating the result */
class CLLogits1DShiftExpSumKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLLogits1DShiftExpSumKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DShiftExpSumKernel(const CLLogits1DShiftExpSumKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLLogits1DShiftExpSumKernel &operator=(const CLLogits1DShiftExpSumKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLLogits1DShiftExpSumKernel(CLLogits1DShiftExpSumKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLLogits1DShiftExpSumKernel &operator=(CLLogits1DShiftExpSumKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in]  max    Max values tensor. Data types supported: same as @p input
     * @param[out] output Destination tensor. Data types supported: same as @p input
     * @param[out] sum    Sum of 1D logits tensor. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const ICLTensor *max, ICLTensor *output, ICLTensor *sum);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_max;
    ICLTensor       *_output;
    ICLTensor       *_sum;
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
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in]  sum    Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const ICLTensor *input, const ICLTensor *sum, ICLTensor *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_sum;
    ICLTensor       *_output;
};
}
#endif /*__ARM_COMPUTE_CLSOFTMAXLAYERKERNEL_H__ */
