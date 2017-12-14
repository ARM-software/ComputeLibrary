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
#ifndef __ARM_COMPUTE_GCSOFTMAXLAYERKERNEL_H__
#define __ARM_COMPUTE_GCSOFTMAXLAYERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCSimple3DKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the identifying the max value of 1D Logits */
class GCLogits1DMaxKernel : public IGCSimple3DKernel
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F16/F32
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const IGCTensor *input, IGCTensor *output);
};

/** Interface for shifting the logits values around the max value and exponentiating the result */
class GCLogits1DShiftExpSumKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCLogits1DShiftExpSumKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCLogits1DShiftExpSumKernel(const GCLogits1DShiftExpSumKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCLogits1DShiftExpSumKernel &operator=(const GCLogits1DShiftExpSumKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCLogits1DShiftExpSumKernel(GCLogits1DShiftExpSumKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCLogits1DShiftExpSumKernel &operator=(GCLogits1DShiftExpSumKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F16/F32
     * @param[in]  max    Max values tensor. Data types supported: same as @p input
     * @param[out] output Destination tensor. Data types supported: same as @p input
     * @param[out] sum    Sum of 1D logits tensor. Data types supported: same as @p input
     */
    void configure(const IGCTensor *input, const IGCTensor *max, IGCTensor *output, IGCTensor *sum);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    const IGCTensor *_max;
    IGCTensor       *_output;
    IGCTensor       *_sum;
};

/** Interface for calculating the final step of the Softmax Layer where each logit value is multiplied by the inverse of the sum of the logits. */
class GCLogits1DNormKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCLogits1DNormKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCLogits1DNormKernel(const GCLogits1DNormKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCLogits1DNormKernel &operator=(const GCLogits1DNormKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCLogits1DNormKernel(GCLogits1DNormKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCLogits1DNormKernel &operator=(GCLogits1DNormKernel &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: F16/F32
     * @param[in]  sum    Sum tensor. Dimensions should be dim(input)-1. Data types supported: same as @p input
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const IGCTensor *input, const IGCTensor *sum, IGCTensor *output);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    const IGCTensor *_sum;
    IGCTensor       *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_GCSOFTMAXLAYERKERNEL_H__ */
