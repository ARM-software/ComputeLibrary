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
#ifndef __ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H__
#define __ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the identifying the max value of 1D Logits */
class NELogits1DMaxKernel : public INESimpleKernel
{
public:
    /** Default constructor */
    NELogits1DMaxKernel();
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogits1DMaxKernel
     *
     * @param[in] input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in] output Destination tensor. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    using Logits1DMaxFunction = void(const ITensor *in, ITensor *out, const Window &window);

private:
    Logits1DMaxFunction *_func;
    BorderSize           _border_size;
};

/** Interface for shifting the logits values around the max value and exponentiating the result */
class NELogits1DShiftExpSumKernel : public INEKernel
{
public:
    /** Default constructor */
    NELogits1DShiftExpSumKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DShiftExpSumKernel(const NELogits1DShiftExpSumKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DShiftExpSumKernel &operator=(const NELogits1DShiftExpSumKernel &) = delete;
    /** Allow instances of this class to be moved */
    NELogits1DShiftExpSumKernel(NELogits1DShiftExpSumKernel &&) = default;
    /** Allow instances of this class to be moved */
    NELogits1DShiftExpSumKernel &operator=(NELogits1DShiftExpSumKernel &&) = default;
    /** Default destructor */
    ~NELogits1DShiftExpSumKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[in]  max    Max values tensor. Data types supported: same as @p input.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[out] sum    Sum of 1D logits tensor. Data types supported: same as @p input.
     * @param[in]  beta   (Optional) A scaling factor for the exponent. QS8/QS16 only support a beta value of 1.
     */
    void configure(const ITensor *input, const ITensor *max, ITensor *output, ITensor *sum, float beta = 1.0f);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogits1DShiftExpSumKernel
     *
     * @param[in] input  Source tensor. Data types supported: QS8/QS16/F16/F32
     * @param[in] max    Max values tensor. Data types supported: same as @p input
     * @param[in] output Destination tensor. Data types supported: same as @p input.
     * @param[in] sum    Sum of 1D logits tensor. Data types supported: same as @p input.
     * @param[in] beta   (Optional) A scaling factor for the exponent. QS8/QS16 only support a beta value of 1.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *max, const ITensorInfo *output, const ITensorInfo *sum, float beta = 1.0f);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using Logits1DShiftExpSumFunction = void(const ITensor *in, const ITensor *max, ITensor *out, ITensor *sum, const Window &window, float beta);

private:
    Logits1DShiftExpSumFunction *_func;
    const ITensor               *_input;
    const ITensor               *_max;
    ITensor                     *_output;
    ITensor                     *_sum;
    float                        _beta;
};

/** Interface for calculating the final step of the Softmax Layer where each logit value is multiplied by the inverse of the sum of the logits. */
class NELogits1DNormKernel : public INEKernel
{
public:
    /** Default constructor */
    NELogits1DNormKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DNormKernel(const NELogits1DNormKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DNormKernel &operator=(const NELogits1DNormKernel &) = delete;
    /** Allow instances of this class to be moved */
    NELogits1DNormKernel(NELogits1DNormKernel &&) = default;
    /** Allow instances of this class to be moved */
    NELogits1DNormKernel &operator=(NELogits1DNormKernel &&) = default;
    /** Default destructor */
    ~NELogits1DNormKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QS8/QS16/F16/F32.
     * @param[in]  sum    Sum tensor. The number of dimensions should be dim(input)-1. Data types supported: same as @p input.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     */
    void configure(const ITensor *input, const ITensor *sum, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogits1DNormKernel
     *
     * @param[in] input  Source tensor. Data types supported: QS8/QS16/S32/F16/F32
     * @param[in] sum    Sum tensor. The number of dimensions should be dim(input)-1. Data types supported: same as @p input.
     * @param[in] output Destination tensor. Data types supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *sum, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using Logits1DNormFunction = void(const ITensor *in, const ITensor *sum, ITensor *out, const Window &window);

private:
    Logits1DNormFunction *_func;
    const ITensor        *_input;
    const ITensor        *_sum;
    ITensor              *_output;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H__ */
