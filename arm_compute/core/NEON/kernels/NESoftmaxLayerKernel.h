/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H
#define ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the identifying the max value of 1D Logits */
class NELogits1DMaxKernel : public INESimpleKernel
{
public:
    const char *name() const override
    {
        return "NELogits1DMaxKernel";
    }
    /** Default constructor */
    NELogits1DMaxKernel();
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor. Data types supported: same as @p input
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogits1DMaxKernel
     *
     * @param[in] input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in] output Destination tensor. Data types supported: same as @p input
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    using Logits1DMaxFunction = void(const ITensor &in, ITensor &out, const Window &window);

private:
    Logits1DMaxFunction *_func;
    BorderSize           _border_size;
};

/** Interface for softmax computation for QASYMM8 with pre-computed max. */
template <bool IS_LOG = false>
class NELogits1DSoftmaxKernel : public INEKernel
{
public:
    const char *name() const override
    {
        if(IS_LOG)
        {
            return "NELogits1DSoftmaxKernel";
        }
        else
        {
            return "NELogits1DLogSoftmaxKernel";
        }
    }
    /** Default constructor */
    NELogits1DSoftmaxKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DSoftmaxKernel(const NELogits1DSoftmaxKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NELogits1DSoftmaxKernel &operator=(const NELogits1DSoftmaxKernel &) = delete;
    /** Allow instances of this class to be moved */
    NELogits1DSoftmaxKernel(NELogits1DSoftmaxKernel &&) = default;
    /** Allow instances of this class to be moved */
    NELogits1DSoftmaxKernel &operator=(NELogits1DSoftmaxKernel &&) = default;
    /** Default destructor */
    ~NELogits1DSoftmaxKernel() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Source tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in]  max    Max values tensor. Same shape as input with dimension 0 set to 1.
     *                    Data types supported: same as @p input.
     * @param[out] output Destination tensor. Data types supported: same as @p input.
     * @param[in]  beta   A scaling factor for the exponent.
     *
     * @param      tmp    Auxiliary tensor. Must be type F32 and same shape as the input.
     */
    void configure(const ITensor *input, const ITensor *max, ITensor *output, const float beta, ITensor *tmp);
    /** Static function to check if given info will lead to a valid configuration of @ref NELogits1DSoftmaxKernel
     *
     * @param[in] input  Source tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[in] max    Max values tensor info. Same shape as input with dimension 0 set to 1.
     *                   Data types supported: same as @p input.
     * @param[in] output Destination tensor info. Data types supported: same as @p input.
     * @param[in] beta   A scaling factor for the exponent.
     * @param[in] tmp    Tensor info of auxiliary. Must be type F32 and same shape as the input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *max,
                           const ITensorInfo *output, const float beta, const ITensorInfo *tmp);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using LogitsSoftmaxFunction = void(const ITensor &in, const ITensor &max, void *const tmp, ITensor &out, const float beta,
                                       const Window &window);

    LogitsSoftmaxFunction *_func;
    const ITensor         *_input;
    const ITensor         *_max;
    ITensor               *_output;
    float                  _beta;
    ITensor               *_tmp; //Temporary. Used internally
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NESOFTMAXLAYERKERNEL_H */
