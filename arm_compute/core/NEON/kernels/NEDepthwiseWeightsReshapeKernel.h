/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEDEPTHWISEWEIGHTSRESHAPEKERNEL_H__
#define __ARM_COMPUTE_NEDEPTHWISEWEIGHTSRESHAPEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the depthwise weights reshape kernel.
 *  This kernel reshape original weights' low 2D dimensions into a single col and
 *  have the second dimension as the original depth size.
 **/
class NEDepthwiseWeightsReshapeKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseWeightsReshapeKernel";
    }
    /** Default constructor */
    NEDepthwiseWeightsReshapeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseWeightsReshapeKernel(const NEDepthwiseWeightsReshapeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseWeightsReshapeKernel &operator=(const NEDepthwiseWeightsReshapeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDepthwiseWeightsReshapeKernel(NEDepthwiseWeightsReshapeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDepthwiseWeightsReshapeKernel &operator=(NEDepthwiseWeightsReshapeKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM].
     *                    Data type supported: QASYMM8/F16/F32.
     * @param[out] output The output tensor. Data type supported: same as @p input.
     * @param[in]  biases (Optional) The input biases to add. Shape [IFM]. Data type supported: same as @p input.
     */
    void configure(const ITensor *input, ITensor *output, const ITensor *biases);

    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseWeightsReshapeKernel
     *
     * @param[in] input  The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM].
     *                   Data type supported: QASYMM8/F16/F32.
     * @param[in] output The output tensor. Data type supported: same as @p input.
     * @param[in] biases (Optional) The input biases to add. Shape [IFM]. Data type supported: same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *biases);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    using DepthwiseWeightsReshapeFunction = void(const ITensor *input, const ITensor *bias, ITensor *output, const Window &window);

private:
    DepthwiseWeightsReshapeFunction *_func;
    const ITensor                   *_input;
    ITensor                         *_output;
    const ITensor                   *_biases;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEDEPTHWISEWEIGHTSRESHAPEKERNEL_H__ */
