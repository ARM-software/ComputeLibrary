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
#ifndef __ARM_COMPUTE_NECONVOLUTIONLAYERWEIGHTSRESHAPEKERNEL_H__
#define __ARM_COMPUTE_NECONVOLUTIONLAYERWEIGHTSRESHAPEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform reshaping on the weights used by convolution layer.
 *
 * Rearranges each 3-dimensional kernel to a single row leading to a matrix with linearized kernels.
 * In combination with the @ref NEIm2ColKernel can transform a convolution to a matrix multiplication.
 *
 * For example assuming a 3D weight kernel of 3x3 dimensions and depth of 2 we have:
 * @f[
 * \left( \begin{array}{ccc}
 * a000 & a001 & a002 \\
 * a010 & a011 & a012 \\
 * a020 & a021 & a022 \\
 * \end{array} \right)
 * \left( \begin{array}{ccc}
 * a100 & a101 & a102 \\
 * a110 & a111 & a112 \\
 * a120 & a121 & a122 \\
 * \end{array} \right)
 * \rightarrow
 * \left( \begin{array}{ccccccccc}
 * a000 & a001 & a002 & a010 & a011 & a012 & a020 & a021 & a022 & a100 & a101 & a102 & a110 & a111 & a112 & a120 & a121 & a122 \\
 * \end{array} \right)
 * @f]
 */
class NEConvolutionLayerWeightsReshapeKernel : public INEKernel
{
public:
    /** Default constructor */
    NEConvolutionLayerWeightsReshapeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionLayerWeightsReshapeKernel(const NEConvolutionLayerWeightsReshapeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEConvolutionLayerWeightsReshapeKernel &operator=(const NEConvolutionLayerWeightsReshapeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEConvolutionLayerWeightsReshapeKernel(NEConvolutionLayerWeightsReshapeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEConvolutionLayerWeightsReshapeKernel &operator=(NEConvolutionLayerWeightsReshapeKernel &&) = default;
    /** Default destructor */
    ~NEConvolutionLayerWeightsReshapeKernel() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input tensor to convert. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data types supported: F32
     * @param[in]  bias   The shared bias tensor to append. Biases are 1D tensor with dimensions [OFM]. Data types supported: Same as @p input
     * @param[out] output The output tensor. Should be a 2D Tensor. Data types supported: Same as @p input
     */
    void configure(const ITensor *input, const ITensor *bias, ITensor *output);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const ITensor *_input;
    const ITensor *_bias;
    ITensor       *_output;
    bool           _has_bias;
};
}

#endif /*__ARM_COMPUTE_NECONVOLUTIONLAYERWEIGHTSRESHAPEKERNEL_H__ */
