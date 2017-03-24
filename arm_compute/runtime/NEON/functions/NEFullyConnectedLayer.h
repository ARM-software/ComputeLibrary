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
#ifndef __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__

#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"

namespace arm_compute
{
/** Basic function to compute a Fully Connected layer on NEON. This function calls the following NEON kernels:
 *  -# @ref NEConvolutionLayer (called when the weights have 4 dimensions. Pass the stride as 1 and padding as 0)
 *  -# @ref NEGEMM (called when the weights have 2 dimensions)
 *  -# @ref NETransposeKernel (called when the weights have 2 dimensions)
 *  -# @ref NEGEMMMatrixAccumulateBiasesKernel (called when the weights have 2 dimensions)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 or 4 dimensions. In particular, the weights tensor has 4 dimensions,
 *        if the fully connected layer is computed after a convolution layer otherwise the tensor has 2 dimensions if the fully connected layer
 *        is computed after another fully connected layer
 */
class NEFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in, out] input   Source tensor. Data type supported: F32. (Written to only if @ref NEGEMM needs to pad with zeros the tensor)
     * @param[in, out] weights Weights tensor. The weights can be 2 dimensional or 4 dimensional. Data type supported: Same as @p input. (Written to only if @ref NEGEMM needs to pad with zeros the tensor)
     * @param[in]      biases  Bias tensor. Data type supported:Same as @p input.
     * @param[out]     output  Destination tensor. Data type supported: Same as @p input.
     */
    void configure(ITensor *input, ITensor *weights, const ITensor *biases, ITensor *output);

    //Inherited methods override
    void run() override;

private:
    /** Run the convolution layer connect to fully connected layer case */
    void run_conv();
    /** Run the fully connected layer connect to fully connected layer case */
    void run_fc();
    /** Common signature for the functions to run */
    using FullyConnectedLayerFunctionPtr = void (NEFullyConnectedLayer::*)(void);

private:
    NEConvolutionLayer                 _conv_function;
    NEGEMM                             _gemm_function;
    NETransposeKernel                  _transpose_kernel;
    NEGEMMMatrixAccumulateBiasesKernel _acc_biases_kernel;
    FullyConnectedLayerFunctionPtr     _run_func;
    Tensor                             _weights_transposed;
    bool                               _is_first_run;
    bool                               _run_acc_biases;
};
}
#endif /* __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__ */
