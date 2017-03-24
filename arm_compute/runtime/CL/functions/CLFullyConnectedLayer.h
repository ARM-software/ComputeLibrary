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
#ifndef __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__

#include "arm_compute/core/CL/kernels/CLGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/CL/kernels/CLTransposeKernel.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"

namespace arm_compute
{
/** Basic function to compute a Fully Connected layer on OpenCL. This function calls the following OpenCL kernels:
 *  -# @ref CLConvolutionLayer (called when the weights have 4 dimensions. Pass the stride as 1 and padding as 0)
 *  -# @ref CLGEMM (called when the weights have 2 dimensions)
 *  -# @ref CLTransposeKernel (called when the weights have 2 dimensions)
 *  -# @ref CLGEMMMatrixAccumulateBiasesKernel (called when the weights have 2 dimensions)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 or 4 dimensions. In particular, the weights tensor has 4 dimensions,
 *        if the fully connected layer is computed after a convolution layer otherwise the tensor has 2 dimensions if the fully connected layer
 *        is computed after another fully connected layer
 */
class CLFullyConnectedLayer : public IFunction
{
public:
    /**Constructor */
    CLFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in, out] input   Source tensor. Data type supported: F16, F32. (Written to only if @ref CLGEMM needs to pad with zeros the tensor)
     * @param[in, out] weights Weights tensor. The weights can be 2 dimensional or 4 dimensional. Data type supported: Same as @p input. (Written to only if @ref CLGEMM needs to pad with zeros the tensor)
     * @param[in]      biases  Bias tensor. Data type supported:Same as @p input.
     * @param[out]     output  Destination tensor. Data type supported: Same as @p input.
     */
    void configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *biases, ICLTensor *output);

    //Inherited methods override
    void run() override;

private:
    /** Run the convolution layer connect to fully connected layer case */
    void run_conv();
    /** Run the fully connected layer connect to fully connected layer case */
    void run_fc();
    /** Common signature for the functions to run */
    using FullyConnectedLayerFunction = void (CLFullyConnectedLayer::*)(void);

private:
    CLConvolutionLayer                 _conv_function;
    CLGEMM                             _gemm_function;
    CLTransposeKernel                  _transpose_kernel;
    CLGEMMMatrixAccumulateBiasesKernel _acc_biases_kernel;
    FullyConnectedLayerFunction        _run_func;
    CLTensor                           _weights_transpose;
    bool                               _is_first_run;
    bool                               _run_acc_biases;
};
}
#endif /* __ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H__ */
