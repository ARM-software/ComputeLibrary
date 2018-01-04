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
#ifndef __ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTransposeKernel.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with OpenGL ES. This function calls the following kernels:
 *
 *  -# @ref GCTransposeKernel
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class GCFullyConnectedLayerReshapeWeights : public IGCSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input  Weights tensor. The weights must be 2 dimensional. Data types supported: F16/F32.
     * @param[out] output Destination tensor which stores the transposed input tensor. Data type supported: Same as @p input.
     */
    void configure(const IGCTensor *input, IGCTensor *output);
};

/** Basic function to compute a Fully Connected layer on OpenGL ES. This function calls the following OpenGL ES kernels:
 *
 *  -# @ref GCIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref GCFullyConnectedLayerReshapeWeights (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref GCGEMMMatrixMultiplyKernel
 *  -# @ref GCGEMMMatrixAccumulateBiasesKernel (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class GCFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    GCFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input                Source tensor. Data type supported: F16/F32.
     * @param[in]  weights              Weights tensor. The weights must be 2 dimensional. Data type supported: Same as @p input
     * @param[in]  biases               Bias tensor. It can be nullptr. Data type supported:Same as @p input.
     * @param[out] output               Destination tensor. Data type supported: Same as @p input.
     * @param[in]  transpose_weights    (Optional) Transpose weights if true. Defaults to true.
     * @param[in]  are_weights_reshaped (Optional) Reshape the weights tensor if false. Defaults to false.
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, bool transpose_weights = true, bool are_weights_reshaped = false);

    //Inherited methods override
    void run() override;

private:
    void configure_fc_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output);
    void configure_conv_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output);

    GCIm2ColKernel                      _im2col_kernel;
    GCFullyConnectedLayerReshapeWeights _reshape_weights_kernel;
    GCGEMMMatrixMultiplyKernel          _mm_kernel;
    GCGEMMMatrixAccumulateBiasesKernel  _accumulate_biases_kernel;
    GCTensor                            _im2col_output;
    GCTensor                            _reshape_weights_output;
    bool                                _are_weights_reshaped;
    bool                                _is_fc_after_conv;
    bool                                _accumulate_biases;
};
}
#endif /* __ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H__ */
