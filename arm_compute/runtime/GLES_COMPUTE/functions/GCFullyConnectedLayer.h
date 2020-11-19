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
#ifndef ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H
#define ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H

#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTransposeKernel.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with OpenGL ES. This function calls the following kernels:
 *
 *  -# @ref GCTransposeKernel
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
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
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    GCFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCFullyConnectedLayer(const GCFullyConnectedLayer &) = delete;
    /** Default move constructor */
    GCFullyConnectedLayer(GCFullyConnectedLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCFullyConnectedLayer &operator=(const GCFullyConnectedLayer &) = delete;
    /** Default move assignment operator */
    GCFullyConnectedLayer &operator=(GCFullyConnectedLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data type supported: F16/F32.
     * @param[in]  weights Weights tensor. The weights must be 2 dimensional. Data type supported: Same as @p input
     * @param[in]  biases  Bias tensor. It can be nullptr. Data type supported:Same as @p input.
     * @param[out] output  Destination tensor. Data type supported: Same as @p input.
     * @param[in]  fc_info (Optional) Fully connected layer additional info
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

    //Inherited methods override
    void run() override;
    void prepare() override;

private:
    void configure_fc_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output);
    void configure_conv_fc(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output);

    MemoryGroup                         _memory_group;
    IWeightsManager                    *_weights_manager;
    GCIm2ColKernel                      _im2col_kernel;
    GCFullyConnectedLayerReshapeWeights _reshape_weights_kernel;
    GCGEMMMatrixMultiplyKernel          _mm_kernel;
    GCGEMMMatrixAccumulateBiasesKernel  _accumulate_biases_kernel;
    GCTensor                            _im2col_output;
    GCTensor                            _reshape_weights_output;
    const IGCTensor                    *_original_weights;
    bool                                _are_weights_reshaped;
    bool                                _is_fc_after_conv;
    bool                                _accumulate_biases;
};
}
#endif /* ARM_COMPUTE_GCFULLYCONNECTEDLAYER_H */
