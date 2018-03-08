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
#ifndef __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__
#define __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NETransposeKernel.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with NEON. This function calls the following kernels:
 *
 *  -# @ref NETransposeKernel        (if @p transpose_weights is set to true)
 *  -# @ref NEGEMMTranspose1xWKernel (if @p is_batched_fc_layer is set to true)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input               Weights tensor. The weights must be 2 dimensional. Data types supported: QS8/QS16/F32.
     * @param[out] output              Destination tensor. Data type supported: Same as @p input.
     * @param[in]  transpose_weights   True if the weights must be transposed. Data types supported: Same as @p weights.
     * @param[in]  is_batched_fc_layer True if it is a batched fully connected layer
     */
    void configure(const ITensor *input, ITensor *output, bool transpose_weights, bool is_batched_fc_layer);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFullyConnectedLayerReshapeWeights
     *
     * @param[in] input               Weights tensor info. The weights must be 2 dimensional. Data types supported: QS8/QS16/F32.
     * @param[in] output              Destination tensor info. Data type supported: Same as @p input.
     * @param[in] transpose_weights   True if the weights must be transposed. Data types supported: Same as @p weights.
     * @param[in] is_batched_fc_layer True if it is a batched fully connected layer
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, bool transpose_weights, bool is_batched_fc_layer);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup              _memory_group;
    NETransposeKernel        _transpose_kernel;
    NEGEMMTranspose1xWKernel _transpose1xW_kernel;
    Tensor                   _transpose_output;
    bool                     _transpose_weights;
    bool                     _is_batched_fc_layer;
};

/** Basic function to compute a Fully Connected layer on NEON. This function calls the following NEON kernels:
 *  -# @ref NEIm2ColKernel                      (called when the input comes from a convolutional layer)
 *  -# @ref NEFullyConnectedLayerReshapeWeights (if @p are_weights_reshaped flag is set to false) (called once)
 *  -# @ref NEGEMMInterleave4x4Kernel (called if we have a multi-batch input)
 *  -# @ref NEGEMMMatrixMultiplyKernel
 *  -# @ref NEGEMMMatrixAccumulateBiasesKernel (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer(const NEFullyConnectedLayer &) = delete;
    /** Default move constructor */
    NEFullyConnectedLayer(NEFullyConnectedLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer &operator=(const NEFullyConnectedLayer &) = delete;
    /** Default move assignment operator */
    NEFullyConnectedLayer &operator=(NEFullyConnectedLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input                Source tensor. Data type supported: QS8/QS16/F32.
     * @param[in]  weights              Weights tensor. The weights must be 2 dimensional. Data type supported: Same as @p input.
     * @param[in]  biases               Bias tensor. Can be nullptr. Data type supported:Same as @p input.
     * @param[out] output               Destination tensor. Data type supported: Same as @p input.
     * @param[in]  transpose_weights    (Optional) Transpose the weights tensor if true. Defaults to true.
     * @param[in]  are_weights_reshaped (Optional) Reshape the weights tensor if false. Defaults to false.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose_weights = true, bool are_weights_reshaped = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFullyConnectedLayer
     *
     * @param[in] input                Source tensor info. Data type supported: QS8/QS16/F16/F32.
     * @param[in] weights              Weights tensor info. The weights must be 2 dimensional. Data type supported: Same as @p input
     * @param[in] biases               Bias tensor info. It can be nullptr. Data type supported:Same as @p input.
     * @param[in] output               Destination tensor info. Data type supported: Same as @p input.
     * @param[in] transpose_weights    (Optional) Transpose weights if true. Defaults to true.
     * @param[in] are_weights_reshaped (Optional) Reshape the weights tensor if false. Defaults to false.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, bool transpose_weights = true, bool are_weights_reshaped = false);

    //Inherited methods override
    void run() override;

private:
    MemoryGroup                         _memory_group;
    NEIm2ColKernel                      _im2col_kernel;
    NEFullyConnectedLayerReshapeWeights _reshape_weights_kernel;
    NEGEMMInterleave4x4Kernel           _interleave4x4_kernel;
    NEGEMMMatrixMultiplyKernel          _mm_kernel;
    NEGEMMMatrixAccumulateBiasesKernel  _accumulate_biases_kernel;
    Tensor                              _im2col_output;
    Tensor                              _interleave4x4_output;
    Tensor                              _reshape_weights_output;
    bool                                _are_weights_reshaped;
    bool                                _is_batched_fc_layer;
    bool                                _linearize_input;
    bool                                _accumulate_biases;
    const ITensor                      *_original_weights;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H__ */
