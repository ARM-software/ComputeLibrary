/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H
#define ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H

#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/NEON/functions/NEFlattenLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with Neon. This function calls the following kernels:
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayerReshapeWeights : public INESimpleFunctionNoBorder
{
public:
    /** Constructor */
    NEFullyConnectedLayerReshapeWeights() = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayerReshapeWeights(const NEFullyConnectedLayerReshapeWeights &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayerReshapeWeights &operator=(const NEFullyConnectedLayerReshapeWeights &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFullyConnectedLayerReshapeWeights(NEFullyConnectedLayerReshapeWeights &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFullyConnectedLayerReshapeWeights &operator=(NEFullyConnectedLayerReshapeWeights &&) = delete;
    /** Default destructor */
    ~NEFullyConnectedLayerReshapeWeights() = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input  Weights tensor. The weights must be 2 dimensional. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[out] output Destination tensor. Data type supported: Same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEFullyConnectedLayerReshapeWeights
     *
     * @param[in] input  Weights tensor info. The weights must be 2 dimensional. Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] output Destination tensor info. Data type supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

namespace weights_transformations
{
/** Basic function to manage the reshape weights generated from @ref NEFullyConnectedLayerReshapeWeights */
class NEFullyConnectedLayerReshapeWeightsManaged : public ITransformWeights
{
public:
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    void release() override
    {
        _output.allocator()->free();
    }

    ITensor *get_weights() override
    {
        return &_output;
    }

    uint32_t uid() override
    {
        return _uid;
    }

    void configure(const ITensor *input)
    {
        _func.configure(input, &_output);
    }

private:
    static constexpr uint32_t           _uid = 0x0;
    Tensor                              _output{};
    NEFullyConnectedLayerReshapeWeights _func{};
};
} // namespace weights_transformations

/** Basic function to compute a Fully Connected layer on Neon. This function calls the following Neon kernels:
 *  -# @ref NEIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref NEFullyConnectedLayerReshapeWeights (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref NEGEMMMatrixMultiplyKernel or @ref NEGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 *  -# @ref NEGEMMMatrixAdditionKernel or @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint (if quantized asymmetric) (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class NEFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer(const NEFullyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEFullyConnectedLayer(NEFullyConnectedLayer &&) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFullyConnectedLayer &operator=(const NEFullyConnectedLayer &) = delete;
    /** Prevent instances of this class from being moved (As this class contains pointers) */
    NEFullyConnectedLayer &operator=(NEFullyConnectedLayer &&) = delete;
    /** Default destructor */
    ~NEFullyConnectedLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights Weights tensor. The weights must be 2 dimensional.
     *                     If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                     If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                     Data type supported: Same as @p input.
     * @param[in]  biases  Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
     * @param[out] output  Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
     *                     - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                     - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                     Data type supported: Same as @p input.
     * @param[in]  fc_info (Optional) Fully connected layer additional info
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEFullyConnectedLayer
     *
     * @param[in] input   Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights Weights tensor info. The weights must be 2 dimensional.
     *                    If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                    If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                    Data type supported: Same as @p input.
     * @param[in] biases  Bias tensor. Can be nullptr. Data type supported: Same as @p weights, S32 if @p weights is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output  Destination tensor info. Its shape should be equal to the output of a matrix multiplication between:
     *                    - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                    - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                    Data type supported: Same as @p input.
     * @param[in] fc_info (Optional) Fully connected layer additional info
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

    //Inherited methods override
    void run() override;
    void prepare() override;

private:
    void configure_fc_fc(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act);
    void configure_conv_fc(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act);
    void configure_mm(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act);

    MemoryGroup                                                         _memory_group;
    IWeightsManager                                                    *_weights_manager;
    NEFlattenLayer                                                      _flatten;
    NEConvertFullyConnectedWeights                                      _convert_weights;
    weights_transformations::NEConvertFullyConnectedWeightsManaged      _convert_weights_managed;
    NEFullyConnectedLayerReshapeWeights                                 _reshape_weights_function;
    weights_transformations::NEFullyConnectedLayerReshapeWeightsManaged _reshape_weights_managed_function;
    NEGEMM                                                              _mm_gemm;
    NEGEMMLowpMatrixMultiplyCore                                        _mm_gemmlowp;
    Tensor                                                              _flatten_output;
    Tensor                                                              _converted_weights_output;
    Tensor                                                              _reshape_weights_output;
    const ITensor                                                      *_original_weights;
    bool                                                                _are_weights_converted;
    bool                                                                _are_weights_reshaped;
    bool                                                                _is_fc_after_conv;
    bool                                                                _is_quantized_asymmetric;
    bool                                                                _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEFULLYCONNECTEDLAYER_H */
