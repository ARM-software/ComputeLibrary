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
#ifndef ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H
#define ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H

#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

#include "arm_compute/core/CL/kernels/CLGEMMMatrixAccumulateBiasesKernel.h"
#include "arm_compute/core/CL/kernels/CLTransposeKernel.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/CL/functions/CLFlattenLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
/** Basic function to reshape the weights of Fully Connected layer with OpenCL. This function calls the following kernels:
 *
 *  -# @ref CLTransposeKernel
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class CLFullyConnectedLayerReshapeWeights : public ICLSimpleFunction
{
public:
    /** Set the input and output tensors.
     *
     * @param[in]  input  Weights tensor. The weights must be 2 dimensional. Data types supported: QASYMM8/F16/F32.
     * @param[out] output Destination tensor which stores the transposed input tensor. Data type supported: Same as @p input.
     */
    void configure(const ICLTensor *input, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFullyConnectedLayerReshapeWeights
     *
     * @param[in] input  Weights tensor. The weights must be 2 dimensional. Data types supported: QASYMM8/F16/F32.
     * @param[in] output Destination tensor which stores the transposed input tensor. Data type supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
};

namespace weights_transformations
{
/** Basic function to manage the reshape weights generated from @ref CLFullyConnectedLayerReshapeWeights */
class CLFullyConnectedLayerReshapeWeightsManaged : public ITransformWeights
{
public:
    //Inherited method override
    void run() override
    {
        _output.allocator()->allocate();
        _func.run();
        _reshape_run = true;
    }

    //Inherited method override
    void release() override
    {
        _output.allocator()->free();
    }

    //Inherited method override
    ICLTensor *get_weights() override
    {
        return &_output;
    }

    //Inherited method override
    uint32_t uid() override
    {
        return _uid;
    }

    /** Configures the @ref CLFullyConnectedLayerReshapeWeights function
     *
     * @param[in] input Source tensor. Data type supported: QASYMM8/F16/F32.
     */
    void configure(const ICLTensor *input)
    {
        _func.configure(input, &_output);
    }

private:
    static constexpr uint32_t           _uid = 0x0;
    CLTensor                            _output{};
    CLFullyConnectedLayerReshapeWeights _func{};
};
} // namespace weights_transformations

/** Basic function to compute a Fully Connected layer on OpenCL. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLIm2ColKernel (called when the input comes from a convolutional layer)
 *  -# @ref CLFullyConnectedLayerReshapeWeights (if @p are_weights_reshaped is set to false and transpose_weights is set to true ) (called once)
 *  -# @ref CLGEMMMatrixMultiplyKernel or @ref CLGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 *  -# @ref CLGEMMMatrixAccumulateBiasesKernel or @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint (if quantized asymmetric) (if @p biases is not equal to nullptr)
 *
 * @note  The fully connected layer accepts "weights" tensors only with 2 dimensions.
 */
class CLFullyConnectedLayer : public IFunction
{
public:
    /** Constructor */
    CLFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr, IWeightsManager *weights_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFullyConnectedLayer(const CLFullyConnectedLayer &) = delete;
    /** Default move constructor */
    CLFullyConnectedLayer(CLFullyConnectedLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFullyConnectedLayer &operator=(const CLFullyConnectedLayer &) = delete;
    /** Default move assignment operator */
    CLFullyConnectedLayer &operator=(CLFullyConnectedLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input   Source tensor. Data type supported: QASYMM8/F16/F32.
     * @param[in]  weights Weights tensor. The weights must be 2 dimensional.
     *                     If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                     If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                     Data type supported: Same as @p input.
     * @param[in]  biases  Bias tensor. Can be nullptr. Data type supported:Same as @p input.
     * @param[out] output  Destination tensor. Its shape should be equal to the output of a matrix multiplication between:
     *                     - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                     - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                     Data type supported: Same as @p input.
     * @param[in]  fc_info (Optional) Fully connected layer additional info
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                   FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLFullyConnectedLayer
     *
     * @param[in]  input   Source tensor info. Data type supported: QASYMM8/F16/F32.
     * @param[in]  weights Weights tensor info. The weights must be 2 dimensional.
     *                     If this function is called after a Convolution Layer, the (transposed) weights will have as many rows as the product of the first 3 input's dimensions.
     *                     If it is called after another FullyConnected Layer, the (transposed) weights will have as many rows as the input's first dimension.
     *                     Data type supported: Same as @p input.
     * @param[in]  biases  Bias tensor info. Can be nullptr. Data type supported:Same as @p input.
     * @param[out] output  Destination tensor info. Its shape should be equal to the output of a matrix multiplication between:
     *                     - The output of im2col on the input and the (transposed) 2D weights, if the function is called after a Convolution Layer
     *                     - The input tensor and the (transposed) 2D weights, if the function is called after another FullyConnected Layer.
     *                     Data type supported: Same as @p input.
     * @param[in]  fc_info (Optional) Fully connected layer additional info
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                           FullyConnectedLayerInfo fc_info = FullyConnectedLayerInfo());

    //Inherited methods override
    void run() override;
    void prepare() override;

private:
    void configure_fc_fc(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const FullyConnectedLayerInfo &fc_info);
    void configure_conv_fc(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const FullyConnectedLayerInfo &fc_info);
    void configure_mm(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *bias, ICLTensor *output, const FullyConnectedLayerInfo &fc_info);

    MemoryGroup                                                         _memory_group;
    IWeightsManager                                                    *_weights_manager;
    CLConvertFullyConnectedWeights                                      _convert_weights;
    weights_transformations::CLConvertFullyConnectedWeightsManaged      _convert_weights_managed;
    weights_transformations::CLFullyConnectedLayerReshapeWeightsManaged _reshape_weights_managed_function;
    CLFlattenLayer                                                      _flatten_layer;
    CLFullyConnectedLayerReshapeWeights                                 _reshape_weights_function;
    CLGEMM                                                              _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore                                        _mm_gemmlowp;
    CLTensor                                                            _flatten_output;
    CLTensor                                                            _converted_weights_output;
    CLTensor                                                            _reshape_weights_output;
    bool                                                                _are_weights_converted;
    bool                                                                _are_weights_reshaped;
    bool                                                                _is_fc_after_conv;
    bool                                                                _is_quantized;
    bool                                                                _is_prepared;
    const ICLTensor                                                    *_original_weights;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLFULLYCONNECTEDLAYER_H */
