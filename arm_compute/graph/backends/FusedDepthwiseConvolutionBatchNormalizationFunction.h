/*
 * Copyright (c) 2019 Arm Limited.
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

#ifndef ARM_COMPUTE_GRAPH_BACKENDS_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_FUNCTION_H
#define ARM_COMPUTE_GRAPH_BACKENDS_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_FUNCTION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Wrapper function to first apply {NE, CL}BatchNormalizationLayer on the weights and then run {NE, CL}DepthwiseConvolutionLayer with the modified weights */
template <typename TargetInfo, typename FusedLayerTypes>
class FusedDepthwiseConvolutionBatchNormalizationFunction : public IFunction
{
public:
    using TensorType         = typename TargetInfo::TensorType;
    using TensorConcreteType = typename TargetInfo::TensorConcreteType;

    FusedDepthwiseConvolutionBatchNormalizationFunction(std::shared_ptr<IMemoryManager> memory_manager = nullptr)
        : _depth_conv_layer(memory_manager), _fused_batch_norm_layer(), _fused_bias(), _is_prepared(false)
    {
    }

    /** Set the input and output tensors.
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor.  These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
     * @param[in]  bias             Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [IFM].
     *                              Data type supported: Should match @p input data type.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  mean             Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  var              Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  beta             Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
     * @param[in]  gamma            Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
     * @param[in]  epsilon          Small value to avoid division with zero. Default value is 0.001f.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  depth_multiplier Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  fused_act        Activation layer information in case of a fused activation.
     *
     */
    void configure(TensorType       *input,
                   TensorType       *weights,
                   TensorType       *bias,
                   TensorType       *output,
                   const TensorType *mean,
                   const TensorType *var,
                   const TensorType *beta,
                   const TensorType *gamma,
                   float epsilon, const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo const &fused_act)
    {
        // We don't run any validate, as we assume that the layers have been already validated
        const bool        has_bias = (bias != nullptr);
        const TensorType *bias_to_use;

        // We check if the layer has a bias. If yes, use it in-place. If not, we need to create one
        // as batch normalization might end up with a bias != 0
        if(has_bias)
        {
            _fused_batch_norm_layer.configure(weights, mean, var, nullptr, nullptr, bias, beta, gamma, epsilon, FuseBatchNormalizationType::DEPTHWISECONVOLUTION);
            bias_to_use = bias;
        }
        else
        {
            _fused_batch_norm_layer.configure(weights, mean, var, nullptr, &_fused_bias, nullptr, beta, gamma, epsilon, FuseBatchNormalizationType::DEPTHWISECONVOLUTION);
            bias_to_use = &_fused_bias;
        }

        _depth_conv_layer.configure(input, weights, bias_to_use, output, conv_info, depth_multiplier, fused_act.enabled() ? fused_act : ActivationLayerInfo());

        if(!has_bias)
        {
            _fused_bias.allocator()->allocate();
        }
    }

    // Inherited methods overridden:
    void run()
    {
        prepare();
        _depth_conv_layer.run();
    }

    void prepare()
    {
        if(!_is_prepared)
        {
            _fused_batch_norm_layer.run();
            _is_prepared = true;
        }
    }

private:
    typename FusedLayerTypes::DepthwiseConvolutionLayer _depth_conv_layer;
    typename FusedLayerTypes::FuseBatchNormalization    _fused_batch_norm_layer;
    TensorConcreteType                                  _fused_bias;
    bool                                                _is_prepared;
};
} // namespace backends
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_BACKENDS_FUSED_DEPTHWISE_CONVOLUTION_BATCH_NORMALIZATION_FUNCTION_H */
