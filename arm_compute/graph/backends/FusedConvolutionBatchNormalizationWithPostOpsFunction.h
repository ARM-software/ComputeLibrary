/*
 * Copyright (c) 2021 Arm Limited.
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

#ifndef ARM_COMPUTE_GRAPH_BACKENDS_FUSED_CONVOLUTION_BATCH_NORMAZLIZATION_WITH_POST_OPS_FUNCTION_H
#define ARM_COMPUTE_GRAPH_BACKENDS_FUSED_CONVOLUTION_BATCH_NORMAZLIZATION_WITH_POST_OPS_FUNCTION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/IPostOp.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
/** Wrapper function to first apply {NE, CL}BatchNormalizationLayer on the weights and then run {NE, CL}ConvolutionLayer with the modified weights */
template <typename TargetInfo, typename FusedLayerTypes>
class FusedConvolutionBatchNormalizationWithPostOpsFunction : public IFunction
{
public:
    using TensorType         = typename TargetInfo::TensorType;
    using TensorConcreteType = typename TargetInfo::TensorConcreteType;

    FusedConvolutionBatchNormalizationWithPostOpsFunction(std::shared_ptr<IMemoryManager> memory_manager = nullptr)
        : _conv_layer(memory_manager), _fused_batch_norm_layer(), _fused_bias(), _is_prepared(false)
    {
    }

    /** Set the input and output tensors.
     *
     * @param[in]  input      Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                        while every optional dimension from 4 and above represent a batch of inputs.
     *                        Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  bias       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                        Data type supported: Should match @p input data type.
     * @param[out] output     Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                        Data types supported: Same as @p input.
     * @param[in]  mean       Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  var        Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
     * @param[in]  beta       Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
     * @param[in]  gamma      Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
     * @param[in]  epsilon    Small value to avoid division with zero. Default value is 0.001f.
     * @param[in]  conv_info  Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  num_groups Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     * @param[in]  fast_math  Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                        available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  post_ops   A sequence of post operations that are performed after the main operation.
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
                   float epsilon, const PadStrideInfo &conv_info, unsigned int num_groups, bool fast_math,
                   const arm_compute::experimental::PostOpList<TensorType *> &post_ops = experimental::PostOpList<TensorType *> {})
    {
        // We don't run any validate, as we assume that the layers have been already validated
        const bool        has_bias = (bias != nullptr);
        const TensorType *bias_to_use;

        // We check if the layer has a bias. If yes, use it in-place. If not, we need to create one
        // as batch normalization might end up with a bias != 0
        if(has_bias)
        {
            _fused_batch_norm_layer.configure(weights, mean, var, nullptr, nullptr, bias, beta, gamma, epsilon);
            bias_to_use = bias;
        }
        else
        {
            _fused_batch_norm_layer.configure(weights, mean, var, nullptr, &_fused_bias, nullptr, beta, gamma, epsilon);
            bias_to_use = &_fused_bias;
        }

        ActivationLayerInfo fused_act = ActivationLayerInfo(); // Passing an empty ActivationLayerInfo.
        _conv_layer.configure(input, weights, bias_to_use, output, conv_info, WeightsInfo(), Size2D(1U, 1U), fused_act, fast_math, num_groups, post_ops);

        if(!has_bias)
        {
            _fused_bias.allocator()->allocate();
        }
    }

    // Inherited methods overridden:
    void run()
    {
        prepare();
        _conv_layer.run();
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
    typename FusedLayerTypes::ConvolutionLayer       _conv_layer;
    typename FusedLayerTypes::FuseBatchNormalization _fused_batch_norm_layer;
    TensorConcreteType                               _fused_bias;
    bool                                             _is_prepared;
};
} // namespace backends
} // namespace graph
} // namespace arm_compute

#endif /* ARM_COMPUTE_GRAPH_BACKENDS_FUSED_CONVOLUTION_BATCH_NORMAZLIZATION_WITH_POST_OPS_FUNCTION_H */
