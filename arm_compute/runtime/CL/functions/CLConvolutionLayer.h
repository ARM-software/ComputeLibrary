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
#ifndef __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__

#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLGEMMConvolutionLayer
 * -# @ref CLWinogradConvolutionLayer
 * -# @ref CLDirectConvolutionLayer
 * -# @ref CLFFTConvolutionLayer
 */
class CLConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                              Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported:Same as @p input.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false,
                           unsigned int num_groups = 1);
    /** Static function to check if given info will return the convolution called by @ref CLConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] gpu_target       Specifies the @p GPUTarget.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     *
     * @return a status
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                    const WeightsInfo &weights_info, const ActivationLayerInfo &act_info, const GPUTarget gpu_target, const Size2D &dilation = Size2D(1U, 1U), bool enable_fast_math = false);
    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    std::shared_ptr<IMemoryManager> _memory_manager;
    std::unique_ptr<IFunction>      _function;
};
}
#endif /* __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__ */
