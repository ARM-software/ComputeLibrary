/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_NECONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NECONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"
#include <memory>

namespace arm_compute
{
class ITensor;

/** Basic function to simulate a convolution layer. This function calls one of the following NEON functions:
 * -# @ref NEGEMMConvolutionLayer     (executed only in case GEMM is required for the operation)
 * -# @ref NEWinogradConvolutionLayer (executed only in case Winograd is required for the operation)
 * -# @ref NEDirectConvolutionLayer   (executed only in case Direct Convolution is required for the operation)
 * -# @ref NEFFTConvolutionLayer      (executed only in case FFT is required for the operation)
 */
class NEConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

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
     * @param[in]  weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                              tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     * @param[in]  num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                             Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     * @param[in] num_groups       (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false,
                           unsigned int num_groups = 1);
    /** Static function to check if given info will return the convolution called by @ref NEConvolutionLayer
     *
     * @param[in] input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                             while every optional dimension from 4 and above represent a batch of inputs.
     *                             Data types supported: QASYMM8/F16/F32.
     * @param[in] weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                             Data types supported: Same as @p input.
     * @param[in] conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info     Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                             tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in] enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                             available which may introduce a drop of accuracy as well. Default is false
     *
     * @return the Convolution Method Hint
     */
    static ConvolutionMethod get_convolution_method(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                    const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);
    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    std::shared_ptr<IMemoryManager> _memory_manager;
    std::unique_ptr<IFunction>      _function; /**< Function to run */
};
}
#endif /* __ARM_COMPUTE_NECONVOLUTIONLAYER_H__ */
