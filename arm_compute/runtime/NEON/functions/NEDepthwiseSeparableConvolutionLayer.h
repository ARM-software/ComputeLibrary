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
#ifndef __ARM_COMPUTE_NEON_DEPTHWISE_SEPARABLE_CONVOLUTION_H__
#define __ARM_COMPUTE_NEON_DEPTHWISE_SEPARABLE_CONVOLUTION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;

/** Basic function to execute depthwise convolution. This function calls the following NEON kernels and function:
 *
 * -# @ref NEDepthwiseConvolutionLayer
 * -# @ref NEDirectConvolutionLayer
 *
 * @deprecated This function is deprecated and will be removed in release 19.11
 *
 */
class NEDepthwiseSeparableConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    NEDepthwiseSeparableConvolutionLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input               Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                                 while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F32.
     * @param[in]  depthwise_weights   Depthwise convolution weights tensor. These are 3D tensors with dimensions [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
     * @param[in]  depthwise_biases    (Optional) Biases tensor.Biases are 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                                 Data type supported: Same as @p weights.
     * @param[out] depthwise_out       Depthwise destination tensor.
     * @param[in]  pointwise_weights   Pointwise convolution weights tensor. These are 4D tensors with dimensions [1, 1, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  pointwise_biases    (Optional) Biases tensor. Biases are 1D tensor with dimensions [OFM]. Must be nullptr if not needed.
     *                                 Data type supported: Same as @p weights.
     * @param[out] output              Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                                 Data types supported: Same as @p input.
     * @param[in]  depthwise_conv_info Contains padding and stride information described in @ref PadStrideInfo for depthwise convolution.
     * @param[in]  pointwise_conv_info Contains padding and stride information described in @ref PadStrideInfo for pointwise convolution.
     */
    void configure(ITensor *input, const ITensor *depthwise_weights, const ITensor *depthwise_biases, ITensor *depthwise_out,
                   const ITensor *pointwise_weights, const ITensor *pointwise_biases, ITensor *output,
                   const PadStrideInfo &depthwise_conv_info, const PadStrideInfo &pointwise_conv_info);

    // Inherited methods overriden:
    void run() override;
    void prepare() override;

private:
    NEDepthwiseConvolutionLayer _depthwise_conv;
    NEDirectConvolutionLayer    _pointwise_conv;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEON_DEPTHWISE_SEPARABLE_CONVOLUTION_H__ */
