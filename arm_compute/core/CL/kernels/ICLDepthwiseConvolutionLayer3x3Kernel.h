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
#ifndef __ARM_COMPUTE_ICLDEPTHWISECONVOLUTIONKERNEL3x3_H__
#define __ARM_COMPUTE_ICLDEPTHWISECONVOLUTIONKERNEL3x3_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor.
 */
class ICLDepthwiseConvolutionLayer3x3Kernel : public ICLKernel
{
public:
    /** Default constructor */
    ICLDepthwiseConvolutionLayer3x3Kernel()
        : _border_size(0), _input(), _output(), _weights(), _biases(), _conv_stride_y(1)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLDepthwiseConvolutionLayer3x3Kernel(const ICLDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    ICLDepthwiseConvolutionLayer3x3Kernel &operator=(const ICLDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Default Move Constructor. */
    ICLDepthwiseConvolutionLayer3x3Kernel(ICLDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Default move assignment operator */
    ICLDepthwiseConvolutionLayer3x3Kernel &operator=(ICLDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input            Source tensor. DataType supported: QASYMM8/F16/F32.
     * @param[in]  weights          Weights tensor. A 3D tensor with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for QASYMM8 supported.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    virtual void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                           ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U)) = 0;

protected:
    BorderSize       _border_size;
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_weights;
    const ICLTensor *_biases;
    unsigned int     _conv_stride_y;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_ICLDEPTHWISECONVOLUTIONKERNEL3x3_H__ */
