/*
 * Copyright (c) 2017-2019 Arm Limited.
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
#ifndef ARM_COMPUTE_GCDEPTHWISECONVOLUTIONKERNEL3x3_H
#define ARM_COMPUTE_GCDEPTHWISECONVOLUTIONKERNEL3x3_H

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor.
 */
class GCDepthwiseConvolutionLayer3x3Kernel : public IGCKernel
{
public:
    /** Default constructor */
    GCDepthwiseConvolutionLayer3x3Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDepthwiseConvolutionLayer3x3Kernel(const GCDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDepthwiseConvolutionLayer3x3Kernel &operator=(const GCDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Default Move Constructor. */
    GCDepthwiseConvolutionLayer3x3Kernel(GCDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Default move assignment operator */
    GCDepthwiseConvolutionLayer3x3Kernel &operator=(GCDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input            Source tensor. DataType supported: F16.
     * @param[in]  weights          Weights tensor. A 3D tensor with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]  biases           (Optional) Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1);

    // Inherited methods overridden:
    void run(const Window &window) override;
    BorderSize border_size() const override;

private:
    BorderSize       _border_size;
    const IGCTensor *_input;
    IGCTensor       *_output;
    const IGCTensor *_weights;
    const IGCTensor *_biases;
    unsigned int     _conv_stride_x;
    unsigned int     _conv_stride_y;
    unsigned int     _conv_pad_left;
    unsigned int     _conv_pad_top;
    gles::NDRange    _lws;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_GCDEPTHWISECONVOLUTIONKERNEL3x3_H */
