/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYERKERNEL_H__
#define __ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYERKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/OpenGLES.h"

namespace arm_compute
{
class IGCTensor;

/** Interface for the direct convolution kernel.
 */
template <unsigned int kernel_size>
class GCDirectConvolutionLayerKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCDirectConvolutionLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDirectConvolutionLayerKernel(const GCDirectConvolutionLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCDirectConvolutionLayerKernel &operator=(const GCDirectConvolutionLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCDirectConvolutionLayerKernel(GCDirectConvolutionLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCDirectConvolutionLayerKernel &operator=(GCDirectConvolutionLayerKernel &&) = default;
    /** Default destructor */
    ~GCDirectConvolutionLayerKernel() = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input     The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F16/F32
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  bias      Biases tensor. Shared bias supported. Biases are 1D tensor with dimensions [OFM]. Data type supported:Same as @p input.
     * @param[out] output    The output tensor. First 2 lower dimensions represent a transform of each 3D input,
     *                       while every dimension above represents a batch. Data types supported: Same as @p input
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *bias, IGCTensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    BorderSize border_size() const override;

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    const IGCTensor *_bias;
    const IGCTensor *_weights;
    IGCTensor       *_output;
    BorderSize       _border_size;
    int              _conv_stride_x;
    int              _conv_stride_y;
    int              _conv_pad_x;
    int              _conv_pad_y;
    gles::NDRange    _lws;
};

using GCDirectConvolutionLayer1x1Kernel = GCDirectConvolutionLayerKernel<1>;
using GCDirectConvolutionLayer3x3Kernel = GCDirectConvolutionLayerKernel<3>;
using GCDirectConvolutionLayer5x5Kernel = GCDirectConvolutionLayerKernel<5>;
}
#endif /*__ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYERKERNEL_H__ */
