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
#ifndef __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor.
 */
class CLDepthwiseConvolutionKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionKernel(const CLDepthwiseConvolutionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionKernel &operator=(const CLDepthwiseConvolutionKernel &) = delete;
    /** Default Move Constructor. */
    CLDepthwiseConvolutionKernel(CLDepthwiseConvolutionKernel &&) = default;
    /** Default move assignment operator. */
    CLDepthwiseConvolutionKernel &operator=(CLDepthwiseConvolutionKernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input     Source tensor. DataType supported: F32.
     * @param[out] output    Destination tensor. DataType supported: F32.
     * @param[in]  weights   Weights tensor. These are 3D tensors with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]  conv_info Padding and stride information to use for the convolution. DataType supported: F32.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *weights, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    BorderSize       _border_size;
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_weights;
    unsigned int     _conv_stride_x;
    unsigned int     _conv_stride_y;
    unsigned int     _conv_pad_x;
    unsigned int     _conv_pad_y;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHWISECONVOLUTIONKERNEL_H__ */
