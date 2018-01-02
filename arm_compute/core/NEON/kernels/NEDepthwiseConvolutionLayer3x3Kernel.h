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
#ifndef __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__
#define __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor.
 */
class NEDepthwiseConvolutionLayer3x3Kernel : public INEKernel
{
public:
    /** Default constructor */
    NEDepthwiseConvolutionLayer3x3Kernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayer3x3Kernel(const NEDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayer3x3Kernel &operator=(const NEDepthwiseConvolutionLayer3x3Kernel &) = delete;
    /** Default Move Constructor. */
    NEDepthwiseConvolutionLayer3x3Kernel(NEDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Default move assignment operator. */
    NEDepthwiseConvolutionLayer3x3Kernel &operator=(NEDepthwiseConvolutionLayer3x3Kernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input     Source tensor. DataType supported: F32.
     * @param[in]  weights   Weights tensor. This is a 3D tensor with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[out] output    Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info Padding and stride information to use for the convolution.
     */
    void configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    BorderSize     _border_size;
    const ITensor *_input;
    ITensor       *_output;
    const ITensor *_weights;
    PadStrideInfo  _conv_info;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONKERNEL3x3_H__ */