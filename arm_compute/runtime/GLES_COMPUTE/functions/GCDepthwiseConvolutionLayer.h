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
#ifndef __ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__
#define __ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/IGCSimpleFunction.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute a depthwise convolution for kernel size 3x3xC. This function calls the following OpenGLES kernels:
 *
 * -# @ref GCDepthwiseConvolutionLayer3x3Kernel
 * -# @ref GCFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class GCDepthwiseConvolutionLayer3x3 : public IGCSimpleFunction
{
public:
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in, out] input     Source tensor. Data type supported: F16. (Written to only for border filling).
     * @param[in]      weights   Weights tensor. A 3D tensor with shape [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases    (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                           Data type supported: Same as @p input.
     * @param[out]     output    Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info Padding and stride information to use for the convolution.
     */
    void configure(IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info);
};
}
#endif /*__ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__ */
