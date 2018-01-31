/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H__
#define __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H__

#include "arm_compute/core/CL/kernels/ICLDepthwiseConvolutionLayer3x3Kernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor when the data layout is NCHW.
 */
class CLDepthwiseConvolutionLayer3x3NCHWKernel : public ICLDepthwiseConvolutionLayer3x3Kernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer3x3NCHWKernel();
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input     Source tensor. DataType supported: QASYMM8/F16/F32.
     * @param[in]  weights   Weights tensor. A 3D tensor with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]  biases    (Optional) Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                       Data type supported: Same as @p input.
     * @param[out] output    Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info Padding and stride information to use for the convolution.
     * @param[in]  act_info  (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for QASYMM8 supported.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   ActivationLayerInfo act_info) override;

    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    unsigned int _conv_stride_x;
    unsigned int _conv_pad_top;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H__ */
