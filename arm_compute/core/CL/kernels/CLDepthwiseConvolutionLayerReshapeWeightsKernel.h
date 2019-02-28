/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to reshape the weights of depthwise convolution. */
class CLDepthwiseConvolutionLayerReshapeWeightsKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayerReshapeWeightsKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerReshapeWeightsKernel(const CLDepthwiseConvolutionLayerReshapeWeightsKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerReshapeWeightsKernel &operator=(const CLDepthwiseConvolutionLayerReshapeWeightsKernel &) = delete;
    /** Default Move Constructor. */
    CLDepthwiseConvolutionLayerReshapeWeightsKernel(CLDepthwiseConvolutionLayerReshapeWeightsKernel &&) = default;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayerReshapeWeightsKernel &operator=(CLDepthwiseConvolutionLayerReshapeWeightsKernel &&) = default;

    /** Initialize the function's source and destination.
     *
     * @param[in]  input  The input tensor of dimension [IFM, W, H]. Data types supported: QASYMM8. Data layouts supported: NHWC
     * @param[out] output The output tensor of dimension [W*H*C0, ceil(IFM/C0)]. C0 is the number of channels read by each thread. Data types supported: same as @p weights.
     * @param[in]  info   Depthwise convolution information to reshape the input tensor.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const DepthwiseConvolutionReshapeInfo &info);

    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer3x3NHWCKernel
     *
     * @param[in] input  The input tensor info of dimension [IFM, W, H]. Data types supported: QASYMM8. Data layouts supported: NHWC
     * @param[in] output The output tensor info of dimension [W*H*C0, ceil(IFM/C0)]. C0 is the number of channels read by each thread. Data types supported: same as @p weights.
     * @param[in] info   Depthwise convolution information to reshape the input tensor.
     *
     * @return a Status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const DepthwiseConvolutionReshapeInfo &info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;

    void configure_dot_product(const DepthwiseConvolutionReshapeInfo &info);
    void configure_generic(const DepthwiseConvolutionReshapeInfo &info);
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERRESHAPEWEIGHTSKERNEL_H__ */
