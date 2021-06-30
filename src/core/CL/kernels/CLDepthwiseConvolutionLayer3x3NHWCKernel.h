/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNHWCKERNEL3x3_H
#define ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNHWCKERNEL3x3_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor when the data layout is NHWC.
 */
class CLDepthwiseConvolutionLayer3x3NHWCKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer3x3NHWCKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3NHWCKernel(const CLDepthwiseConvolutionLayer3x3NHWCKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3NHWCKernel &operator=(const CLDepthwiseConvolutionLayer3x3NHWCKernel &) = delete;
    /** Default Move Constructor. */
    CLDepthwiseConvolutionLayer3x3NHWCKernel(CLDepthwiseConvolutionLayer3x3NHWCKernel &&) = default;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayer3x3NHWCKernel &operator=(CLDepthwiseConvolutionLayer3x3NHWCKernel &&) = default;
    /** Default move assignment operator. */
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input            Source tensor. DataType supported: F16/F32.
     * @param[in]  weights          Weights tensor. A 3D tensor with dimensions [IFM, 3, 3].
     *                              Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU are supported.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  compile_context  The compile context to be used.
     * @param[in]  input            Source tensor. DataType supported: F16/F32.
     * @param[in]  weights          Weights tensor. A 3D tensor with dimensions [IFM, 3, 3].
     *                              Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU are supported.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer3x3NHWCKernel
     *
     * @param[in] input            Source tensor info. DataType supported: F16/F32.
     * @param[in] weights          Weights tensor info. A 3D tensor with dimensions [IFM, 3, 3].
     *                             Data type supported: Same as @p input.
     * @param[in] biases           Biases tensor info. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                             Data type supported: Same as @p input.
     * @param[in] output           Destination tensor info. Data type supported: Same as @p input.
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info         (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU are supported.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_weights;
    const ICLTensor *_biases;

    unsigned int _num_planes_processed_per_iteration;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNHWCKERNEL3x3_H */
