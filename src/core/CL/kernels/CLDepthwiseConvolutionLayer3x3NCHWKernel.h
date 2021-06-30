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
#ifndef ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H
#define ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H

#include "src/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a 3x3 depthwise convolution on a tensor when the data layout is NCHW.
 */
class CLDepthwiseConvolutionLayer3x3NCHWKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer3x3NCHWKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3NCHWKernel(const CLDepthwiseConvolutionLayer3x3NCHWKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer3x3NCHWKernel &operator=(const CLDepthwiseConvolutionLayer3x3NCHWKernel &) = delete;
    /** Default Move Constructor. */
    CLDepthwiseConvolutionLayer3x3NCHWKernel(CLDepthwiseConvolutionLayer3x3NCHWKernel &&) = default;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayer3x3NCHWKernel &operator=(CLDepthwiseConvolutionLayer3x3NCHWKernel &&) = default;
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input              Source tensor. DataType supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights            Weights tensor. A 3D tensor with dimensions [3, 3, IFM].
     *                                Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases             Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                                Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out] output             Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info          Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier   (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info           (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for QASYMM8 supported.
     * @param[in]  dilation           (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  output_multipliers (Optional) Output multipliers tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     * @param[in]  output_shifts      (Optional) Output shifts tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U),
                   const ICLTensor *output_multipliers = nullptr, const ICLTensor *output_shifts = nullptr);
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  compile_context    The compile context to be used.
     * @param[in]  input              Source tensor. DataType supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in]  weights            Weights tensor. A 3D tensor with dimensions [3, 3, IFM].
     *                                Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in]  biases             Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                                Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out] output             Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info          Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier   (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info           (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU for QASYMM8 supported.
     * @param[in]  dilation           (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  output_multipliers (Optional) Output multipliers tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     * @param[in]  output_shifts      (Optional) Output shifts tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U),
                   const ICLTensor *output_multipliers = nullptr, const ICLTensor *output_shifts = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer3x3NCHWKernel
     *
     * @param[in] input              Source tensor info. DataType supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
     * @param[in] weights            Weights tensor info. A 3D tensor with dimensions [3, 3, IFM].
     *                               Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] biases             Biases tensor info. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                               Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output             Destination tensor. Data type supported: Same as @p input.
     * @param[in] conv_info          Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier   (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] act_info           (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU are supported.
     * @param[in] dilation           (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] output_multipliers (Optional) Output multipliers tensor info for quantized computations. In case of per-channel quantization,
     *                               the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     * @param[in] output_shifts      (Optional) Output shifts tensor for quantized computations. In case of per-channel quantization,
     *                               the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(),
                           const Size2D &dilation = Size2D(1U, 1U), const ITensorInfo *output_multipliers = nullptr, const ITensorInfo *output_shifts = nullptr);

    void run(const Window &window, cl::CommandQueue &queue) override;
    BorderSize border_size() const override;

private:
    BorderSize       _border_size;
    const ICLTensor *_input;
    ICLTensor       *_output;
    const ICLTensor *_weights;
    const ICLTensor *_biases;
    unsigned int     _conv_stride_y;
    const ICLTensor *_output_multipliers;
    const ICLTensor *_output_shifts;
    bool             _is_quantized;

    unsigned int _conv_stride_x;
    unsigned int _conv_pad_top;
    unsigned int _conv_pad_left;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHWISECONVOLUTIONNCHWKERNEL3x3_H */
