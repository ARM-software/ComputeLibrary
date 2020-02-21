/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H
#define ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H

#include "arm_compute/core/CL/ICLKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run a MxN depthwise convolution. M and N are respectively the rows and columns of the filter
    This kernel assumes that tensor for the weights is NOT reshaped (Native version) */
class CLDepthwiseConvolutionLayerNativeKernel : public ICLKernel
{
public:
    /** Default Constructor */
    CLDepthwiseConvolutionLayerNativeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerNativeKernel(const CLDepthwiseConvolutionLayerNativeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayerNativeKernel &operator=(const CLDepthwiseConvolutionLayerNativeKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDepthwiseConvolutionLayerNativeKernel(CLDepthwiseConvolutionLayerNativeKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDepthwiseConvolutionLayerNativeKernel &operator=(CLDepthwiseConvolutionLayerNativeKernel &&) = default;
    /** Initialize the function's source, destination and parameters
     *
     * @param[in]  input              Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/FP32/FP16. Data layout supported: NHWC
     * @param[in]  weights            Weights tensor. A 3D tensor with dimensions [IFM, N, M].
     *                                Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]  biases             Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                                Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out] output             Destination tensor. Data type supported: Same as @p input.
     * @param[in]  dwc_weights_info   Depthwise convolution layer weights info to retrieve the number of output elements processed by each thread
     * @param[in]  dwc_info           Depthwise convolution layer info
     * @param[in]  conv_info          Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier   (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  dilation           (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  output_multipliers (Optional) Output multipliers tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     * @param[in]  output_shifts      (Optional) Output shifts tensor for quantized computations. In case of per-channel quantization,
     *                                the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const DWCWeightsKernelInfo &dwc_weights_info,
                   const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1, const Size2D &dilation = Size2D(1U, 1U),
                   const ICLTensor *output_multipliers = nullptr, const ICLTensor *output_shifts = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayerNativeKernel
     *
     * @param[in] input              Source tensor info. Data type supported: QASYMM8/QASYMM8_SIGNED/FP32/FP16. Data layout supported: NHWC
     * @param[in] weights            Weights tensor info. A 3D tensor with dimensions [IFM, N, M].
     *                               Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in] biases             Biases tensor info. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                               Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[in] output             Destination tensor info. Data type supported: Same as @p input.
     * @param[in] dwc_weights_info   Depthwise convolution layer weights info to retrieve the number of output elements processed by each thread
     * @param[in] dwc_info           Depthwise convolution layer info
     * @param[in] conv_info          Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier   (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] dilation           (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] output_multipliers (Optional) Output multipliers tensor for quantized computations. In case of per-channel quantization,
     *                               the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     * @param[in] output_shifts      (Optional) Output shifts tensor for quantized computations. In case of per-channel quantization,
     *                               the number of multipliers must be equal to the number of filters (IFM). Supported data types: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const DWCWeightsKernelInfo &dwc_weights_info,
                           const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1, const Size2D &dilation = Size2D(1U, 1U),
                           const ITensorInfo *output_multipliers = nullptr, const ITensorInfo *output_shifts = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    const ICLTensor *_weights;
    const ICLTensor *_biases;
    ICLTensor       *_output;
    unsigned int     _depth_multiplier;
    const ICLTensor *_output_multipliers;
    const ICLTensor *_output_shifts;
    bool             _is_quantized;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H */
