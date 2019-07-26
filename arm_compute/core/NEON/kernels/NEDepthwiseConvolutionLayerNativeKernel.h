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
#ifndef __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H__
#define __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Interface for the kernel to run a depthwise convolution native on a tensor. */
class NEDepthwiseConvolutionLayerNativeKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEDepthwiseConvolutionLayerNativeKernel";
    }
    /** Default constructor */
    NEDepthwiseConvolutionLayerNativeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayerNativeKernel(const NEDepthwiseConvolutionLayerNativeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionLayerNativeKernel &operator=(const NEDepthwiseConvolutionLayerNativeKernel &) = delete;
    /** Default Move Constructor. */
    NEDepthwiseConvolutionLayerNativeKernel(NEDepthwiseConvolutionLayerNativeKernel &&) = default;
    /** Default move assignment operator */
    NEDepthwiseConvolutionLayerNativeKernel &operator=(NEDepthwiseConvolutionLayerNativeKernel &&) = default;
    /** Initialize the function's source, destination and parameters.
     *
     * @note Supported data layouts: NHWC
     *
     * @param[in]  input            Source tensor. DataType supported: F32.
     * @param[in]  weights          Weights tensor. This is a 3D tensor with dimensions [IFM, W, H]. Data type supported: Same as @p input.
     * @param[in]  biases           Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed. Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: Same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                   const Size2D &dilation = Size2D(1U, 1U));
    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseConvolutionLayerNativeKernel
     *
     * @note Supported data layouts: NHWC
     *
     * @param[in] input            Source tensor info. DataType supported: F32.
     * @param[in] weights          Weights tensor info. This is a 3D tensor with dimensions [IFM, W, H]. Data type supported: Same as @p input.
     * @param[in] biases           Biases tensor info. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed. Data type supported: Same as @p input.
     * @param[in] output           Destination tensor info. Data type supported: Same as @p input.
     * @param[in] conv_info        Padding and stride information to use for the convolution.
     * @param[in] depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1,
                           const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
    BorderSize border_size() const override;

private:
    template <typename T, int S, bool has_biases>
    void run_depthwise(const Window &window);

    /** Common signature for all the specialised depthwise convolution native functions
     *
     * @param[in] window Region on which to execute the kernel.
     */
    using DepthwiseFunctionPtr = void (NEDepthwiseConvolutionLayerNativeKernel::*)(const Window &window);

    DepthwiseFunctionPtr _func;
    BorderSize           _border_size;
    const ITensor       *_input;
    const ITensor       *_weights;
    const ITensor       *_biases;
    ITensor             *_output;
    PadStrideInfo        _conv_info;
    unsigned int         _depth_multiplier;
    Size2D               _dilation;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H__ */
