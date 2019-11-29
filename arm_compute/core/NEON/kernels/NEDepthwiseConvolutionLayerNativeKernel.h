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
#ifndef ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H
#define ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H

#include "arm_compute/core/NEON/INEKernel.h"

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <arm_neon.h>
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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
     * @param[in]  input            Source tensor. DataType supported: QASYMM8/F16/F32.
     * @param[in]  weights          Weights tensor. This is a 3D tensor with dimensions [IFM, W, H].
     *                              Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]  biases           Biases tensor. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input, S32 when input is QASYMM8.
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
     * @param[in] input            Source tensor info. DataType supported: QASYMM8/F16/F32.
     * @param[in] weights          Weights tensor info. This is a 3D tensor with dimensions [IFM, W, H].
     *                             Data type supported: Same as @p input or QASYMM8/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in] biases           Biases tensor info. A 1D tensor with dimensions [IFM]. Must be nullptr if not needed.
     *                             Data type supported: Same as @p input, S32 when input is QASYMM8.
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
    template < typename T, typename TW, int S, bool has_biases, bool is_per_channel, typename std::enable_if < std::is_same<T, float>::value
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                                                               || std::is_same<T, float16_t>::value
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                                                                                               ,
                                                                                                               int >::type = 0 >
    void run_depthwise(const Window &window);

    template <typename T, typename TW, int S, bool has_biases, bool is_per_channel, typename std::enable_if<std::is_same<T, uint8_t>::value, int>::type = 0>
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
    std::vector<int>     _output_multiplier;
    std::vector<int>     _output_shift;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_NEDEPTHWISECONVOLUTIONLAYERNATIVEKERNEL_H */
