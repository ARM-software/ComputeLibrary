/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H
#define ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
class CLCompileContext;
class CLDepthwiseConvolutionLayerNativeKernel;
class ICLTensor;

/** Function to execute a depthwise convolution
 *
 * -# @ref CLDepthwiseConvolutionLayerNativeKernel
 * -# @ref CLPermute (if the data layout is NCHW)
 *
 */
class CLDepthwiseConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer(const CLDepthwiseConvolutionLayer &) = delete;
    /** Default move constructor */
    CLDepthwiseConvolutionLayer(CLDepthwiseConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseConvolutionLayer &operator=(const CLDepthwiseConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLDepthwiseConvolutionLayer &operator=(CLDepthwiseConvolutionLayer &&) = default;
    /** Default destructor */
    ~CLDepthwiseConvolutionLayer();
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * Valid data layouts:
     * - NHWC
     * - NCHW
     *
     * Valid data type configurations:
     * |src0           |src1               |src2   |dst            |
     * |:--------------|:------------------|:------|:--------------|
     * |F16            |F16                |F16    |F16            |
     * |F32            |F32                |F32    |F32            |
     * |QASYMM8        |QASYMM8            |S32    |QASYMM8        |
     * |QASYMM8        |QSYMM8_PER_CHANNEL |S32    |QASYMM8        |
     * |QASYMM8_SIGNED |QASYMM8_SIGNED     |S32    |QASYMM8_SIGNED |
     * |QASYMM8_SIGNED |QSYMM8_PER_CHANNEL |S32    |QASYMM8_SIGNED |
     *
     * @param[in]      compile_context  The compile context to be used.
     * @param[in, out] input            Source tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/FP16/FP32. Data layout supported: NHWC, NCHW
     * @param[in]      weights          Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM].
     *                                  Data type supported: Same as @p input or QASYMM8/QASYMM8_SIGNED/QSYMM8_PER_CHANNEL when @p input is QASYMM8.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input, S32 when input is QASYMM8/QASYMM8_SIGNED.
     * @param[out]     output           Destination tensor. Pass in nullptr or @p input for in-place operation. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @note: For in-place support, please check @ref CLDepthwiseConvolutionLayerNativeKernel
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    /** Initialize the function's source, destination, weights and convolution information.
     *
     * Similar to @ref CLDepthwiseConvolutionLayer::configure()
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthwiseConvolutionLayer
     *
     * Similar to @ref CLDepthwiseConvolutionLayer::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           unsigned int depth_multiplier = 1, ActivationLayerInfo act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overriden:
    void run() override;
    void prepare() override;

    void set_memory_group(std::shared_ptr<IMemoryManager> memory_manager)
    {
        _memory_group = MemoryGroup(std::move(memory_manager));
    };

private:
    MemoryGroup _memory_group;

    std::unique_ptr<CLDepthwiseConvolutionLayerNativeKernel> _dwc_native_kernel;
    CLPermute                                                _permute_input_to_nhwc;
    CLPermute                                                _permute_weights_to_nhwc;
    CLPermute                                                _permute_output_to_nchw;

    CLTensor       _permuted_input;
    CLTensor       _permuted_weights;
    CLTensor       _permuted_output;
    CLTensor       _output_multipliers;
    CLTensor       _output_shifts;
    const ITensor *_original_weights;
    const ITensor *_input;
    const ITensor *_output;

    bool _needs_permute;
    bool _is_prepared;
    bool _is_quantized;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H */
