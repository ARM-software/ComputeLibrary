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
#ifndef __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONASSEMBLYDISPATCH_H__
#define __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONASSEMBLYDISPATCH_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
/** Depthwise convolution assembly kernel glue */
class NEDepthwiseConvolutionAssemblyDispatch : public IFunction
{
public:
    /** Default constructor
     *
     * @param[in,out] memory_manager Memory manager to use
     */
    NEDepthwiseConvolutionAssemblyDispatch(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionAssemblyDispatch(const NEDepthwiseConvolutionAssemblyDispatch &) = delete;
    /** Default move constructor */
    NEDepthwiseConvolutionAssemblyDispatch(NEDepthwiseConvolutionAssemblyDispatch &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDepthwiseConvolutionAssemblyDispatch &operator=(const NEDepthwiseConvolutionAssemblyDispatch &) = delete;
    /** Default move assignment operator */
    NEDepthwiseConvolutionAssemblyDispatch &operator=(NEDepthwiseConvolutionAssemblyDispatch &&) = default;
    /** Default destructor */
    ~NEDepthwiseConvolutionAssemblyDispatch();
    /** Initialize the function's source, destination, kernels and border_size.
     *
     * @note Supports only NHWC format
     *
     * @param[in]  input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
     * @param[in]  weights          Weights tensor. These are 3D tensors with shape [W, H, IFM]. Data type supported: Same as @p input.
     * @param[in]  bias             (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output,
                   const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                   const Size2D &dilation = Size2D(1, 1));
    /** Static function to check if given info will lead to a valid configuration of @ref NEDepthwiseConvolutionAssemblyDispatch
     *
     * @note Supports only NHWC format
     *
     * @param[in]  input            Source tensor. Data type supported: QASYMM8/F16/F32. (Written to only for border filling).
     * @param[in]  weights          Weights tensor. These are 3D tensors with shape [W, H, IFM]. Data type supported: Same as @p input.
     * @param[in]  bias             (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                              Data type supported: Same as @p input.
     * @param[out] output           Destination tensor. Data type supported: same as @p input.
     * @param[in]  conv_info        Padding and stride information to use for the convolution.
     * @param[in]  depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return An error status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *bias, const ITensorInfo *output,
                           const PadStrideInfo &conv_info, unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                           const Size2D &dilation = Size2D(1, 1));
    /** Check if the optimized kernel can be used for the given kernel sizes and strides
     *
     * @warning Even if this return true the inputs and outputs might need to get permuted as the only layout supported is NHWC
     *
     * @param[in] input            Input tensor info.
     * @param[in] weights          Weights tensor info.
     * @param[in] conv_info        Convolution layer metadata.
     * @param[in] depth_multiplier (Optional) Depth multiplier to be used.
     * @param[in] dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     *
     * @return True if the assembly kernel could be used else false. Note that transformations of input/output could be needed.
     */
    static bool is_optimized_supported(const ITensorInfo *input, const ITensorInfo *weights, PadStrideInfo conv_info, unsigned int depth_multiplier = 1, const Size2D &dilation = Size2D(1, 1));

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    struct LocalImpl;

private:
    MemoryGroup                _memory_group;
    const ITensor             *_input;
    const ITensor             *_weights;
    const ITensor             *_bias;
    ITensor                   *_output;
    Tensor                     _packed_weights;
    Tensor                     _workspace;
    bool                       _is_prepared;
    std::unique_ptr<LocalImpl> _pImpl;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEDEPTHWISECONVOLUTIONASSEMBLYDISPATCH_H__ */
