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
#ifndef __ARM_COMPUTE_NEFFTCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NEFFTCONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEFFT2D.h"
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReverse.h"
#include "arm_compute/runtime/NEON/functions/NESlice.h"

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Basic function to execute FFT-based convolution on NEON. This function calls the following NEON functions/kernels:
 *
 *  -# @ref NEPermute                        Permute input if NHWC(only NCHW is supported).
 *  -# @ref NEPadLayer                       Pad input.
 *  -# @ref NEFFT2D                          Forward transform to the frequency domain.
 *  -# @ref NEComplexPixelWiseMultiplication Complex element-wise product of input and the weights.
 *  -# @ref NEReductionOperation             Reduction across channels.
 *  -# @ref NEFFT2D                          Inverse transform back to the time domain.
 *  -# @ref NEStridedSlice                   Extract valid output.
 *  -# @ref NEArithmeticAddition             Add bias.
 *  -# @ref NEActivationLayer                Perform activation.
 *  -# @ref NEPermute                        Permute output if NHWC(only NCHW is supported).
 */
class NEFFTConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    NEFFTConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTConvolutionLayer(const NEFFTConvolutionLayer &) = delete;
    /** Default move constructor */
    NEFFTConvolutionLayer(NEFFTConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFFTConvolutionLayer &operator=(const NEFFTConvolutionLayer &) = delete;
    /** Default move assignment operator */
    NEFFTConvolutionLayer &operator=(NEFFTConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @note: This function only works with any square kernel size and unit strides for both NCHW and NHWC data layout
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info  (Optional) Activation layer information in case of a fused activation.
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEFFTConvolutionLayer
     *
     * @note: This function only works with any square kernel size and unit strides for both NCHW and NHWC data layout
     *
     * @param[in] input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                      while every optional dimension from 4 and above represent a batch of inputs.
     *                      Data types supported: F32.
     * @param[in] weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[in] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                      Data types supported: Same as @p input.
     * @param[in] conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] act_info  (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                      _memory_group;
    NEReverse                        _flip_weights_func;
    NEPermute                        _permute_input_func;
    NEPermute                        _permute_output_func;
    NEPermute                        _permute_weights_func;
    NEPermute                        _permute_bias_func;
    NEPadLayer                       _pad_input_func;
    NEPadLayer                       _pad_weights_func;
    NEFFT2D                          _transform_input_func;
    std::unique_ptr<NEFFT2D>         _transform_weights_func;
    NEFFT2D                          _itransform_output_func;
    NEComplexPixelWiseMultiplication _prod_func;
    NEReductionOperation             _reduce_func;
    NESlice                          _extract_output_func;
    NEArithmeticAddition             _bias_add_func;
    NEActivationLayer                _activation_layer_func;

    Tensor _permuted_input;
    Tensor _permuted_weights;
    Tensor _permuted_bias;
    Tensor _permuted_output;
    Tensor _padded_input;
    Tensor _padded_weights;
    Tensor _flip_axis;
    Tensor _flipped_weights;
    Tensor _transformed_input;
    Tensor _transformed_weights;
    Tensor _input_weights_product;
    Tensor _output_product;
    Tensor _output_reduced;
    Tensor _itransformed_output;
    Tensor _reshaped_output;
    Tensor _bias_output;

    const ITensor *_original_weights;
    const ITensor *_original_bias;
    bool           _is_activationlayer_enabled;
    bool           _needs_permute;
    bool           _has_bias;
    bool           _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEFFTCONVOLUTIONLAYER_H__ */
