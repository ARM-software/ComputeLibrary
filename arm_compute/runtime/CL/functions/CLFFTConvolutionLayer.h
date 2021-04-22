/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLFFTCONVOLUTIONLAYER_H
#define ARM_COMPUTE_CLFFTCONVOLUTIONLAYER_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLFFT2D.h"
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/CL/functions/CLReverse.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"

namespace arm_compute
{
// Forward declarations
class ICLTensor;

/** Basic function to execute FFT-based convolution on OpenCL. This function calls the following OpenCL functions/kernels:
 *
 *  -# @ref CLPermute                        Permute input if NHWC(only NCHW is supported).
 *  -# @ref CLPadLayer                       Pad input.
 *  -# @ref CLFFT2D                          Forward transform to the frequency domain.
 *  -# @ref CLComplexPixelWiseMultiplication Complex element-wise product of input and the weights.
 *  -# @ref CLReductionOperation             Reduction across channels.
 *  -# @ref CLFFT2D                          Inverse transform back to the time domain.
 *  -# @ref CLStridedSlice                   Extract valid output.
 *  -# @ref CLArithmeticAddition             Add bias.
 *  -# @ref CLActivationLayer                Perform activation.
 *  -# @ref CLPermute                        Permute output if NHWC(only NCHW is supported).
 */
class CLFFTConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLFFTConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTConvolutionLayer(const CLFFTConvolutionLayer &) = delete;
    /** Default move constructor */
    CLFFTConvolutionLayer(CLFFTConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFFTConvolutionLayer &operator=(const CLFFTConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLFFTConvolutionLayer &operator=(CLFFTConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src    |dst    |
     * |:------|:------|
     * |F32    |F32    |
     * |F16    |F16    |
     *
     * @note: This function only works with any square kernel size and unit strides for both NCHW and NHWC data layout
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported:  F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);
    /** Set the input and output tensors.
     *
     * @note: This function only works with any square kernel size and unit strides for both NCHW and NHWC data layout
     *
     * @param[in]  compile_context  The compile context to be used.
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     */
    void configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CLFFTConvolutionLayer
     *
     * @note: This function only works with any square kernel size and unit strides for both NCHW and NHWC data layout
     *
     * @param[in]  input            Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                              while every optional dimension from 4 and above represent a batch of inputs.
     *                              Data types supported: F16/F32.
     * @param[in]  weights          Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]  biases           Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].Data type supported: Same as @p input
     * @param[out] output           Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                              Data types supported: Same as @p input.
     * @param[in]  conv_info        Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]  enable_fast_math (Optional) Enable fast math computation. In case this flag were set, the function could dispatch the fastest implementation
     *                              available which may introduce a drop of accuracy as well. Default is false
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo(), bool enable_fast_math = false);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    MemoryGroup                      _memory_group;
    CLReverse                        _flip_weights_func;
    CLPermute                        _permute_input_func;
    CLPermute                        _permute_output_func;
    CLPermute                        _permute_weights_func;
    CLPermute                        _permute_bias_func;
    CLPadLayer                       _pad_input_func;
    CLPadLayer                       _pad_weights_func;
    CLFFT2D                          _transform_input_func;
    std::unique_ptr<CLFFT2D>         _transform_weights_func;
    CLFFT2D                          _itransform_output_func;
    CLComplexPixelWiseMultiplication _prod_func;
    CLReductionOperation             _reduce_func;
    CLSlice                          _extract_output_func;
    CLArithmeticAddition             _bias_add_func;
    CLActivationLayer                _activation_layer_func;

    CLTensor _permuted_input;
    CLTensor _permuted_weights;
    CLTensor _permuted_bias;
    CLTensor _permuted_output;
    CLTensor _padded_input;
    CLTensor _padded_weights;
    CLTensor _flip_axis;
    CLTensor _flipped_weights;
    CLTensor _transformed_input;
    CLTensor _transformed_weights;
    CLTensor _input_weights_product;
    CLTensor _output_product;
    CLTensor _output_reduced;
    CLTensor _itransformed_output;
    CLTensor _reshaped_output;
    CLTensor _bias_output;

    const ICLTensor *_original_weights;
    const ICLTensor *_original_bias;
    bool             _is_activationlayer_enabled;
    bool             _needs_permute;
    bool             _has_bias;
    bool             _is_prepared;
};
} // namespace arm_compute
#endif /* ARM_COMPUTE_CLFFTCONVOLUTIONLAYER_H */
