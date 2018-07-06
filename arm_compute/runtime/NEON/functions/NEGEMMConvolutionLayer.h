/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NEGEMMCONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"
#include "arm_compute/core/NEON/kernels/NECol2ImKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMAssemblyBaseKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Function to reshape and perform 1xW transposition on the weights. This function calls the following kernels:
 * -# @ref NEWeightsReshapeKernel
 * -# @ref NEGEMMTranspose1xWKernel (executed in case GEMM is required for the operation)
 */
class NEConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    NEConvolutionLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/F32.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output       Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  transpose1xW True if the weights are to undergo a 1xW transposition after reshaping (in case of GEMM operation), false otherwise.
     *                          Data types supported: Same as @p weights.
     */
    void configure(const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose1xW);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvolutionLayerReshapeWeights
     *
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/F16/F32.
     * @param[in] biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output       Destination tensor. Data types supported: Same as @p weights.
     * @param[in] transpose1xW True if the weights are to undergo a 1xW transposition after reshaping (in case of GEMM operation), false otherwise.
     *                         Data types supported: Same as @p weights.
     *
     * @return an error status
     */
    static Status validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, bool transpose1xW);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup              _memory_group;
    NEWeightsReshapeKernel   _weights_reshape_kernel;
    NEGEMMTranspose1xWKernel _weights_transposed_kernel;
    Tensor                   _weights_reshaped;
    bool                     _transpose1xW;
};

/** Basic function to simulate a convolution layer. This function calls the following NEON kernels:
 * -# @ref NEWeightsReshapeKernel   (executed only once for each configuration)
 * -# @ref NEIm2ColKernel
 * -# @ref NEGEMMInterleave4x4Kernel (executed only in case GEMM is required for the operation)
 * -# @ref NEGEMMMatrixMultiplyKernel or @ref NEGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 * -# @ref NEGEMMLowpQuantizeDownInt32ToUint8Scale (if quantized asymmetric)
 * -# @ref NECol2ImKernel
 * -# @ref NEActivationLayer (executed only if the activation layer is enabled)
 */
class NEGEMMConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEGEMMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConvolutionLayer(const NEGEMMConvolutionLayer &) = delete;
    /** Default move constructor */
    NEGEMMConvolutionLayer(NEGEMMConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMConvolutionLayer &operator=(const NEGEMMConvolutionLayer &) = delete;
    /** Default move assignment operator */
    NEGEMMConvolutionLayer &operator=(NEGEMMConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer
     *
     * @param[in] input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs.
     *                         Data types supported: QASYMM8/F16/F32.
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                         Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                         Data types supported: Same as @p input.
     * @param[in] conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                         tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info     (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param[in]  input          Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights        Weights tensor. Data type supported: Same as @p input.
     * @param[out] output         Output tensor. Data types supported: Same as @p input,
     *                            except for input of QASYMM8 type where output should be of S32 type.
     * @param[in]  is_interleaved (Optional) True if input0 and input1 have been reshaped respectively using @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel
     * @param[in]  reshape_info   (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     */
    void configure_mm(const ITensor *input, const ITensor *weights, ITensor *output, bool is_interleaved, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo());

private:
    MemoryGroup                                         _memory_group;
    NEGEMMAssemblyDispatchF32                           _asm_glue;
    NEIm2ColKernel                                      _input_im2col_kernel;
    NEGEMMInterleave4x4Kernel                           _input_interleave_kernel;
    NEConvolutionLayerReshapeWeights                    _reshape_weights;
    NEGEMMMatrixMultiplyKernel                          _mm_kernel;
    NEGEMMLowpMatrixMultiplyCore                        _mm_gemmlowp;
    NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint _gemmlowp_output_stage;
    NECol2ImKernel                                      _output_col2im_kernel;
    NEActivationLayer                                   _activationlayer_function;
    NEArithmeticAdditionKernel                          _add_bias_kernel;

    const ITensor *_original_weights;

    Tensor _input_im2col_reshaped;
    Tensor _input_interleaved_reshaped;
    Tensor _weights_reshaped;
    Tensor _gemm_output;
    Tensor _tmp_output;

    DataLayout _data_layout;
    bool       _append_bias;
    bool       _is_fully_connected_convolution;
    bool       _are_weights_reshaped;
    bool       _is_quantized;
    bool       _is_interleaved;
    bool       _is_activationlayer_enabled;
    bool       _skip_im2col;
    bool       _is_prepared;
};
}
#endif /* __ARM_COMPUTE_NECONVOLUTIONGEMMLAYER_H__ */
