/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NEWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
class ITensor;

/** Function to reshape the weights. This function calls the following kernel:
 * -# @ref NEWeightsReshapeKernel
 */
class NEConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    NEConvolutionLayerReshapeWeights();
    /** Set the input and output tensors.
     *
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/F16/F32.
     * @param[in]  biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output  Destination tensor. Data types supported: Same as @p weights.
     */
    void configure(const ITensor *weights, const ITensor *biases, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvolutionLayerReshapeWeights
     *
     * @param[in] weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/F16/F32.
     * @param[in] biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output  Destination tensor. Data types supported: Same as @p weights.
     *
     * @return an error status
     */
    static Status validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output);

    // Inherited methods overridden:
    void run() override;

private:
    NEWeightsReshapeKernel _weights_reshape_kernel;
};

/** Basic function to compute the convolution layer. This function calls the following NEON kernels/functions:
 *
 * -# @ref NEIm2ColKernel
 * -# @ref NEGEMM (if the data type is FP32 or FP16)
 * -# @ref NEGEMMLowpMatrixMultiplyCore (if the data type is QASYMM8)
 * -# @ref NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint (if the data type is QASYMM8)
 * -# @ref NEArithmeticAdditionKernel (if biases != nullptr and we have a 1x1 convolution with the NHWC data layout)
 * -# @ref NECol2ImKernel (if NCHW data layout)
 *
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
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);
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
     * @param[in] num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param[in]  input         Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights       Weights tensor. Data type supported: Same as @p input.
     * @param[in]  biases        Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                           Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output        Output tensor. Data types supported: Same as @p input,
     *                           except for input of QASYMM8 type where output should be of S32 type.
     * @param[in]  act_info      (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  gemm_3d_depth (Optional) Depth of GEMM 3D (Defaults to 1)
     */
    void configure_mm(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info = ActivationLayerInfo(), int gemm_3d_depth = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] input         Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in] weights       Weights tensor. Data type supported: Same as @p input.
     * @param[in] biases        Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in] output        Output tensor. Data types supported: Same as @p input,
     *                          except for input of QASYMM8 type where output should be of S32 type.
     * @param[in] act_info      (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] gemm_3d_depth (Optional) Depth of GEMM 3D (Defaults to 1)
     * @param[in] skip_im2col   (Optional) Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout. (Default to false)
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ActivationLayerInfo &act_info = ActivationLayerInfo(),
                              int gemm_3d_depth = 1, bool skip_im2col = false);
    /** Static function to check if GEMM3D is supported in @ref NEGEMM or in @ref NEGEMMLowpMatrixMultiplyCore
     *
     * @param[in] input_info    Input tensor info. Data types supported: QASYMM8/F16/F32.
     * @param[in] act_info      Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] gemm_3d_depth Depth of GEMM 3D
     * @param[in] skip_im2col   Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout
     *
     * @return a status
     */
    static Status validate_gemm3d(const ITensorInfo *input_info, const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col);

private:
    MemoryGroup                      _memory_group;
    NEConvolutionLayerReshapeWeights _reshape_weights;
    NEIm2ColKernel                   _im2col_kernel;
    NEGEMM                           _mm_gemm;
    NEGEMMLowpMatrixMultiplyCore     _mm_gemmlowp;
    NECol2ImKernel                   _col2im_kernel;
    NEActivationLayer                _activationlayer_function;
    NEArithmeticAdditionKernel       _add_bias_kernel;
    NEReshapeLayer                   _reshape_layer;

    const ITensor *_original_weights;

    Tensor _im2col_output;
    Tensor _weights_reshaped;
    Tensor _gemm_output;
    Tensor _tmp_output;

    DataLayout _data_layout;

    bool _append_bias;
    bool _skip_im2col;
    bool _skip_col2im;
    bool _is_quantized;
    bool _is_activationlayer_enabled;
    bool _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NECONVOLUTIONGEMMLAYER_H__ */
