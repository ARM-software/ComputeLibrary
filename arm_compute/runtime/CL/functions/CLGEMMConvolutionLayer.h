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
#ifndef __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMTranspose1xWKernel.h"
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <memory>

namespace arm_compute
{
class ICLTensor;

/** Function to reshape and transpose the weights. This function calls the following kernels:
 * -# @ref CLWeightsReshapeKernel
 */
class CLConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    CLConvolutionLayerReshapeWeights();
    /** Set the input and output tensors.
     *
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                     Data type supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in]  biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output  Destination tensor. Data types supported: Same as @p weights.
     */
    void configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvolutionLayerReshapeWeights
     *
     * @param[in] weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                    Data type supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in] biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output  Destination tensor. Data types supported: Same as @p weights.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output);
    // Inherited methods overridden:
    void run() override;

private:
    CLWeightsReshapeKernel _weights_reshape_kernel;
};

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * Note: weights already reshaped for quantized asymmetric is not supported
 *
 * -# @ref CLIm2ColKernel
 * -# @ref CLGEMMLowpMatrixMultiplyCore (if quantized asymmetric)
 * -# @ref CLGEMMLowpQuantizeDownInt32ToUint8Scale (if quantized asymmetric)
 * -# @ref CLCol2ImKernel
 *
 * if the weights are already reshaped:
 * -# @ref CLGEMMInterleave4x4Kernel
 * -# @ref CLGEMMMatrixMultiplyKernel
 * else
 * -# @ref CLGEMM
 */
class CLGEMMConvolutionLayer : public IFunction
{
public:
    /** Default constructor
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    CLGEMMConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer(const CLGEMMConvolutionLayer &) = delete;
    /** Default move constructor */
    CLGEMMConvolutionLayer(CLGEMMConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMConvolutionLayer &operator=(const CLGEMMConvolutionLayer &) = delete;
    /** Default move assignment operator */
    CLGEMMConvolutionLayer &operator=(CLGEMMConvolutionLayer &&) = default;
    /** Set the input and output tensors.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param input                     Input tensor. Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param weights                   Weights tensor. Data type supported: Same as @p input.
     * @param output                    Output tensor. Data types supported: Same as @p input,
     *                                                 except for input of QASYMM8 type where output should be of S32 type.
     */
    void configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] input   Input tensor. Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in] weights Weights tensor. Data type supported: Same as @p input.
     * @param[in] output  Output tensor. Data types supported: Same as @p input,
     *                                      except for input of QASYMM8 type where output should be of S32 type.
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output);

private:
    CLMemoryGroup                                       _memory_group;
    CLConvolutionLayerReshapeWeights                    _reshape_weights;
    CLIm2ColKernel                                      _im2col_kernel;
    CLGEMM                                              _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore                        _mm_gemmlowp;
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint _gemmlowp_output_stage;
    CLCol2ImKernel                                      _col2im_kernel;
    CLActivationLayer                                   _activationlayer_function;

    const ICLTensor *_original_weights;

    CLTensor _im2col_output;
    CLTensor _weights_reshaped;
    CLTensor _gemm_output;
    CLTensor _tmp_output;

    bool _is_quantized;
    bool _is_first_run;
    bool _is_activationlayer_enabled;
};
}
#endif /* __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__ */
