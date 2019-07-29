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
#ifndef __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/kernels/CLCol2ImKernel.h"
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/CL/kernels/CLIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
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
     * @param[in]  weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                        Data type supported: QASYMM8/F16/F32.
     * @param[in]  biases     Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output     Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  num_groups (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLConvolutionLayerReshapeWeights
     *
     * @param[in] weights    Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                       Data type supported: QASYMM8/F16/F32.
     * @param[in] biases     Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[in] output     Destination tensor. Data types supported: Same as @p weights.
     * @param[in] num_groups (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, unsigned int num_groups = 1);
    // Inherited methods overridden:
    void run() override;

private:
    CLWeightsReshapeKernel _weights_reshape_kernel;
};

/** Basic function to compute the convolution layer. This function calls the following OpenCL kernels/functions:
 *
 * -# @ref CLIm2ColKernel
 * -# @ref CLGEMM (if the data type is FP32 or FP16)
 * -# @ref CLGEMMLowpMatrixMultiplyCore (if the data type is QASYMM8)
 * -# @ref CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint (if the data type is QASYMM8)
 * -# @ref CLElementwiseOperationKernel for addition (if biases != nullptr and we have a 1x1 convolution with the NHWC data layout)
 * -# @ref CLCol2ImKernel (if NCHW data layout)
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
     *                          Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo(),
                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with CLWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with CLGEMMReshapeRHSMatrixKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is only supported for NCHW data layout
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
     * @param[in]      input                 Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in]      weights               Weights tensor. Data type supported: Same as @p input.
     * @param[in]      biases                Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                       Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in, out] output                Output tensor. Data types supported: Same as @p input,
     *                                       except for input of QASYMM8 type where output should be of S32 type.
     * @param[in]      gemmlowp_output_stage GEMMLowp output stage info
     * @param[in]      gemm_3d_depth         Depth of GEMM 3D
     * @param[in]      act_info              Activation to apply after the matrix multiplication
     */
    void configure_mm(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const GEMMLowpOutputStageInfo &gemmlowp_output_stage, int gemm_3d_depth,
                      const ActivationLayerInfo &act_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] input                 Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in] weights               Weights tensor. Data type supported: Same as @p input.
     * @param[in] output                Output tensor. Data types supported: Same as @p input,
     *                                  except for input of QASYMM8 type where output should be of S32 type.
     * @param[in] biases                Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                                  Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in] gemmlowp_output_stage GEMMLowp output stage info
     * @param[in] gemm_3d_depth         Depth of GEMM 3D
     * @param[in] skip_im2col           Flag which specifies if im2col has to be skipped. i.e. 1x1 convolution with NHWC data layout.
     * @param[in] act_info              Activation to apply after the matrix multiplication
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const GEMMLowpOutputStageInfo &gemmlowp_output_stage,
                              int gemm_3d_depth, bool skip_im2col, const ActivationLayerInfo &act_info);

private:
    CLMemoryGroup                    _memory_group;
    CLConvolutionLayerReshapeWeights _reshape_weights;
    CLIm2ColKernel                   _im2col_kernel;
    CLGEMM                           _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore     _mm_gemmlowp;
    CLCol2ImKernel                   _col2im_kernel;
    CLActivationLayer                _activationlayer_function;

    const ICLTensor *_original_weights;

    CLTensor _im2col_output;
    CLTensor _weights_reshaped;
    CLTensor _gemm_output;

    bool _skip_im2col;
    bool _skip_col2im;
    bool _is_quantized;
    bool _fuse_activation;
    bool _is_prepared;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMCONVOLUTIONLAYER_H__ */
