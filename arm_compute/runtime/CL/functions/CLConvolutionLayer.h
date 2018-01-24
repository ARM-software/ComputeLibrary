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
#ifndef __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__

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
 * -# @ref CLGEMMTranspose1xWKernel
 */
class CLConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    CLConvolutionLayerReshapeWeights(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input and output tensors.
     *
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: QS8/QASYMM8/QS16/F16/F32.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output       Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  transpose1xW True if the weights are to undergo a 1xW transposition after reshaping (in case of GEMM operation), false otherwise.
     *                          Data types supported: Same as @p weights.
     */
    void configure(const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, bool transpose1xW);
    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup            _memory_group;
    CLWeightsReshapeKernel   _weights_reshape_kernel;
    CLGEMMTranspose1xWKernel _weights_transposed_kernel;
    CLTensor                 _weights_reshaped;
    bool                     _transpose1xW;
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
class CLConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
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
     */
    void configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param input                     Input tensor. Data types supported: QS8/QASYMM8/QS16/F16/F32.
     * @param weights                   Weights tensor. Data type supported: Same as @p input.
     * @param output                    Output tensor. Data types supported: Same as @p input,
     *                                                 except for input of QASYMM8 type where output should be of S32 type.
     * @param is_interleaved_transposed Flag that signals if matrix is interleaved transposed
     */
    void configure_mm(const ICLTensor *input, const ICLTensor *weights, ICLTensor *output, bool is_interleaved_transposed, bool are_weights_reshaped);

private:
    CLMemoryGroup                                       _memory_group;
    CLConvolutionLayerReshapeWeights                    _reshape_weights;
    CLIm2ColKernel                                      _im2col_kernel;
    CLGEMMInterleave4x4Kernel                           _interleave_kernel;
    CLGEMMMatrixMultiplyKernel                          _mm_kernel;
    CLGEMM                                              _mm_gemm;
    CLGEMMLowpMatrixMultiplyCore                        _mm_gemmlowp;
    CLGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint _gemmlowp_output_stage;
    CLCol2ImKernel                                      _col2im_kernel;

    CLTensor _im2col_output;
    CLTensor _interleave_output;
    CLTensor _weights_reshaped;
    CLTensor _weights_transposed;
    CLTensor _gemm_output;
    CLTensor _tmp_output;

    bool _are_weights_reshaped;
    bool _is_quantized;
    bool _is_interleaved_transposed;
};
}
#endif /* __ARM_COMPUTE_CLCONVOLUTIONLAYER_H__ */
