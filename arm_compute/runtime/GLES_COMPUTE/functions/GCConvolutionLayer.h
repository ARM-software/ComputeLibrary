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

#ifndef __ARM_COMPUTE_GCCONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_GCCONVOLUTIONLAYER_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCCol2ImKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMTranspose1xWKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class IGCTensor;

/** Function to reshape and transpose the weights. This function calls the following kernels:
 * -# @ref GCWeightsReshapeKernel
 * -# @ref GCGEMMTranspose1xWKernel
 */
class GCConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    GCConvolutionLayerReshapeWeights();
    /** Set the input and output tensors.
     *
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                          Data type supported: F16/F32.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output       Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  transpose1xW True if the weights are to undergo a 1xW transposition after reshaping (in case of GEMM operation), false otherwise.
     *                          Data types supported: Same as @p weights.
     */
    void configure(const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, bool transpose1xW);
    // Inherited methods overridden:
    void run() override;

private:
    GCWeightsReshapeKernel   _weights_reshape_kernel;
    GCGEMMTranspose1xWKernel _weights_transposed_kernel;
    GCTensor                 _weights_reshaped;
    bool                     _transpose1xW;
};

/** Basic function to compute the convolution layer. This function calls the following GLES kernels:
 *
 * -# @ref GCWeightsReshapeKernel (executed only once for each configuration)
 * -# @ref GCGEMMTranspose1xWKernel (executed only once for each configuration)
 * -# @ref GCIm2ColKernel
 * -# @ref GCGEMMInterleave4x4Kernel
 * -# @ref GCCol2ImKernel
 */
class GCConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    GCConvolutionLayer();

    /** Set the input and output tensors.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with GCWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with GCGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param input                     Input tensor. Data types supported: F16/F32.
     * @param weights                   Weights tensor. Data type supported: Same as @p input.
     * @param output                    Output tensor. Data types supported: Same as @p input,
     * @param is_interleaved_transposed Flag that signals if matrix is interleaved transposed
     */
    void configure_mm(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output, bool is_interleaved_transposed = true);

private:
    GCConvolutionLayerReshapeWeights _reshape_weights;
    GCIm2ColKernel                   _input_im2col_kernel;
    GCGEMMInterleave4x4Kernel        _input_interleave_kernel;
    GCGEMMMatrixMultiplyKernel       _mm_kernel;
    GCCol2ImKernel                   _output_col2im_kernel;
    GCFillBorderKernel               _fill_border;

    GCTensor _input_im2col_reshaped;
    GCTensor _input_interleaved_reshaped;
    GCTensor _weights_reshaped;
    GCTensor _weights_transposed;
    GCTensor _gemm_output;
    GCTensor _tmp_output;

    bool _append_bias;
    bool _is_fully_connected_convolution;
    bool _are_weights_reshaped;
};
}

#endif /* __ARM_COMPUTE_GCCONVOLUTIONLAYER_H__ */
