/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#ifndef ARM_COMPUTE_GCCONVOLUTIONLAYER_H
#define ARM_COMPUTE_GCCONVOLUTIONLAYER_H

#include "arm_compute/core/GLES_COMPUTE/kernels/GCCol2ImKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCIm2ColKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCWeightsReshapeKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCGEMM.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/MemoryGroup.h"

#include <memory>

namespace arm_compute
{
class IGCTensor;

/** Function to reshape and transpose the weights. This function calls the following kernels:
 * -# @ref GCWeightsReshapeKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCConvolutionLayerReshapeWeights : public IFunction
{
public:
    /** Constructor */
    GCConvolutionLayerReshapeWeights();
    /** Set the input and output tensors.
     *
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM].
     *                     Data type supported: F16/F32.
     * @param[in]  biases  Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output  Destination tensor. Data types supported: Same as @p weights.
     */
    void configure(const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output);
    // Inherited methods overridden:
    void run() override;

private:
    GCWeightsReshapeKernel _weights_reshape_kernel;
};

/** Basic function to compute the convolution layer. This function calls the following GLES kernels:
 *
 * -# @ref GCWeightsReshapeKernel (executed only once for each configuration)
 * -# @ref GCGEMMTranspose1xWKernel (executed only once for each configuration)
 * -# @ref GCIm2ColKernel
 * -# @ref GCGEMMInterleave4x4Kernel
 * -# @ref GCCol2ImKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    GCConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCConvolutionLayer(const GCConvolutionLayer &) = delete;
    /** Default move constructor */
    GCConvolutionLayer(GCConvolutionLayer &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCConvolutionLayer &operator=(const GCConvolutionLayer &) = delete;
    /** Default move assignment operator */
    GCConvolutionLayer &operator=(GCConvolutionLayer &&) = default;
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
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info,
                   const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run() override;
    void prepare() override;

private:
    /** Configures the appropriate matrix multiply routine
     *
     * @param input                     Input tensor. Data types supported: F16/F32.
     * @param weights                   Weights tensor. Data type supported: Same as @p input.
     * @param output                    Output tensor. Data types supported: Same as @p input,
     */
    void configure_mm(const IGCTensor *input, const IGCTensor *weights, IGCTensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref GCGEMMConvolutionLayer matrix multiply routines
     *
     * @param[in] input   Input tensor. Data types supported: QASYMM8/F16/F32.
     * @param[in] weights Weights tensor. Data type supported: Same as @p input.
     * @param[in] output  Output tensor. Data types supported: Same as @p input,
     *                                      except for input of QASYMM8 type where output should be of S32 type.
     *
     * @return a status
     */
    static Status validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *output);

private:
    MemoryGroup                      _memory_group;
    GCConvolutionLayerReshapeWeights _reshape_weights;
    GCIm2ColKernel                   _input_im2col_kernel;
    GCGEMM                           _mm_gemm;
    GCCol2ImKernel                   _output_col2im_kernel;
    GCFillBorderKernel               _fill_border;
    GCActivationLayer                _activationlayer_function;

    const IGCTensor *_original_weights;

    GCTensor _input_im2col_reshaped;
    GCTensor _input_interleaved_reshaped;
    GCTensor _weights_reshaped;
    GCTensor _weights_transposed;
    GCTensor _gemm_output;
    GCTensor _tmp_output;

    bool _is_activationlayer_enabled;
    bool _is_prepared;
};
}

#endif /* ARM_COMPUTE_GCCONVOLUTIONLAYER_H */
