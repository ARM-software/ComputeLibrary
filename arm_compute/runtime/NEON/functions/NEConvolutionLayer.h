/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NECONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NECONVOLUTIONLAYER_H__

#include "arm_compute/runtime/IFunction.h"

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
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QS8/QS16/F32.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p weights.
     * @param[out] output       Destination tensor. Data types supported: Same as @p weights.
     * @param[in]  transpose1xW True if the weights are to undergo a 1xW transposition after reshaping (in case of GEMM operation), false otherwise.
     *                          Data types supported: Same as @p weights.
     */
    void configure(const ITensor *weights, const ITensor *biases, ITensor *output, bool transpose1xW);
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvolutionLayerReshapeWeights
     *
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QS8/QS16/F16/F32.
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
 * -# @ref NEGEMMMatrixMultiplyKernel
 * -# @ref NECol2ImKernel
 */
class NEConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEConvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Set the input and output tensors.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QS8/QS16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                          tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info = WeightsInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEConvolutionLayer
     *
     * @param[in] input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs.
     *                         Data types supported: QS8/QS16/F16/F32.
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in] biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported:Same as @p input.
     * @param[in] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                         Data types supported: Same as @p input.
     * @param[in] conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *                         tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                           const WeightsInfo &weights_info = WeightsInfo());

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                               _memory_group;
    NEIm2ColKernel                            _input_im2col_kernel;
    NEGEMMInterleave4x4Kernel                 _input_interleave_kernel;
    NEConvolutionLayerReshapeWeights          _reshape_weights;
    NEGEMMMatrixMultiplyKernel                _mm_kernel;
    std::unique_ptr<NEGEMMAssemblyBaseKernel> _mm_optimised_kernel;
    NECol2ImKernel                            _output_col2im_kernel;
    Tensor                                    _input_im2col_reshaped;
    Tensor                                    _input_interleaved_reshaped;
    Tensor                                    _weights_reshaped;
    Tensor                                    _gemm_output;
    Tensor                                    _workspace;
    bool                                      _has_bias;
    bool                                      _is_fully_connected_convolution;
    bool                                      _are_weights_reshaped;
};
}
#endif /* __ARM_COMPUTE_NECONVOLUTIONLAYER_H__ */
