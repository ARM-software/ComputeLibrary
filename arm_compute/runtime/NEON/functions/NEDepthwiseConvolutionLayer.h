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
#ifndef __ARM_COMPUTE_NEDEPTHWISECONVOLUTION_H__
#define __ARM_COMPUTE_NEDEPTHWISECONVOLUTION_H__

#include "arm_compute/core/NEON/kernels/NEDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseIm2ColKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseVectorToTensorKernel.h"
#include "arm_compute/core/NEON/kernels/NEDepthwiseWeightsReshapeKernel.h"
#include "arm_compute/core/NEON/kernels/NEDirectConvolutionLayerOutputStageKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixVectorMultiplyKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to execute a depthwise convolution for kernel size 3x3xC. This function calls the following NEON kernels:
 *
 * -# @ref NEDepthwiseConvolutionLayer3x3
 * -# @ref NEFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class NEDepthwiseConvolutionLayer3x3 : public IFunction
{
public:
    /** Default constructor */
    NEDepthwiseConvolutionLayer3x3();
    /** Initialize the function's source, destination, kernels and border_size.
     *
     * @param[in, out] input     Source tensor. Data type supported: QASYMM8, F32. (Written to only for border filling).
     * @param[in]      weights   Weights tensor. These are 3D tensors with shape [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases    (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                           Data type supported: Same as @p input.
     * @param[out]     output    Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info Padding and stride information to use for the convolution.
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overriden:
    void run() override;

private:
    NEDepthwiseConvolutionLayer3x3Kernel      _kernel;
    NEDirectConvolutionLayerOutputStageKernel _output_stage_kernel;
    NEFillBorderKernel                        _border_handler;
    Tensor                                    _accumulator;
    bool                                      _has_bias;
    bool                                      _is_quantized;
};

/** Basic function to execute a generic depthwise convolution. This function calls the following NEON kernels:
 *
 * -# @ref NEDepthwiseIm2ColKernel
 * -# @ref NEDepthwiseWeightsReshapeKernel
 * -# @ref NEGEMMMatrixVectorMultiplyKernel
 * -# @ref NEFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class NEDepthwiseConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    NEDepthwiseConvolutionLayer();
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in, out] input     Source tensor. Data type supported: F32. (Written to only for border filling).
     * @param[out]     output    Destination tensor. Data type supported: same as @p input.
     * @param[in]      weights   Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases    (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                           Data type supported: Same as @p input.
     * @param[in]      conv_info Padding and stride information to use for the convolution.
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overriden:
    void run() override;

private:
    NEDepthwiseIm2ColKernel          _im2col_kernel;
    NEDepthwiseWeightsReshapeKernel  _weights_reshape_kernel;
    NEGEMMMatrixVectorMultiplyKernel _v2mm_kernel;
    NEDepthwiseVectorToTensorKernel  _vector_to_tensor_kernel;
    Tensor                           _input_reshaped;
    Tensor                           _weights_reshaped;
    Tensor                           _v2mm_output;
};
}
#endif /* __ARM_COMPUTE_NEDEPTHWISECONVOLUTION_H__ */