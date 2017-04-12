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
#include "arm_compute/core/NEON/kernels/NEConvolutionLayerWeightsReshapeKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

namespace arm_compute
{
class ITensor;

/** Basic function to simulate a convolution layer. This function calls the following OpenCL kernels:
 * -# @ref NEConvolutionLayerWeightsReshapeKernel (executed only once for each configuration)
 * -# @ref NEGEMMTranspose1xWKernel               (executed only once for each configuration)
 * -# @ref NEIm2ColKernel
 * -# @ref NEGEMMInterleave4x4Kernel
 * -# @ref NEGEMMMatrixMultiplyKernel
 * -# @ref NECol2ImKernel
 */
class NEConvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEConvolutionLayer();
    /** Set the input and output tensors.
     *
     * @param[in]  input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                       while every optional dimension from 4 and above represent a batch of inputs.
     *                       Data types supported: F32.
     * @param[in]  weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: Same as @p input.
     * @param[in]  biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported: Same as @p input.
     * @param[out] output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                       Data types supported: Same as @p input.
     * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
     */
    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run() override;

private:
    NEIm2ColKernel                         _input_im2col_kernel;
    NEGEMMInterleave4x4Kernel              _input_interleave_kernel;
    NEConvolutionLayerWeightsReshapeKernel _weights_reshape_kernel;
    NEGEMMTranspose1xWKernel               _weights_transposed_kernel;
    NEGEMMMatrixMultiplyKernel             _mm_kernel;
    NECol2ImKernel                         _output_col2im_kernel;
    Tensor                                 _input_im2col_reshaped;
    Tensor                                 _input_interleaved_reshaped;
    Tensor                                 _weights_reshaped;
    Tensor                                 _weights_transposed;
    Tensor                                 _gemm_output;
    bool                                   _is_first_run;
    bool                                   _has_bias;
};
}
#endif /* __ARM_COMPUTE_NECONVOLUTIONLAYER_H__ */
