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
#ifndef __ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H__
#define __ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H__

#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseIm2ColKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseVectorToTensorKernel.h"
#include "arm_compute/core/CL/kernels/CLDepthwiseWeightsReshapeKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLGEMMMatrixVectorMultiplyKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute a depthwise convolution for kernel size 3x3xC. This function calls the following OpenCL kernels:
 *
 * -# @ref CLDepthwiseConvolutionLayer3x3Kernel
 * -# @ref CLFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class CLDepthwiseConvolutionLayer3x3 : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer3x3();
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in, out] input     Source tensor. Data type supported: QASYMM8/F32. (Written to only for border filling).
     * @param[in]      weights   Weights tensor. A 3D tensor with shape [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases    (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                           Data type supported: Same as @p input.
     * @param[out]     output    Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info Padding and stride information to use for the convolution.
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overriden:
    void run() override;

private:
    CLDepthwiseConvolutionLayer3x3Kernel _kernel;
    CLFillBorderKernel                   _border_handler;
};

/** Basic function to execute a generic depthwise convolution. This function calls the following OpenCL kernels:
 *
 * -# @ref CLDepthwiseIm2ColKernel
 * -# @ref CLGEMMMatrixVectorMultiplyKernel
 * -# @ref CLDepthwiseWeightsReshapeKernel
 * -# @ref CLFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class CLDepthwiseConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolutionLayer();
    /** Initialize the function's source, destination, weights and convolution information.
     *
     * @param[in, out] input     Source tensor. Data type supported: F32. (Written to only for border filling).
     * @param[in]      weights   Weights tensor. These are 3D tensors with shape [kernel_x, kernel_y, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases    (Optional) Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                           Data type supported: Same as @p input.
     * @param[out]     output    Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info Padding and stride information to use for the convolution.
     */
    void configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overriden:
    void run() override;

private:
    CLDepthwiseIm2ColKernel          _im2col_kernel;
    CLDepthwiseWeightsReshapeKernel  _weights_reshape_kernel;
    CLGEMMMatrixVectorMultiplyKernel _v2mm_kernel;
    CLDepthwiseVectorToTensorKernel  _vector_to_tensor_kernel;
    CLFillBorderKernel               _v2mm_input_fill_border;
    CLFillBorderKernel               _v2mm_weights_fill_border;
    CLTensor                         _input_reshaped;
    CLTensor                         _weights_reshaped;
    CLTensor                         _v2mm_output;
};
}
#endif /*__ARM_COMPUTE_CLDEPTHWISECONVOLUTION_H__ */
