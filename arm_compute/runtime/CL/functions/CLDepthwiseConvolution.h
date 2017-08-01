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
#ifndef __ARM_COMPUTE_CL_DEPTHWISE_CONVOLUTION_H__
#define __ARM_COMPUTE_CL_DEPTHWISE_CONVOLUTION_H__

#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionKernel.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstdint>

namespace arm_compute
{
class ICLTensor;

/** Basic function to execute depthwise convolution. This function calls the following OpenCL kernels:
 *
 * -# @ref CLFillBorderKernel (executed if border_mode == CONSTANT or border_mode == REPLICATE)
 * -# @ref CLDepthwiseConvolutionKernel
 *
 */
class CLDepthwiseConvolution : public IFunction
{
public:
    /** Default constructor */
    CLDepthwiseConvolution();
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in]  input     Source tensor. DataType supported: F32. (Written to only for border filling).
     * @param[out] output    Destination tensor. DataType supported: F32.
     * @param[in]  weights   Weights tensor. These are 3D tensors with dimensions [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]  conv_info Padding and stride information to use for the convolution. DataType supported: F32.
     */
    void configure(ICLTensor *input, ICLTensor *output, const ICLTensor *weights, const PadStrideInfo &conv_info);

    // Inherited methods overriden:
    void run() override;

private:
    CLDepthwiseConvolutionKernel _kernel;
    CLFillBorderKernel           _border_handler;
};
}
#endif /*__ARM_COMPUTE_CL_DEPTHWISE_CONVOLUTION_H__ */
