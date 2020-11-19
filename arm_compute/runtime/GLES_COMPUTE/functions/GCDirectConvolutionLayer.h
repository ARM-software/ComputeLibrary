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
#ifndef ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYER_H
#define ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYER_H

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDirectConvolutionLayerKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTensorShiftKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute direct convolution function. This function calls the following kernels:
 *
 * -# @ref GCDirectConvolutionLayerKernel
 * -# @ref GCFillBorderKernel
 * -# @ref GCTensorShiftKernel
 *
 * @note Supported kernel size: 1x1, 3x3, and 5x5
 * @note This OpenGL ES implementation works with stride_x = 1 and 2
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class GCDirectConvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    GCDirectConvolutionLayer();
    /** Set the input and output tensors.
     *
     * @param[in,out] input     Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: F16/F32.
     *                          input will be written to only if it is currently left aligned.
     * @param[in]     weights   Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported:Same as @p input.
     * @param[in]     biases    Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM]. Data type supported:Same as @p input.
     * @param[out]    output    Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *                          Data types supported: Same as @p input.
     * @param[in]     conv_info Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]     act_info  (Optional) Activation layer information in case of a fused activation.
     */
    void configure(IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run() override final;

private:
    std::unique_ptr<IGCKernel> _kernel;
    GCFillBorderKernel         _border_handler;
    GCTensorShiftKernel        _shift_handler;
};
}
#endif /* ARM_COMPUTE_GCDIRECTCONVOLUTIONLAYER_H */
