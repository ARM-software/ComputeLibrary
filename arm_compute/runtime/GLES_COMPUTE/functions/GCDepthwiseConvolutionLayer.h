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
#ifndef __ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__
#define __ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthwiseConvolutionLayer3x3Kernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCFillBorderKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCTensorShiftKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCActivationLayer.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute a depthwise convolution for kernel size 3x3xC. This function calls the following OpenGLES kernels:
 *
 * -# @ref GCDepthwiseConvolutionLayer3x3Kernel
 * -# @ref GCFillBorderKernel (if pad_x or pad_y > 0)
 *
 */
class GCDepthwiseConvolutionLayer3x3 : public IFunction
{
public:
    /** Default constructor */
    GCDepthwiseConvolutionLayer3x3();
    /** Initialize the function's source, destination, conv and border_size.
     *
     * @param[in, out] input            Source tensor. Data type supported: F16. (Written to only for border filling).
     * @param[in]      weights          Weights tensor. A 3D tensor with shape [3, 3, IFM]. Data type supported: Same as @p input.
     * @param[in]      biases           Biases tensor. A 1D tensor with shape [IFM]. Must be nullptr if not needed.
     *                                  Data type supported: Same as @p input.
     * @param[out]     output           Destination tensor. Data type supported: same as @p input.
     * @param[in]      conv_info        Padding and stride information to use for the convolution.
     * @param[in]      depth_multiplier (Optional) Multiplier to apply to the input's depth in order to retrieve the output's depth. Defaults to 1.
     * @param[in]      act_info         (Optional) Activation layer information in case of a fused activation.
     * @param[in]      dilation         (Optional) Dilation, in elements, across x and y. Defaults to (1, 1). Currently supports (1,1) only.
     */
    void configure(IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info,
                   unsigned int depth_multiplier = 1, const ActivationLayerInfo &act_info = ActivationLayerInfo(), const Size2D &dilation = Size2D(1U, 1U));

    // Inherited methods overridden:
    void run() override final;

private:
    std::unique_ptr<IGCKernel> _kernel;
    GCFillBorderKernel         _border_handler;
    GCTensorShiftKernel        _shift_handler;
    GCActivationLayer          _activationlayer_function;

    bool _is_activationlayer_enabled;
};
}
#endif /*__ARM_COMPUTE_GCDEPTHWISECONVOLUTION_H__ */
