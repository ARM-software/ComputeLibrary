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
#ifndef __ARM_COMPUTE_NEDECONVOLUTIONLAYER_H__
#define __ARM_COMPUTE_NEDECONVOLUTIONLAYER_H__

#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayerUpsample.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <memory>

namespace arm_compute
{
/** Function to run the deconvolution layer.
 *
 *  The operation is similar to convolution but it's implemented by up-sampling the inputs with zeros insertions between the inputs and convolving
 *  the kernels on the up-sampled result.
 *
 *  Before the Deconvolution is done, up-scaling the first 2D with zeros is performed. The relation between input to
 *  output is as follows:
 *      width_output = round((width_input − 1) ∗ upscale_x − 2 ∗ padding_x + kernel_x + a_x )
 *      height_output = round((height_input − 1) ∗ upscale_y − 2 ∗ padding_y + kernel_y + a_y )
 *
 *  where
 *      width is the size of the first input dimension.
 *      height is the size of the second input dimension.
 *      width_output is the size of the first output dimension.
 *      height_output is the size of the second output dimension.
 *      kernel_x and kernel_y are the convolution sizes in x and y.
 *      ax and ay the number of zeros added to the top and right edges of the input.
 *      upscale_x and upscale_y how much to scale the X and Y axis.
 *
 *  This function calls the following NEON kernels:
 *
 * -# @ref NEDeconvolutionLayerUpsampleKernel
 * -# @ref NEDirectConvolutionLayer
 *
 */
class NEDeconvolutionLayer : public IFunction
{
public:
    /** Constructor */
    NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Set the input, weights, biases and output tensors.
     *
     * @param[in,out] input    Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: F32.
     * @param[in]     weights  The 4d weights with dimensions [width, height, OFM, IFM]. Data type supported: Same as @p input.
     * @param[in]     bias     Optional, ignored if NULL. The biases have one dimension. Data type supported: Same as @p input.
     * @param[out]    output   Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info     Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     * @param[in]     ax       The number of zeros added to right edge of the input.
     * @param[in]     ay       The number of zeros added to top edge of the input.
     * @param[in]     upscalex How much to scale the X axis.
     * @param[in]     upscaley How much to scale the Y axis.
     *
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info,
                   unsigned int ax, unsigned int ay, float upscalex, float upscaley);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                  _memory_group;
    NEDeconvolutionLayerUpsample _scale_f;
    NEDirectConvolutionLayer     _conv_f;
    Tensor                       _scaled_output;
};
} // arm_compute
#endif /* __ARM_COMPUTE_NEDECONVOLUTIONLAYER_H__ */
