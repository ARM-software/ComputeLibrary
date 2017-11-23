/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
 * Deconvolution Layer is the backward pass of Convolution Layer. First we transform the input depending on the stride and pad info and then perfrom a 1x1
 * convolution pass. Input stride defines how many zeroes we should put between each element of the input, pad is the amount of padding and finaly a is a user
 * specified value where a < stride - 1 that increases the padding top and right of the input image.
 *
 *  The relation between input to output is as follows:
 *       width_output = round((width_input − 1) ∗ (stride_x - 1) − 2 ∗ padding_x + kernel_x + inner_border_right )
 *       height_output = round((height_input − 1) ∗ (stride_y - 1) − 2 ∗ padding_y + kernel_y + inner_border_top )
 *
 *  where
 *      width is the size of the first input dimension.
 *      height is the size of the second input dimension.
 *      width_output is the size of the first output dimension.
 *      height_output is the size of the second output dimension.
 *      kernel_x and kernel_y are the convolution sizes in x and y.
 *      inner_border_right and inner_border_top the number of zeros added to the top and right edges of the input.
 *      stride_x and stride_y is the input stride of the first and second dimension.
 *
 *  This function calls the following NEON kernels:
 *
 * -# @ref NEDirectConvolutionLayer
 *
 */
class NEDeconvolutionLayer : public IFunction
{
public:
    /** Default constructor */
    NEDeconvolutionLayer(std::shared_ptr<IMemoryManager> memory_manager = nullptr);

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayer(const NEDeconvolutionLayer &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDeconvolutionLayer &operator=(const NEDeconvolutionLayer &) = delete;
    /** Allow instances of this class to be moved */
    NEDeconvolutionLayer(NEDeconvolutionLayer &&) = default;
    /** Allow instances of this class to be moved */
    NEDeconvolutionLayer &operator=(NEDeconvolutionLayer &&) = default;
    /** Default destructor */
    virtual ~NEDeconvolutionLayer() = default;
    /** Set the input, weights, biases and output tensors.
     *
     * @param[in,out] input              Input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Data types supported: F32.
     * @param[in]     weights            The 4d weights with dimensions [width, height, OFM, IFM]. Data type supported: Same as @p input.
     * @param[in]     bias               Optional, ignored if NULL. The biases have one dimension. Data type supported: Same as @p input.
     * @param[out]    output             Output tensor. The output has the same number of dimensions as the @p input.
     * @param[in]     info               Contains padding and policies to be used in the deconvolution, this is decribed in @ref PadStrideInfo.
     * @param[in]     inner_border_right The number of zeros added to right edge of the input.
     * @param[in]     inner_border_top   The number of zeros added to top edge of the input.
     *
     */
    void configure(ITensor *input, const ITensor *weights, const ITensor *bias, ITensor *output, const PadStrideInfo &info,
                   unsigned int inner_border_right, unsigned int inner_border_top);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup              _memory_group;
    NEDirectConvolutionLayer _conv_f;
    Tensor                   _scaled_output;
    ITensor                 *_input;
    PadStrideInfo            _info;
    std::pair<unsigned int, unsigned int> _inner_border;
};
} // arm_compute
#endif /* __ARM_COMPUTE_NEDECONVOLUTIONLAYER_H__ */
