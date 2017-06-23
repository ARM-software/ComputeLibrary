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
#ifndef __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H__
#define __ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON interface for Direct Convolution Layer kernel */
class NEDirectConvolutionLayerKernel : public INEKernel
{
public:
    /** Default constructor */
    NEDirectConvolutionLayerKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerKernel(const NEDirectConvolutionLayerKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEDirectConvolutionLayerKernel &operator=(const NEDirectConvolutionLayerKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerKernel(NEDirectConvolutionLayerKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEDirectConvolutionLayerKernel &operator=(NEDirectConvolutionLayerKernel &&) = default;
    /** Default destructor */
    ~NEDirectConvolutionLayerKernel() = default;
    /** Set the input, weights and output tensors.
      *
      * @param[in]  input     Input tensor. Data types supported: QS8/F32.
      * @param[in]  weights   Set of kernels to convolve the input volume.
      *                       The 3rd dimension must be the same as the input's volume 3rd dimension.
      *                       Data type supported: Same as @p input.
      * @param[out] output    Output tensor.
      *                       The 3rd dimensions must be equal to the 4th dimension of the @p kernels tensor. Data types supported: Same as @p input.
      * @param[in]  conv_info Contains padding and stride information described in @ref PadStrideInfo.
      */
    void configure(const ITensor *input, const ITensor *weights, ITensor *output, const PadStrideInfo &conv_info);

    // Inherited methods overridden:
    void run(const Window &window) override;
    BorderSize border_size() const override;

private:
    const ITensor *_input;
    const ITensor *_weights;
    ITensor       *_output;
    PadStrideInfo  _conv_info;
    BorderSize     _border_size;
    unsigned int   _kernel_size;
    unsigned int   _num_elems_read_per_iteration;
    unsigned int   _num_elems_written_per_iteration;
};
}
#endif /*__ARM_COMPUTE_NEDIRECTCONVOLUTIONLAYERKERNEL_H__ */
