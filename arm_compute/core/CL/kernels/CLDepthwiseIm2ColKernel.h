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
#ifndef __ARM_COMPUTE_CLDEPTHWISEIM2COLKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHWISEIM2COLKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the depthwise im2col reshape kernel.
 *  This kernel reshape the input low 3 dimensions to a new 3D shape  where the output's first dimension is
 *  the linear patch size (FILTER_WIDTH * FILTER_HEIGHT) and second dimension is number of patches in per image and third dimension unchanged .
 **/
class CLDepthwiseIm2ColKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseIm2ColKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseIm2ColKernel(const CLDepthwiseIm2ColKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseIm2ColKernel &operator=(const CLDepthwiseIm2ColKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDepthwiseIm2ColKernel(CLDepthwiseIm2ColKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDepthwiseIm2ColKernel &operator=(CLDepthwiseIm2ColKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs. Data types supported: F32
     * @param[out] output      The output tensor. First 3 lower dimensions represent a transform of each 3D input,
     *                         while every dimension above 3 represents a batch. Data types supported: Same as @p input
     * @param[in]  kernel_dims The kernel dimensions (width and height).
     * @param[in]  conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias    Boolean that specifies if the depthwise convolution has bias.
     */
    void configure(const ICLTensor *input, ICLTensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info, bool has_bias = false);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHWISEIM2COLKERNEL_H__ */
