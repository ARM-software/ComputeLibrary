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
#ifndef __ARM_COMPUTE_CLDEPTHWISEVECTORTOTENSORKERNEL_H__
#define __ARM_COMPUTE_CLDEPTHWISEVECTORTOTENSORKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the depthwise vector to tensor kernel.
 *
 *  This kernel takes the 1D tensor that's been produced by the MatrixVectorMultiply
 *  kernel and reshapes it to given width and height (previously calculated, based
 *  on input/weights dimensions and convolution strides and padding).
 *
 **/
class CLDepthwiseVectorToTensorKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLDepthwiseVectorToTensorKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseVectorToTensorKernel(const CLDepthwiseVectorToTensorKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthwiseVectorToTensorKernel &operator=(const CLDepthwiseVectorToTensorKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLDepthwiseVectorToTensorKernel(CLDepthwiseVectorToTensorKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLDepthwiseVectorToTensorKernel &operator=(CLDepthwiseVectorToTensorKernel &&) = default;
    /** Set the input and output of the kernel.
     *
     * @param[in]  input  The input vector to convert. Data type supported: QASYMM8/S32/F16/F32.
     * @param[out] output The output tensor. 3 lower dimensions represent a single input [width, height, IFM]. Data type supported: same as @p input.
     * @param[in]  conv_w The converted tensor's width.
     * @param[in]  conv_h The converted tensor's height.
     */
    void configure(const ICLTensor *input, ICLTensor *output, size_t conv_w, size_t conv_h);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLDEPTHWISEVECTORTOTENSORKERNEL_H__ */
