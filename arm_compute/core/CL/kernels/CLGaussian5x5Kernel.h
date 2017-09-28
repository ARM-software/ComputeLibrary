/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGAUSSIAN5X5KERNEL_H__
#define __ARM_COMPUTE_CLGAUSSIAN5X5KERNEL_H__

#include "arm_compute/core/CL/kernels/CLConvolutionKernel.h"

namespace arm_compute
{
class ICLTensor;

/** Interface for the kernel to run the horizontal pass of 5x5 Gaussian filter on a tensor. */
class CLGaussian5x5HorKernel : public CLSeparableConvolution5x5HorKernel
{
public:
    /** Initialise the kernel's source, destination and border.
     *
     * @param[in]  input            Source tensor. Data types supported: U8.
     * @param[out] output           Destination tensor. Data types supported: S16.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, bool border_undefined);

private:
    //Make the configure method of the parent class private
    using CLSeparableConvolution5x5HorKernel::configure;
};

/** Interface for the kernel to run the vertical pass of 5x5 Gaussian filter on a tensor. */
class CLGaussian5x5VertKernel : public CLSeparableConvolution5x5VertKernel
{
public:
    /** Initialise the kernel's source, destination and border.
     *
     * @param[in]  input            Input tensor(output of horizontal pass). Data types supported: S16.
     * @param[out] output           Destination tensor. Data types supported: U8.
     * @param[in]  border_undefined True if the border mode is undefined. False if it's replicate or constant.
     */
    void configure(const ICLTensor *input, ICLTensor *output, bool border_undefined);

private:
    //Make the configure method of the parent class private
    using CLSeparableConvolution5x5VertKernel::configure;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGAUSSIAN5X5KERNEL_H__ */
