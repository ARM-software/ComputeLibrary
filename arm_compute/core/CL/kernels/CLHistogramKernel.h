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
#ifndef __ARM_COMPUTE_CLHISTOGRAMKERNEL_H__
#define __ARM_COMPUTE_CLHISTOGRAMKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLDistribution1D;
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface to run the histogram kernel. This kernel processes the part of image with width can be divided by 16.
 *  If the image width is not a multiple of 16, remaining pixels have to be processed with the @ref CLHistogramBorderKernel
 */
class CLHistogramKernel : public ICLKernel
{
public:
    /** Constructor */
    CLHistogramKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHistogramKernel(const CLHistogramKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHistogramKernel &operator=(const CLHistogramKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHistogramKernel(CLHistogramKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHistogramKernel &operator=(CLHistogramKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input  Source image. Data types supported: U8.
     * @param[out] output Destination distribution.
     */
    void configure(const ICLImage *input, ICLDistribution1D *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLImage    *_input;
    ICLDistribution1D *_output;
};

/** Interface to run the histogram kernel to handle the leftover part of image
 *
 */
class CLHistogramBorderKernel : public ICLKernel
{
public:
    /** Constructor */
    CLHistogramBorderKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHistogramBorderKernel(const CLHistogramBorderKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHistogramBorderKernel &operator=(const CLHistogramBorderKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLHistogramBorderKernel(CLHistogramBorderKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLHistogramBorderKernel &operator=(CLHistogramBorderKernel &&) = default;
    /** Initialise the kernel's input, output and border mode.
     *
     * @param[in]  input  Source image. Data types supported: U8.
     * @param[out] output Destination distribution.
     */
    void configure(const ICLImage *input, ICLDistribution1D *output);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLImage    *_input;
    ICLDistribution1D *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLHISTOGRAMKERNEL_H__*/
