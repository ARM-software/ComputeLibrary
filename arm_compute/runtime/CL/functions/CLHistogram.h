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
#ifndef __ARM_COMPUTE_CLHISTOGRAM_H__
#define __ARM_COMPUTE_CLHISTOGRAM_H__

#include "arm_compute/core/CL/kernels/CLHistogramKernel.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLDistribution1D;
class ICLTensor;
using ICLTensor = ICLImage;

/** Basic function to execute histogram. This function calls the following OpenCL kernels:
 *
 *  -# @ref CLHistogramKernel
 *  -# @ref CLHistogramBorderKernel
 *
 */
class CLHistogram : public IFunction
{
public:
    /*
     * @ Default constructor
     */
    CLHistogram();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHistogram(const CLHistogram &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    const CLHistogram &operator=(const CLHistogram &) = delete;
    /** Initialize the function
     *
     * @param[in]  input  Source image. Data types supported: U8
     * @param[out] output Output distribution.
     */
    void configure(const ICLImage *input, ICLDistribution1D *output);

    // Inherited methods overridden:
    void run() override;

private:
    CLHistogramKernel       _kernel;        /**< kernel to run */
    CLHistogramBorderKernel _kernel_border; /**< Border kernel to run */
};
}
#endif /*__ARM_COMPUTE_CLHISTOGRAM_H__ */
