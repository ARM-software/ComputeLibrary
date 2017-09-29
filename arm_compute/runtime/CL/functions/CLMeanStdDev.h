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
#ifndef __ARM_COMPUTE_CLMEANSTDDEV_H__
#define __ARM_COMPUTE_CLMEANSTDDEV_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLMeanStdDevKernel.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
/** Basic function to execute mean and standard deviation by calling @ref CLMeanStdDevKernel */
class CLMeanStdDev : public IFunction
{
public:
    /** Default Constructor. */
    CLMeanStdDev();
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in, out] input  Input image. Data types supported: U8. (Written to only for border filling)
     * @param[out]     mean   Output average pixel value.
     * @param[out]     stddev (Optional)Output standard deviation of pixel values.
     */
    void configure(ICLImage *input, float *mean, float *stddev = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    CLMeanStdDevKernel _mean_stddev_kernel; /**< Kernel that standard deviation calculation. */
    CLFillBorderKernel _fill_border_kernel; /**< Kernel that fills the border with zeroes. */
    cl::Buffer         _global_sum;         /**< Variable that holds the global sum among calls in order to ease reduction */
    cl::Buffer         _global_sum_squared; /**< Variable that holds the global sum of squared values among calls in order to ease reduction */
};
}
#endif /*__ARM_COMPUTE_CLMEANSTDDEV_H__ */
