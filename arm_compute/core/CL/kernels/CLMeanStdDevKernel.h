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
#ifndef __ARM_COMPUTE_CLMEANSTDDEVKERNEL_H__
#define __ARM_COMPUTE_CLMEANSTDDEVKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace cl
{
class Buffer;
}

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for the kernel to calculate mean and standard deviation of input image pixels. */
class CLMeanStdDevKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLMeanStdDevKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMeanStdDevKernel(const CLMeanStdDevKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMeanStdDevKernel &operator=(const CLMeanStdDevKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMeanStdDevKernel(CLMeanStdDevKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMeanStdDevKernel &operator=(CLMeanStdDevKernel &&) = default;
    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input              Input image. Data types supported: U8.
     * @param[out] mean               Input average pixel value.
     * @param[out] global_sum         Keeps global sum of pixel values (Buffer size: 1 cl_ulong).
     * @param[out] stddev             (Optional) Output standard deviation of pixel values.
     * @param[out] global_sum_squared (Optional if stddev is not set, required if stddev is set) Keeps global sum of squared pixel values (Buffer size: 1 cl_ulong).
     */
    void configure(const ICLImage *input, float *mean, cl::Buffer *global_sum, float *stddev = nullptr, cl::Buffer *global_sum_squared = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

    BorderSize border_size() const override;

private:
    const ICLImage *_input;
    float          *_mean;
    float          *_stddev;
    cl::Buffer     *_global_sum;
    cl::Buffer     *_global_sum_squared;
    BorderSize      _border_size;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLMEANSTDDEVKERNEL_H__ */
