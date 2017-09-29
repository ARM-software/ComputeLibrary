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
#ifndef __ARM_COMPUTE_NEMEANSTDDEVKERNEL_H__
#define __ARM_COMPUTE_NEMEANSTDDEVKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "support/Mutex.h"

#include <cstdint>

namespace arm_compute
{
class ITensor;
using IImage = ITensor;

/** Interface for the kernel to calculate mean and standard deviation of input image pixels. */
class NEMeanStdDevKernel : public INEKernel
{
public:
    /** Default constructor */
    NEMeanStdDevKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDevKernel(const NEMeanStdDevKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDevKernel &operator=(const NEMeanStdDevKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEMeanStdDevKernel(NEMeanStdDevKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEMeanStdDevKernel &operator=(NEMeanStdDevKernel &&) = default;
    /** Default destructor */
    ~NEMeanStdDevKernel() = default;

    /** Initialise the kernel's input and outputs.
     *
     * @param[in]  input              Input image. Data type supported: U8.
     * @param[out] mean               Input average pixel value.
     * @param[out] global_sum         Keeps global sum of pixel values.
     * @param[out] stddev             (Optional) Output standard deviation of pixel values.
     * @param[out] global_sum_squared (Optional if stddev is not set, required if stddev is set) Keeps global sum of squared pixel values.
     */
    void configure(const IImage *input, float *mean, uint64_t *global_sum, float *stddev = nullptr, uint64_t *global_sum_squared = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

    BorderSize border_size() const override;

private:
    const IImage      *_input;
    float             *_mean;
    float             *_stddev;
    uint64_t          *_global_sum;
    uint64_t          *_global_sum_squared;
    arm_compute::Mutex _mtx;
    BorderSize         _border_size;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEMEANSTDDEVKERNEL_H__ */
