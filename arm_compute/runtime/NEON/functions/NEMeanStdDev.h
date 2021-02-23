/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEMEANSTDDEV_H
#define ARM_COMPUTE_NEMEANSTDDEV_H

#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include <memory>

#include <cstdint>

namespace arm_compute
{
class NEMeanStdDevKernel;
class NEFillBorderKernel;

/** Basic function to execute mean and std deviation. This function calls the following Neon kernels:
 *
 * @ref NEMeanStdDevKernel
 *
 */
class NEMeanStdDev : public IFunction
{
public:
    /** Default Constructor. */
    NEMeanStdDev();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDev(const NEMeanStdDev &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMeanStdDev &operator=(const NEMeanStdDev &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMeanStdDev(NEMeanStdDev &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMeanStdDev &operator=(NEMeanStdDev &&) = delete;
    /** Default destructor */
    ~NEMeanStdDev();
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in, out] input  Input image. Data types supported: U8. (Written to only for border filling)
     * @param[out]     mean   Output average pixel value.
     * @param[out]     stddev (Optional) Output standard deviation of pixel values.
     */
    void configure(IImage *input, float *mean, float *stddev = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEMeanStdDevKernel> _mean_stddev_kernel; /**< Kernel that standard deviation calculation. */
    std::unique_ptr<NEFillBorderKernel> _fill_border_kernel; /**< Kernel that fills tensor's borders with zeroes. */
    uint64_t                            _global_sum;         /**< Variable that holds the global sum among calls in order to ease reduction */
    uint64_t                            _global_sum_squared; /**< Variable that holds the global sum of squared values among calls in order to ease reduction */
};
}
#endif /*ARM_COMPUTE_NEMEANSTDDEV_H */
