/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

namespace arm_compute
{
/** Basic function to execute mean and standard deviation by calling @ref CLMeanStdDevKernel */
class CLMeanStdDev : public IFunction
{
public:
    /** Default Constructor. */
    CLMeanStdDev(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMeanStdDev(const CLMeanStdDev &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMeanStdDev &operator=(const CLMeanStdDev &) = delete;
    /** Allow instances of this class to be moved */
    CLMeanStdDev(CLMeanStdDev &&) = default;
    /** Allow instances of this class to be moved */
    CLMeanStdDev &operator=(CLMeanStdDev &&) = default;
    /** Default destructor */
    ~CLMeanStdDev() = default;
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in, out] input  Input image. Data types supported: U8/F16/F32. (Written to only for border filling)
     * @param[out]     mean   Output average pixel value.
     * @param[out]     stddev (Optional) Output standard deviation of pixel values.
     */
    void configure(ICLImage *input, float *mean, float *stddev = nullptr);
    /** Static function to check if given info will lead to a valid configuration of @ref CLMeanStdDev
     *
     * @param[in] input  Input image. Data types supported: U8/F16/F32.
     * @param[in] mean   Output average pixel value.
     * @param[in] stddev (Optional) Output standard deviation of pixel values.
     *
     * @return a status
     */
    static Status validate(ITensorInfo *input, float *mean, float *stddev = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    template <typename T>
    void run_float();
    void run_int();

    CLMemoryGroup        _memory_group;               /**< Function's memory group */
    DataType             _data_type;                  /**< Input data type. */
    unsigned int         _num_pixels;                 /**< Number of image's pixels. */
    bool                 _run_stddev;                 /**< Flag for knowing if we should run stddev reduction function. */
    CLReductionOperation _reduction_operation_mean;   /**< Reduction operation function for computing mean value. */
    CLReductionOperation _reduction_operation_stddev; /**< Reduction operation function for computing standard deviation. */
    CLTensor             _reduction_output_mean;      /**< Reduction operation output tensor for mean value. */
    CLTensor             _reduction_output_stddev;    /**< Reduction operation output tensor for standard deviation value. */
    float               *_mean;                       /**< Pointer that holds the mean value. */
    float               *_stddev;                     /**< Pointer that holds the standard deviation value. */
    CLMeanStdDevKernel   _mean_stddev_kernel;         /**< Kernel that standard deviation calculation. */
    CLFillBorderKernel   _fill_border_kernel;         /**< Kernel that fills the border with zeroes. */
    cl::Buffer           _global_sum;                 /**< Variable that holds the global sum among calls in order to ease reduction */
    cl::Buffer           _global_sum_squared;         /**< Variable that holds the global sum of squared values among calls in order to ease reduction */
};
}
#endif /*__ARM_COMPUTE_CLMEANSTDDEV_H__ */
