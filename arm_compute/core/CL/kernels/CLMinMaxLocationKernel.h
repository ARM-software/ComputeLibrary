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
#ifndef __ARM_COMPUTE_CLMINMAXLOCATIONKERNEL_H__
#define __ARM_COMPUTE_CLMINMAXLOCATIONKERNEL_H__

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLKernel.h"

#include <array>

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Interface for the kernel to perform min max search on an image.
 */
class CLMinMaxKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLMinMaxKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxKernel(const CLMinMaxKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxKernel &operator=(const CLMinMaxKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMinMaxKernel(CLMinMaxKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMinMaxKernel &operator=(CLMinMaxKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input   Input Image. Data types supported: U8/S16/F32.
     * @param[out] min_max Buffer of 2 elements to store the min value at position 0 and the max value at position 1. Data type supported: S32 if input type is U8/S16, F32 if input type is F32.
     */
    void configure(const ICLImage *input, cl::Buffer *min_max);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;               /**< Input image. */
    cl::Buffer      *_min_max;             /**< Minimum/maximum value. */
    std::array<int, 2> _data_type_max_min; /**< Maximum and minimum data type value respectively. */
};

/** Interface for the kernel to find min max locations of an image.
 */
class CLMinMaxLocationKernel : public ICLKernel
{
public:
    /** Constructor */
    CLMinMaxLocationKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLocationKernel(const CLMinMaxLocationKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLocationKernel &operator=(const CLMinMaxLocationKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLMinMaxLocationKernel(CLMinMaxLocationKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLMinMaxLocationKernel &operator=(CLMinMaxLocationKernel &&) = default;
    /** Initialise the kernel's input and outputs.
     *
     * @note When locations of min and max occurrences are requested, the reported number of locations is limited to the given array size.
     *
     * @param[in]  input         Input image. Data types supported: U8/S16/F32.
     * @param[out] min_max       Buffer of 2 elements to store the min value at position 0 and the max value at position 1. Data type supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] min_max_count Buffer of 2 elements to store the min value occurrences at position 0 and the max value occurrences at position 1. Data type supported: S32
     * @param[out] min_loc       (Optional) Array of Coordinates2D used to store minimum value locations.
     * @param[out] max_loc       (Optional) Array of Coordinates2D used to store maximum value locations.
     */
    void configure(const ICLImage *input, cl::Buffer *min_max, cl::Buffer *min_max_count,
                   ICLCoordinates2DArray *min_loc = nullptr, ICLCoordinates2DArray *max_loc = nullptr);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLImage *_input;         /**< Input image. */
    cl::Buffer     *_min_max_count; /**< Minimum/maximum value occurrences. */
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLMINMAXLOCATIONKERNEL_H__ */
