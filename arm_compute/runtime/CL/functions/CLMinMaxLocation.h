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
#ifndef __ARM_COMPUTE_CLMINMAXLOCATION_H__
#define __ARM_COMPUTE_CLMINMAXLOCATION_H__

#include "arm_compute/core/CL/kernels/CLMinMaxLocationKernel.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to execute min and max location. This function calls the following OpenCL kernels:
 *
 * -# @ref CLMinMaxKernel
 * -# @ref CLMinMaxLocationKernel
 */
class CLMinMaxLocation : public IFunction
{
public:
    /** Constructor */
    CLMinMaxLocation();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLocation(const CLMinMaxLocation &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLMinMaxLocation &operator=(const CLMinMaxLocation &) = delete;
    /** Allow instances of this class to be moved */
    CLMinMaxLocation(CLMinMaxLocation &&) = default;
    /** Allow instances of this class to be moved */
    CLMinMaxLocation &operator=(CLMinMaxLocation &&) = default;
    /** Initialise the kernel's inputs and outputs.
     *
     * @note When locations of min and max occurrences are requested, the reported number of locations is limited to the given array size.
     *
     * @param[in]  input     Input image. Data types supported: U8/S16/F32.
     * @param[out] min       Minimum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] max       Maximum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] min_loc   (Optional) Array of Coordinates2D used to store minimum value locations.
     * @param[out] max_loc   (Optional) Array of Coordinates2D used to store maximum value locations.
     * @param[out] min_count (Optional) Number of minimum value encounters.
     * @param[out] max_count (Optional) Number of maximum value encounters.
     */
    void configure(const ICLImage *input, void *min, void *max,
                   CLCoordinates2DArray *min_loc = nullptr, CLCoordinates2DArray *max_loc = nullptr,
                   uint32_t *min_count = nullptr, uint32_t *max_count = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    CLMinMaxKernel         _min_max_kernel;     /**< Kernel that performs min/max */
    CLMinMaxLocationKernel _min_max_loc_kernel; /**< Kernel that counts min/max occurrences and identifies their positions */
    cl::Buffer             _min_max_vals;       /**< Buffer to collect min, max values */
    cl::Buffer             _min_max_count_vals; /**< Buffer to collect min, max values */
    void                  *_min;                /**< Minimum value. */
    void                  *_max;                /**< Maximum value. */
    uint32_t              *_min_count;          /**< Minimum value occurrences. */
    uint32_t              *_max_count;          /**< Maximum value occurrences. */
    CLCoordinates2DArray *_min_loc;             /**< Minimum value occurrences coordinates. */
    CLCoordinates2DArray *_max_loc;             /**< Maximum value occurrences  coordinates. */
};
}
#endif /*__ARM_COMPUTE_CLMINMAXLOCATION_H__ */
