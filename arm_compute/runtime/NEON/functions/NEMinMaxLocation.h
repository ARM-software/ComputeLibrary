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
#ifndef ARM_COMPUTE_NEMINMAXLOCATION_H
#define ARM_COMPUTE_NEMINMAXLOCATION_H

#include "arm_compute/core/IArray.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/IFunction.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NEMinMaxKernel;
class NEMinMaxLocationKernel;
using IImage = ITensor;

/** Basic function to execute min and max location. This function calls the following Neon kernels:
 *
 * -# NEMinMaxKernel
 * -# NEMinMaxLocationKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEMinMaxLocation : public IFunction
{
public:
    /** Constructor */
    NEMinMaxLocation();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLocation(const NEMinMaxLocation &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEMinMaxLocation &operator=(const NEMinMaxLocation &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMinMaxLocation(NEMinMaxLocation &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEMinMaxLocation &operator=(NEMinMaxLocation &&) = delete;
    /** Default destructor */
    ~NEMinMaxLocation();
    /** Initialise the kernel's inputs and outputs.
     *
     * @param[in]  input     Input image. Data types supported: U8/S16/F32.
     * @param[out] min       Minimum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] max       Maximum value of image. Data types supported: S32 if input type is U8/S16, F32 if input type is F32.
     * @param[out] min_loc   (Optional) Array of minimum value locations.
     * @param[out] max_loc   (Optional) Array of maximum value locations.
     * @param[out] min_count (Optional) Number of minimum value encounters.
     * @param[out] max_count (Optional) Number of maximum value encounters.
     */
    void configure(const IImage *input, void *min, void *max,
                   ICoordinates2DArray *min_loc = nullptr, ICoordinates2DArray *max_loc = nullptr,
                   uint32_t *min_count = nullptr, uint32_t *max_count = nullptr);

    // Inherited methods overridden:
    void run() override;

private:
    std::unique_ptr<NEMinMaxKernel>         _min_max;     /**< Kernel that performs min/max */
    std::unique_ptr<NEMinMaxLocationKernel> _min_max_loc; /**< Kernel that extracts min/max locations */
};
}
#endif /*ARM_COMPUTE_NEMINMAXLOCATION_H */
