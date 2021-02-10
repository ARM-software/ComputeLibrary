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
#ifndef ARM_COMPUTE_NEFASTCORNERS_H
#define ARM_COMPUTE_NEFASTCORNERS_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NENonMaximaSuppression3x3Kernel;
class NEFastCornersKernel;
class NEFillBorderKernel;
class NEFillArrayKernel;
using IImage = ITensor;

/** Basic function to execute fast corners. This function call the following Neon kernels:
 *
 * -# @ref NEFastCornersKernel
 * -# @ref NENonMaximaSuppression3x3Kernel (executed if nonmax_suppression == true)
 * -# @ref NEFillArrayKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEFastCorners : public IFunction
{
public:
    /** Constructor */
    NEFastCorners(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFastCorners(const NEFastCorners &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEFastCorners &operator=(const NEFastCorners &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFastCorners(NEFastCorners &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEFastCorners &operator=(NEFastCorners &&) = delete;
    /** Default destructor */
    ~NEFastCorners();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in, out] input                 Source image. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]      threshold             Threshold on difference between intensity of the central pixel and pixels on Bresenham's circle of radius 3.
     * @param[in]      nonmax_suppression    If true, non-maximum suppression is applied to detected corners before being placed in the array.
     * @param[out]     corners               Array of keypoints to store the results.
     * @param[in]      border_mode           Strategy to use for borders.
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(IImage *input, float threshold, bool nonmax_suppression, KeyPointArray *corners,
                   BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                                      _memory_group;
    std::unique_ptr<NEFastCornersKernel>             _fast_corners_kernel;
    std::unique_ptr<NEFillBorderKernel>              _border_handler;
    std::unique_ptr<NENonMaximaSuppression3x3Kernel> _nonmax_kernel;
    std::unique_ptr<NEFillArrayKernel>               _fill_kernel;
    Image                                            _output;
    Image                                            _suppressed;
    bool                                             _non_max;
};
}
#endif /*ARM_COMPUTE_NEFASTCORNERS_H */
