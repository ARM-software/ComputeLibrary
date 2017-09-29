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
#ifndef __ARM_COMPUTE_NEFASTCORNERS_H__
#define __ARM_COMPUTE_NEFASTCORNERS_H__

#include "arm_compute/core/NEON/kernels/NEFastCornersKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillArrayKernel.h"
#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NENonMaximaSuppression3x3Kernel.h"
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
using IImage = ITensor;

/** Basic function to execute fast corners. This function call the following NEON kernels:
 *
 * -# @ref NEFastCornersKernel
 * -# @ref NENonMaximaSuppression3x3Kernel (executed if nonmax_suppression == true)
 * -# @ref NEFillArrayKernel
 *
 */
class NEFastCorners : public IFunction
{
public:
    /** Constructor */
    NEFastCorners(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
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
    MemoryGroup                     _memory_group;
    NEFastCornersKernel             _fast_corners_kernel;
    NEFillBorderKernel              _border_handler;
    NENonMaximaSuppression3x3Kernel _nonmax_kernel;
    NEFillArrayKernel               _fill_kernel;
    Image                           _output;
    Image                           _suppressed;
    bool                            _non_max;
};
}
#endif /*__ARM_COMPUTE_NEFASTCORNERS_H__ */
