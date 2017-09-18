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
#ifndef __ARM_COMPUTE_CLFASTCORNERS_H__
#define __ARM_COMPUTE_CLFASTCORNERS_H__

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/CL/kernels/CLFastCornersKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/CL/CLArray.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLNonMaximaSuppression3x3.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to execute fast corners. This function calls the following CL kernels:
 *
 * -# @ref CLFastCornersKernel
 * -# @ref CLNonMaximaSuppression3x3Kernel (executed if nonmax_suppression == true)
 * -# @ref CLCopyToArrayKernel
 *
 */
class CLFastCorners : public IFunction
{
public:
    /** Constructor */
    CLFastCorners(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLFastCorners(const CLFastCorners &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    const CLFastCorners &operator=(const CLFastCorners &) = delete;
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in]     input                 Source image. Data types supported: U8.
     * @param[in]     threshold             Threshold on difference between intensity of the central pixel and pixels on Bresenham's circle of radius 3.
     * @param[in]     nonmax_suppression    If true, non-maximum suppression is applied to detected corners before being placed in the array.
     * @param[out]    corners               Array of keypoints to store the results.
     * @param[in,out] num_corners           Record number of corners in the array
     * @param[in]     border_mode           Strategy to use for borders.
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(const ICLImage *input, float threshold, bool nonmax_suppression, CLKeyPointArray *corners, unsigned int *num_corners,
                   BorderMode border_mode, uint8_t constant_border_value = 0);
    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup             _memory_group;
    CLFastCornersKernel       _fast_corners_kernel;
    CLNonMaximaSuppression3x3 _suppr_func;
    CLCopyToArrayKernel       _copy_array_kernel;
    CLImage                   _output;
    CLImage                   _suppr;
    Window                    _win;
    bool                      _non_max;
    unsigned int             *_num_corners;
    cl::Buffer                _num_buffer;
    CLKeyPointArray          *_corners;
    uint8_t                   _constant_border_value;
};
}
#endif /*__ARM_COMPUTE_CLFASTCORNERS_H__ */
