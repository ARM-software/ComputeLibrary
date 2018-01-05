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
#ifndef __ARM_COMPUTE_CLHARRISCORNERS_H__
#define __ARM_COMPUTE_CLHARRISCORNERS_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/kernels/CLFillBorderKernel.h"
#include "arm_compute/core/CL/kernels/CLHarrisCornersKernel.h"
#include "arm_compute/core/NEON/kernels/NEHarrisCornersKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLMemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLNonMaximaSuppression3x3.h"
#include "arm_compute/runtime/IMemoryManager.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to execute harris corners detection. This function calls the following CL and NEON kernels and functions:
 *
 * @note Requires CPU support for the kernels: CPPCornerCandidatesKernel and CPPSortEuclideanDistanceKernel.
 *
 * -# @ref CLSobel3x3 (if gradient_size == 3) or<br/>
 *    @ref CLSobel5x5 (if gradient_size == 5) or<br/>
 *    @ref CLSobel7x7 (if gradient_size == 7)
 * -# @ref CLFillBorderKernel
 * -# @ref CLHarrisScoreKernel
 * -# @ref CLNonMaximaSuppression3x3
 * -# @ref CPPCornerCandidatesKernel
 * -# @ref CPPSortEuclideanDistanceKernel
 */
class CLHarrisCorners : public IFunction
{
public:
    /** Constructor */
    CLHarrisCorners(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLHarrisCorners(const CLHarrisCorners &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    const CLHarrisCorners &operator=(const CLHarrisCorners &) = delete;
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in,out] input                 Source image. Data types supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]     threshold             Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
     * @param[in]     min_dist              Radial Euclidean distance for the euclidean distance stage.
     * @param[in]     sensitivity           Sensitivity threshold k from the Harris-Stephens equation
     * @param[in]     gradient_size         The gradient window size to use on the input. The implementation supports 3, 5, and 7
     * @param[in]     block_size            The block window size used to compute the Harris Corner score. The implementation supports 3, 5, and 7.
     * @param[out]    corners               Array of keypoints to store the results.
     * @param[in]     border_mode           Border mode to use
     * @param[in]     constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     * @param[in]     use_fp16              (Optional) If true the FP16 kernels will be used. If false F32 kernels are used.
     */
    void configure(ICLImage *input, float threshold, float min_dist, float sensitivity,
                   int32_t gradient_size, int32_t block_size, ICLKeyPointArray *corners,
                   BorderMode border_mode, uint8_t constant_border_value = 0, bool use_fp16 = false);

    // Inherited methods overridden:
    void run() override;

private:
    CLMemoryGroup                       _memory_group;          /**< Function's memory group */
    std::unique_ptr<IFunction>          _sobel;                 /**< Sobel function */
    CLHarrisScoreKernel                 _harris_score;          /**< Harris score kernel */
    CLNonMaximaSuppression3x3           _non_max_suppr;         /**< Non-maxima suppression function */
    CPPCornerCandidatesKernel           _candidates;            /**< Sort kernel */
    CPPSortEuclideanDistanceKernel      _sort_euclidean;        /**< Euclidean distance kernel */
    CLFillBorderKernel                  _border_gx;             /**< Border handler before running harris score */
    CLFillBorderKernel                  _border_gy;             /**< Border handler before running harris score */
    CLImage                             _gx;                    /**< Source image - Gx component */
    CLImage                             _gy;                    /**< Source image - Gy component */
    CLImage                             _score;                 /**< Source image - Harris score */
    CLImage                             _nonmax;                /**< Source image - Non-Maxima suppressed image */
    std::unique_ptr<InternalKeypoint[]> _corners_list;          /**< Array of InternalKeypoint. It stores the potential corner candidates */
    int32_t                             _num_corner_candidates; /**< Number of potential corner candidates */
    ICLKeyPointArray                   *_corners;               /**< Output corners array */
};
}
#endif /*__ARM_COMPUTE_CLHARRISCORNERS_H__ */
