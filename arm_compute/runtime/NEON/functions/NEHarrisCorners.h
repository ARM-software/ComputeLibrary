/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_NEHARRISCORNERS_H
#define ARM_COMPUTE_NEHARRISCORNERS_H

#include "arm_compute/core/CPP/kernels/CPPCornerCandidatesKernel.h"
#include "arm_compute/core/CPP/kernels/CPPSortEuclideanDistanceKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Array.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/NEON/functions/NENonMaximaSuppression3x3.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstdint>
#include <memory>

namespace arm_compute
{
class ITensor;
class NEFillBorderKernel;
class INEHarrisScoreKernel;
using IImage = ITensor;

/** Basic function to execute harris corners detection. This function calls the following NEON kernels and functions:
 *
 * -# @ref NESobel3x3 (if gradient_size == 3) or<br/>
 *    @ref NESobel5x5 (if gradient_size == 5) or<br/>
 *    @ref NESobel7x7 (if gradient_size == 7)
 * -# @ref NEFillBorderKernel
 * -# NEHarrisScoreKernel<3> (if block_size == 3) or<br/>
 *    NEHarrisScoreKernel<5> (if block_size == 5) or<br/>
 *    NEHarrisScoreKernel<7> (if block_size == 7)
 * -# @ref NENonMaximaSuppression3x3
 * -# @ref CPPCornerCandidatesKernel
 * -# @ref CPPSortEuclideanDistanceKernel
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
 */
class NEHarrisCorners : public IFunction
{
public:
    /** Constructor
     *
     * Initialize _sobel, _harris_score and _corner_list to nullptr.
     *
     * @param[in] memory_manager (Optional) Memory manager.
     */
    NEHarrisCorners(std::shared_ptr<IMemoryManager> memory_manager = nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHarrisCorners(const NEHarrisCorners &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEHarrisCorners &operator=(const NEHarrisCorners &) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHarrisCorners(NEHarrisCorners &&) = delete;
    /** Prevent instances of this class from being moved (As this class contains non movable objects) */
    NEHarrisCorners &operator=(NEHarrisCorners &&) = delete;
    /** Default destructor */
    ~NEHarrisCorners();
    /** Initialize the function's source, destination, conv and border_mode.
     *
     * @param[in, out] input                 Source image. Data type supported: U8. (Written to only for @p border_mode != UNDEFINED)
     * @param[in]      threshold             Minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
     * @param[in]      min_dist              Radial Euclidean distance for the euclidean diatance stage
     * @param[in]      sensitivity           Sensitivity threshold k from the Harris-Stephens equation
     * @param[in]      gradient_size         The gradient window size to use on the input. The implementation supports 3, 5, and 7
     * @param[in]      block_size            The block window size used to compute the Harris Corner score. The implementation supports 3, 5, and 7.
     * @param[out]     corners               Array of keypoints to store the results.
     * @param[in]      border_mode           Border mode to use
     * @param[in]      constant_border_value (Optional) Constant value to use for borders if border_mode is set to CONSTANT.
     */
    void configure(IImage *input, float threshold, float min_dist, float sensitivity,
                   int32_t gradient_size, int32_t block_size, KeyPointArray *corners,
                   BorderMode border_mode, uint8_t constant_border_value = 0);

    // Inherited methods overridden:
    void run() override;

private:
    MemoryGroup                           _memory_group;          /**< Function's memory group */
    std::unique_ptr<IFunction>            _sobel;                 /**< Sobel function */
    std::unique_ptr<INEHarrisScoreKernel> _harris_score;          /**< Harris score kernel */
    NENonMaximaSuppression3x3             _non_max_suppr;         /**< Non-maxima suppression function */
    CPPCornerCandidatesKernel             _candidates;            /**< Sort kernel */
    CPPSortEuclideanDistanceKernel        _sort_euclidean;        /**< Euclidean distance kernel */
    std::unique_ptr<NEFillBorderKernel>   _border_gx;             /**< Border handler before running harris score */
    std::unique_ptr<NEFillBorderKernel>   _border_gy;             /**< Border handler before running harris score */
    Image                                 _gx;                    /**< Source image - Gx component */
    Image                                 _gy;                    /**< Source image - Gy component */
    Image                                 _score;                 /**< Source image - Harris score */
    Image                                 _nonmax;                /**< Source image - Non-Maxima suppressed image */
    std::vector<InternalKeypoint>         _corners_list;          /**< Array of InternalKeypoint. It stores the potential corner candidates */
    int32_t                               _num_corner_candidates; /**< Number of potential corner candidates */
};
}
#endif /*ARM_COMPUTE_NEHARRISCORNERS_H */
