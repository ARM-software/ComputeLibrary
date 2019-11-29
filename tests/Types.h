/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_TYPES_H
#define ARM_COMPUTE_TEST_TYPES_H

#include "arm_compute/core/Types.h"

#include <vector>

namespace arm_compute
{
/** Gradient dimension type. */
enum class GradientDimension
{
    GRAD_X,  /**< x gradient dimension */
    GRAD_Y,  /**< y gradient dimension */
    GRAD_XY, /**< x and y gradient dimension */
};

/** Min and max values and locations */
template <typename MinMaxType>
struct MinMaxLocationValues
{
    MinMaxType                 min{};     /**< Min value */
    MinMaxType                 max{};     /**< Max value */
    std::vector<Coordinates2D> min_loc{}; /**< Min value location */
    std::vector<Coordinates2D> max_loc{}; /**< Max value location */
};

/** Parameters of Optical Flow algorithm. */
struct OpticalFlowParameters
{
    OpticalFlowParameters(Termination termination,
                          float       epsilon,
                          size_t      num_iterations,
                          size_t      window_dimension,
                          bool        use_initial_estimate)
        : termination{ std::move(termination) },
          epsilon{ std::move(epsilon) },
          num_iterations{ std::move(num_iterations) },
          window_dimension{ std::move(window_dimension) },
          use_initial_estimate{ std::move(use_initial_estimate) }
    {
    }

    Termination termination;
    float       epsilon;
    size_t      num_iterations;
    size_t      window_dimension;
    bool        use_initial_estimate;
};

/** Internal keypoint class for Lucas-Kanade Optical Flow */
struct InternalKeyPoint
{
    float x{ 0.f };                 /**< x coordinate of the keypoint */
    float y{ 0.f };                 /**< y coordinate of the keypoint */
    bool  tracking_status{ false }; /**< the tracking status of the keypoint */
};

} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_TYPES_H */
