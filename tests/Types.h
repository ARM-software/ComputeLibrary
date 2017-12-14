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
#ifndef __ARM_COMPUTE_TEST_TYPES_H__
#define __ARM_COMPUTE_TEST_TYPES_H__

#include "arm_compute/core/Types.h"

#include <vector>

namespace arm_compute
{
/** Fixed point operation */
enum class FixedPointOp
{
    ADD,       /**< Addition */
    SUB,       /**< Subtraction */
    MUL,       /**< Multiplication */
    EXP,       /**< Exponential */
    LOG,       /**< Logarithm */
    INV_SQRT,  /**< Inverse square root */
    RECIPROCAL /**< Reciprocal */
};

/** Gradient dimension type. */
enum class GradientDimension
{
    GRAD_X,  /**< x gradient dimension */
    GRAD_Y,  /**< y gradient dimension */
    GRAD_XY, /**< x and y gradient dimension */
};

template <typename MinMaxType>
struct MinMaxLocationValues
{
    MinMaxType                 min{};
    MinMaxType                 max{};
    std::vector<Coordinates2D> min_loc{};
    std::vector<Coordinates2D> max_loc{};
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_TYPES_H__ */
