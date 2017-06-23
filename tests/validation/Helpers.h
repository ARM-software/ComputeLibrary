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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__
#define __ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__

#include "Types.h"

#include <type_traits>
#include <utility>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Helper function to get the testing range for each activation layer.
 *
 * @param[in] activation           Activation function to test.
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part. Defaults to 1.
 *
 * @return A pair containing the lower upper testing bounds for a given function.
 */
template <typename T>
std::pair<T, T> get_activation_layer_test_bounds(ActivationLayerInfo::ActivationFunction activation, int fixed_point_position = 1)
{
    bool is_float = std::is_floating_point<T>::value;
    std::pair<T, T> bounds;

    // Set initial values
    if(is_float)
    {
        bounds = std::make_pair(-255.f, 255.f);
    }
    else
    {
        bounds = std::make_pair(std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());
    }

    // Reduce testing ranges
    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            // Reduce range as exponent overflows
            if(is_float)
            {
                bounds.first  = -40.f;
                bounds.second = 40.f;
            }
            else
            {
                bounds.first  = -(1 << (fixed_point_position));
                bounds.second = 1 << (fixed_point_position);
            }
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            // Reduce range as exponent overflows
            if(!is_float)
            {
                bounds.first  = -(1 << (fixed_point_position));
                bounds.second = 1 << (fixed_point_position);
            }
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            // Reduce range as sqrt should take a non-negative number
            bounds.first = (is_float) ? 0 : 1 << (fixed_point_position);
            break;
        default:
            break;
    }
    return bounds;
}

/** Helper function to get the testing range for batch normalization layer.
 *
 * @param[in] fixed_point_position (Optional) Number of bits for the fractional part. Defaults to 1.
 *
 * @return A pair containing the lower upper testing bounds.
 */
template <typename T>
std::pair<T, T> get_batchnormalization_layer_test_bounds(int fixed_point_position = 1)
{
    bool is_float = std::is_floating_point<T>::value;
    std::pair<T, T> bounds;

    // Set initial values
    if(is_float)
    {
        bounds = std::make_pair(-1.f, 1.f);
    }
    else
    {
        bounds = std::make_pair(1, 1 << (fixed_point_position));
    }

    return bounds;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_VALIDATION_HELPERS_H__
