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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include <cmath>
#include <limits>
#include <numeric>

using namespace arm_compute::quantization;

constexpr int64_t fixed_point_one_Q0 = (1LL << 31);
constexpr float   epsilon            = 0.00001f;

arm_compute::Status arm_compute::quantization::calculate_quantized_multiplier_less_than_one(float multiplier,
                                                                                            int *quant_multiplier,
                                                                                            int *right_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quant_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(right_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < -epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f + epsilon);
    if(std::fabs(1.0f - multiplier) < epsilon)
    {
        *quant_multiplier = 1;
        *right_shift      = 0;
        return arm_compute::Status{};
    }

    if(std::fabs(0.0f - multiplier) < epsilon)
    {
        *quant_multiplier = 0;
        *right_shift      = 0;
        return arm_compute::Status{};
    }

    const double q = std::frexp(multiplier, right_shift);
    *right_shift *= -1;
    auto q_fixed = static_cast<int64_t>(round(q * fixed_point_one_Q0));
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);
    if(q_fixed == fixed_point_one_Q0)
    {
        q_fixed /= 2;
        --*right_shift;
    }
    ARM_COMPUTE_RETURN_ERROR_ON(*right_shift < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
    *quant_multiplier = static_cast<int32_t>(q_fixed);

    return arm_compute::Status{};
}

arm_compute::Status arm_compute::quantization::calculate_quantized_multiplier_greater_than_one(float multiplier,
                                                                                               int *quantized_multiplier,
                                                                                               int *left_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quantized_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(left_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < 1.f);
    const double q       = std::frexp(multiplier, left_shift);
    auto         q_fixed = static_cast<int64_t>(round(q * fixed_point_one_Q0));
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);
    if(q_fixed == fixed_point_one_Q0)
    {
        q_fixed /= 2;
        ++*left_shift;
    }
    ARM_COMPUTE_RETURN_ERROR_ON(*left_shift < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);

    return arm_compute::Status{};
}
