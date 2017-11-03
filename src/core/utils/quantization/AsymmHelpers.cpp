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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include <cmath>
#include <limits>
#include <numeric>

using namespace arm_compute::quantization;

arm_compute::Error arm_compute::quantization::calculate_quantized_multiplier_less_than_one(double multiplier,
                                                                                           int   *quant_multiplier,
                                                                                           int   *right_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quant_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(right_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier >= 1);
    if(multiplier == 0)
    {
        *quant_multiplier = 0;
        *right_shift      = 0;
        return arm_compute::Error{};
    }
    const double q = std::frexp(multiplier, right_shift);
    *right_shift *= -1;
    auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > (1ll << 31));
    if(q_fixed == (1ll << 31))
    {
        q_fixed /= 2;
        --*right_shift;
    }
    ARM_COMPUTE_RETURN_ERROR_ON(*right_shift < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
    *quant_multiplier = static_cast<int>(q_fixed);

    return arm_compute::Error{};
}