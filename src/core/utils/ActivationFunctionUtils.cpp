/*
 * Copyright (c) 2016-2023 Arm Limited.
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

#include "arm_compute/core/utils/ActivationFunctionUtils.h"

#include <map>

namespace arm_compute
{
const std::string &string_from_activation_func(const ActivationFunction& act)
{
    static std::map<ActivationFunction, const std::string> act_map =
    {
        { ActivationFunction::ABS, "ABS" },
        { ActivationFunction::LINEAR, "LINEAR" },
        { ActivationFunction::LOGISTIC, "LOGISTIC" },
        { ActivationFunction::RELU, "RELU" },
        { ActivationFunction::BOUNDED_RELU, "BRELU" },
        { ActivationFunction::LU_BOUNDED_RELU, "LU_BRELU" },
        { ActivationFunction::LEAKY_RELU, "LRELU" },
        { ActivationFunction::SOFT_RELU, "SRELU" },
        { ActivationFunction::ELU, "ELU" },
        { ActivationFunction::SQRT, "SQRT" },
        { ActivationFunction::SQUARE, "SQUARE" },
        { ActivationFunction::TANH, "TANH" },
        { ActivationFunction::IDENTITY, "IDENTITY" },
        { ActivationFunction::HARD_SWISH, "HARD_SWISH" },
        { ActivationFunction::SWISH, "SWISH" },
        { ActivationFunction::GELU, "GELU" }

    };

    return act_map[act];
}

} // namespace arm_compute
