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
#ifndef __ARM_COMPUTE_TEST_ACTIVATION_LAYER_H__
#define __ARM_COMPUTE_TEST_ACTIVATION_LAYER_H__

#include "tests/SimpleTensor.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
inline T activate_float(T x, T a, T b, ActivationLayerInfo::ActivationFunction activation)
{
    T ret;

    switch(activation)
    {
        case ActivationLayerInfo::ActivationFunction::ABS:
            ret = std::abs(x);
            break;
        case ActivationLayerInfo::ActivationFunction::LINEAR:
            ret = a * x + b;
            break;
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            ret = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
            break;
        case ActivationLayerInfo::ActivationFunction::RELU:
            ret = std::max<T>(static_cast<T>(0), x);
            break;
        case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
            ret = std::min<T>(a, std::max(static_cast<T>(0), x));
            break;
        case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
            ret = std::min<T>(a, std::max<T>(b, x));
            break;
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            ret = (x > 0) ? x : a * x;
            break;
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            ret = std::log(static_cast<T>(1) + std::exp(x));
            break;
        case ActivationLayerInfo::ActivationFunction::SQRT:
            ret = std::sqrt(x);
            break;
        case ActivationLayerInfo::ActivationFunction::SQUARE:
            ret = x * x;
            break;
        case ActivationLayerInfo::ActivationFunction::TANH:
            ret = a * std::tanh(b * x);
            break;
        case ActivationLayerInfo::ActivationFunction::IDENTITY:
            ret = x;
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported activation function");
            break;
    }

    return ret;
}

template <typename T>
SimpleTensor<T> activation_layer(const SimpleTensor<T> &src, ActivationLayerInfo info, const QuantizationInfo &oq_info = QuantizationInfo());
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_ACTIVATION_LAYER_H__ */
