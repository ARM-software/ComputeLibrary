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
#include "ActivationLayer.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> activation_layer(const SimpleTensor<T> &src, ActivationLayerInfo info)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1, src.fixed_point_position() };

    // Compute reference
    const T a(info.a());
    const T b(info.b());

    for(int i = 0; i < src.num_elements(); ++i)
    {
        T x = src[i];

        switch(info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::ABS:
                dst[i] = std::abs(x);
                break;
            case ActivationLayerInfo::ActivationFunction::LINEAR:
                dst[i] = a * x + b;
                break;
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                dst[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
                break;
            case ActivationLayerInfo::ActivationFunction::RELU:
                dst[i] = std::max<T>(static_cast<T>(0), x);
                break;
            case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                dst[i] = std::min<T>(a, std::max(static_cast<T>(0), x));
                break;
            case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                dst[i] = std::min<T>(a, std::max<T>(b, x));
                break;
            case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                dst[i] = (x > 0) ? x : a * x;
                break;
            case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                dst[i] = std::log(static_cast<T>(1) + std::exp(x));
                break;
            case ActivationLayerInfo::ActivationFunction::SQRT:
                dst[i] = std::sqrt(x);
                break;
            case ActivationLayerInfo::ActivationFunction::SQUARE:
                dst[i] = x * x;
                break;
            case ActivationLayerInfo::ActivationFunction::TANH:
                dst[i] = a * std::tanh(b * x);
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported activation function");
        }
    }

    return dst;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
SimpleTensor<T> activation_layer(const SimpleTensor<T> &src, ActivationLayerInfo info)
{
    using namespace fixed_point_arithmetic;

    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1, src.fixed_point_position() };

    // Compute reference
    const int            fixed_point_position = src.fixed_point_position();
    const fixed_point<T> a(info.a(), fixed_point_position);
    const fixed_point<T> b(info.b(), fixed_point_position);
    const fixed_point<T> const_0(0, fixed_point_position);
    const fixed_point<T> const_1(1, fixed_point_position);

    for(int i = 0; i < src.num_elements(); ++i)
    {
        fixed_point<T> x(src[i], fixed_point_position, true);

        switch(info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::ABS:
                dst[i] = abs(x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LINEAR:
                dst[i] = add(b, mul(a, x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
                dst[i] = (const_1 / (const_1 + exp(-x))).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::RELU:
                dst[i] = max(const_0, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::BOUNDED_RELU:
                dst[i] = min(a, max(const_0, x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU:
                dst[i] = min(a, max(b, x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
                dst[i] = (x > const_0) ? x.raw() : mul(a, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                dst[i] = log(const_1 + exp(x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SQRT:
                dst[i] = (const_1 / inv_sqrt(x)).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::SQUARE:
                dst[i] = mul(x, x).raw();
                break;
            case ActivationLayerInfo::ActivationFunction::TANH:
                dst[i] = mul(a, tanh(mul(b, x))).raw();
                break;
            default:
                ARM_COMPUTE_ERROR("Unsupported activation function");
        }
    }

    return dst;
}

template <>
SimpleTensor<uint8_t> activation_layer<uint8_t>(const SimpleTensor<uint8_t> &src, ActivationLayerInfo info)
{
    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = activation_layer<float>(src_tmp, info);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric(dst_tmp, src.quantization_info());
    return dst;
}

template SimpleTensor<float> activation_layer(const SimpleTensor<float> &src, ActivationLayerInfo info);
template SimpleTensor<half> activation_layer(const SimpleTensor<half> &src, ActivationLayerInfo info);
template SimpleTensor<qint8_t> activation_layer(const SimpleTensor<qint8_t> &src, ActivationLayerInfo info);
template SimpleTensor<qint16_t> activation_layer(const SimpleTensor<qint16_t> &src, ActivationLayerInfo info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
