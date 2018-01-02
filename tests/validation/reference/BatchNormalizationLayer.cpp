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
#include "BatchNormalizationLayer.h"

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
// Batch Normalization Layer for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type *>
SimpleTensor<T> batch_normalization_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &mean, const SimpleTensor<T> &var, const SimpleTensor<T> &beta, const SimpleTensor<T> &gamma, float epsilon,
                                          int fixed_point_position)
{
    SimpleTensor<T> result(src.shape(), src.data_type());

    const auto cols       = static_cast<int>(src.shape()[0]);
    const auto rows       = static_cast<int>(src.shape()[1]);
    const auto depth      = static_cast<int>(src.shape()[2]);
    const int  upper_dims = src.shape().total_size() / (cols * rows * depth);

    for(int r = 0; r < upper_dims; ++r)
    {
        for(int i = 0; i < depth; ++i)
        {
            for(int k = 0; k < rows; ++k)
            {
                for(int l = 0; l < cols; ++l)
                {
                    const int pos = l + k * cols + i * rows * cols + r * cols * rows * depth;

                    fixed_point_arithmetic::fixed_point<T> src_qs(src[pos], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> var_qs(var[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> mean_qs(mean[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> beta_qs(beta[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> gamma_qs(gamma[i], fixed_point_position, true);
                    fixed_point_arithmetic::fixed_point<T> epsilon_qs(epsilon, fixed_point_position);

                    auto denominator = fixed_point_arithmetic::inv_sqrt(var_qs + epsilon_qs);
                    auto numerator   = src_qs - mean_qs;
                    auto x_bar       = numerator * denominator;
                    x_bar            = beta_qs + x_bar * gamma_qs;
                    result[pos]      = x_bar.raw();
                }
            }
        }
    }

    return result;
}

// Batch Normalization Layer for floating point type
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type *>
SimpleTensor<T> batch_normalization_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &mean, const SimpleTensor<T> &var, const SimpleTensor<T> &beta, const SimpleTensor<T> &gamma, float epsilon,
                                          int fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

    SimpleTensor<T> result(src.shape(), src.data_type());

    const auto cols       = static_cast<int>(src.shape()[0]);
    const auto rows       = static_cast<int>(src.shape()[1]);
    const auto depth      = static_cast<int>(src.shape()[2]);
    const int  upper_dims = src.shape().total_size() / (cols * rows * depth);

    for(int r = 0; r < upper_dims; ++r)
    {
        for(int i = 0; i < depth; ++i)
        {
            for(int k = 0; k < rows; ++k)
            {
                for(int l = 0; l < cols; ++l)
                {
                    const int   pos         = l + k * cols + i * rows * cols + r * cols * rows * depth;
                    const float denominator = sqrt(var[i] + epsilon);
                    const float numerator   = src[pos] - mean[i];
                    const float x_bar       = numerator / denominator;
                    result[pos]             = beta[i] + x_bar * gamma[i];
                }
            }
        }
    }
    return result;
}
template SimpleTensor<float> batch_normalization_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &mean, const SimpleTensor<float> &var, const SimpleTensor<float> &beta,
                                                       const SimpleTensor<float> &gamma, float epsilon, int fixed_point_position);
template SimpleTensor<int8_t> batch_normalization_layer(const SimpleTensor<int8_t> &src, const SimpleTensor<int8_t> &mean, const SimpleTensor<int8_t> &var, const SimpleTensor<int8_t> &beta,
                                                        const SimpleTensor<int8_t> &gamma, float epsilon, int fixed_point_position);
template SimpleTensor<int16_t> batch_normalization_layer(const SimpleTensor<int16_t> &src, const SimpleTensor<int16_t> &mean, const SimpleTensor<int16_t> &var, const SimpleTensor<int16_t> &beta,
                                                         const SimpleTensor<int16_t> &gamma, float epsilon, int fixed_point_position);
template SimpleTensor<half> batch_normalization_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &mean, const SimpleTensor<half> &var,
                                                      const SimpleTensor<half> &beta,
                                                      const SimpleTensor<half> &gamma, float epsilon, int fixed_point_position);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
