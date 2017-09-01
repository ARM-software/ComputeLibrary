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
#include "FullyConnectedLayer.h"

#include "tests/validation/FixedPoint.h"
#include "tests/validation/half.h"

#include <numeric>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
// Vector matrix multiply for floating point
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type = 0>
void vector_matrix_multiply(const T *src, const T *weights, const T *bias, T *dst, int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

    for(int y = 0; y < rows_weights; ++y)
    {
        dst[y] = std::inner_product(src, src + cols_weights, weights, static_cast<T>(0)) + bias[y];
        weights += cols_weights;
    }
}

// Vector matrix multiply for fixed point type
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void vector_matrix_multiply(const T *src, const T *weights, const T *bias, T *dst, int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    using namespace fixed_point_arithmetic;
    using promoted_type = fixed_point_arithmetic::traits::promote_t<T>;

    for(int y = 0; y < rows_weights; ++y)
    {
        // Reset accumulator
        fixed_point<promoted_type> acc(0, fixed_point_position);

        for(int x = 0; x < cols_weights; ++x)
        {
            const fixed_point<promoted_type> i_value(src[x], fixed_point_position, true);
            const fixed_point<promoted_type> w_value(weights[x], fixed_point_position, true);
            acc = acc + i_value * w_value;
        }

        // Get the bias
        const fixed_point<T> b(bias[y], fixed_point_position, true);

        // Convert back and accumulate the bias
        fixed_point<T> res(acc);
        res = res + b;

        // Store the result
        dst[y] = res.raw();

        weights += cols_weights;
    }
}
} // namespace

template <typename T>
SimpleTensor<T> fully_connected_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<T> &bias, const TensorShape &dst_shape)
{
    // Create reference
    SimpleTensor<T> dst{ TensorShape{ dst_shape }, src.data_type(), 1, src.fixed_point_position() };

    // Sanity checks
    const int          num_batch_dimensions = std::max(0, static_cast<int>(dst_shape.num_dimensions()) - 1);
    const int          num_input_dimensions = src.shape().num_dimensions() - num_batch_dimensions;
    const unsigned int linear_input_size    = src.shape().total_size_lower(num_input_dimensions);

    ARM_COMPUTE_UNUSED(num_batch_dimensions);
    ARM_COMPUTE_UNUSED(num_input_dimensions);
    ARM_COMPUTE_UNUSED(linear_input_size);
    ARM_COMPUTE_ERROR_ON(weights.shape().x() != linear_input_size);
    ARM_COMPUTE_ERROR_ON(weights.shape().y() != bias.shape().x());
    ARM_COMPUTE_ERROR_ON(weights.shape().y() != dst.shape().x());

    // Compute reference
    const int cols_weights = weights.shape().x();
    const int rows_weights = weights.shape().y();
    const int num_batches  = dst_shape.total_size_upper(1);

    for(int k = 0; k < num_batches; ++k)
    {
        vector_matrix_multiply<T>(src.data() + k * cols_weights,
                                  weights.data(),
                                  bias.data(),
                                  dst.data() + k * rows_weights,
                                  cols_weights,
                                  rows_weights,
                                  src.fixed_point_position());
    }

    return dst;
}

template SimpleTensor<float> fully_connected_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &dst_shape);
template SimpleTensor<half_float::half> fully_connected_layer(const SimpleTensor<half_float::half> &src, const SimpleTensor<half_float::half> &weights, const SimpleTensor<half_float::half> &bias,
                                                              const TensorShape &dst_shape);
template SimpleTensor<qint8_t> fully_connected_layer(const SimpleTensor<qint8_t> &src, const SimpleTensor<qint8_t> &weights, const SimpleTensor<qint8_t> &bias, const TensorShape &dst_shape);
template SimpleTensor<qint16_t> fully_connected_layer(const SimpleTensor<qint16_t> &src, const SimpleTensor<qint16_t> &weights, const SimpleTensor<qint16_t> &bias, const TensorShape &dst_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
