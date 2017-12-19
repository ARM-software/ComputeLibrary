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

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
template < typename T, typename TB, typename std::enable_if < is_floating_point<T>::value &&is_floating_point<TB>::value, int >::type = 0 >
void vector_matrix_multiply(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &dst, int offset_src, int offset_dst, int cols_weights,
                            int rows_weights, uint8_t fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

    const T *src_ptr     = src.data() + offset_src;
    const T *weights_ptr = weights.data();
    const TB *bias_ptr    = bias.data();
    T        *dst_ptr     = dst.data() + offset_dst;

    for(int y = 0; y < rows_weights; ++y)
    {
        dst_ptr[y] = std::inner_product(src_ptr, src_ptr + cols_weights, weights_ptr, static_cast<T>(0)) + bias_ptr[y];
        weights_ptr += cols_weights;
    }
}

// Vector matrix multiply for fixed point type
template < typename T, typename TB, typename std::enable_if < std::is_integral<T>::value &&std::is_integral<TB>::value, int >::type = 0 >
void vector_matrix_multiply(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &dst, int offset_src, int offset_dst, int cols_weights,
                            int rows_weights, uint8_t fixed_point_position)
{
    const T *src_ptr     = src.data() + offset_src;
    const T *weights_ptr = weights.data();
    const TB *bias_ptr    = bias.data();
    T        *dst_ptr     = dst.data() + offset_dst;

    using namespace fixed_point_arithmetic;
    using promoted_type = fixed_point_arithmetic::traits::promote_t<T>;

    for(int y = 0; y < rows_weights; ++y)
    {
        // Reset accumulator
        fixed_point<promoted_type> acc(0, fixed_point_position);

        for(int x = 0; x < cols_weights; ++x)
        {
            const fixed_point<promoted_type> i_value(src_ptr[x], fixed_point_position, true);
            const fixed_point<promoted_type> w_value(weights_ptr[x], fixed_point_position, true);
            acc = acc + i_value * w_value;
        }

        // Get the bias
        const fixed_point<T> b(bias_ptr[y], fixed_point_position, true);

        // Convert back and accumulate the bias
        fixed_point<T> res(acc);
        res = res + b;

        // Store the result
        dst_ptr[y] = res.raw();

        weights_ptr += cols_weights;
    }
}

// Vector matrix multiply for quantized type
template <>
void vector_matrix_multiply(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, SimpleTensor<uint8_t> &dst, int offset_src, int offset_dst,
                            int cols_weights, int rows_weights, uint8_t fixed_point_position)
{
    ARM_COMPUTE_UNUSED(fixed_point_position);

    const uint8_t *src_ptr     = src.data() + offset_src;
    const uint8_t *weights_ptr = weights.data();
    const int32_t *bias_ptr    = bias.data();
    uint8_t       *dst_ptr     = dst.data() + offset_dst;

    const int   input_offset   = -src.quantization_info().offset;
    const float input_scale    = src.quantization_info().scale;
    const int   weights_offset = -weights.quantization_info().offset;
    const float weights_scale  = weights.quantization_info().scale;
    const int   output_offset  = dst.quantization_info().offset;
    const float output_scale   = dst.quantization_info().scale;

    int         output_multiplier = 0;
    int         output_shift      = 0;
    const float multiplier        = input_scale * weights_scale / output_scale;
    arm_compute::quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

    for(int y = 0; y < rows_weights; ++y)
    {
        // Reset accumulator
        int32_t acc = 0;

        for(int x = 0; x < cols_weights; ++x)
        {
            acc += (src_ptr[x] + input_offset) * (weights_ptr[x] + weights_offset);
        }

        // Accumulate the bias
        acc += bias_ptr[y];

        acc = asymm_rounding_divide_by_pow2(asymm_int_mult(acc, output_multiplier), output_shift);
        acc += output_offset;
        acc = utility::clamp<int32_t>(acc, 0, 255);

        // Store the result
        dst_ptr[y] = static_cast<uint8_t>(acc);

        weights_ptr += cols_weights;
    }
}
} // namespace

template <typename T, typename TB>
SimpleTensor<T> fully_connected_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &dst_shape)
{
    // Create reference
    SimpleTensor<T> dst{ TensorShape{ dst_shape }, src.data_type(), 1, src.fixed_point_position(), src.quantization_info() };

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
        const int offset_in  = k * cols_weights;
        const int offset_out = k * rows_weights;

        vector_matrix_multiply<T>(src,
                                  weights,
                                  bias,
                                  dst,
                                  offset_in,
                                  offset_out,
                                  cols_weights,
                                  rows_weights,
                                  src.fixed_point_position());
    }

    return dst;
}

template SimpleTensor<float> fully_connected_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &dst_shape);
template SimpleTensor<half> fully_connected_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, const TensorShape &dst_shape);
template SimpleTensor<qint8_t> fully_connected_layer(const SimpleTensor<qint8_t> &src, const SimpleTensor<qint8_t> &weights, const SimpleTensor<qint8_t> &bias, const TensorShape &dst_shape);
template SimpleTensor<qint16_t> fully_connected_layer(const SimpleTensor<qint16_t> &src, const SimpleTensor<qint16_t> &weights, const SimpleTensor<qint16_t> &bias, const TensorShape &dst_shape);
template SimpleTensor<uint8_t> fully_connected_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, const TensorShape &dst_shape);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
