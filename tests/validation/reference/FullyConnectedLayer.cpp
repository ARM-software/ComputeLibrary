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
#include "FullyConnectedLayer.h"

#include "arm_compute/core/Types.h"
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
                            int rows_weights)
{
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

// Vector matrix multiply for quantized type
template < typename T, typename TB, typename std::enable_if < std::is_same<T, uint8_t>::value &&std::is_same<TB, int32_t>::value, int >::type = 0 >
void vector_matrix_multiply(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &dst, int offset_src, int offset_dst,
                            int cols_weights, int rows_weights)
{
    const T *src_ptr     = src.data() + offset_src;
    const T *weights_ptr = weights.data();
    const TB *bias_ptr    = bias.data();
    T        *dst_ptr     = dst.data() + offset_dst;

    const UniformQuantizationInfo iq_info = src.quantization_info().uniform();
    const UniformQuantizationInfo wq_info = weights.quantization_info().uniform();
    const UniformQuantizationInfo oq_info = dst.quantization_info().uniform();

    const int   input_offset   = -iq_info.offset;
    const float input_scale    = iq_info.scale;
    const int   weights_offset = -wq_info.offset;
    const float weights_scale  = wq_info.scale;
    const int   output_offset  = oq_info.offset;
    const float output_scale   = oq_info.scale;

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
        dst_ptr[y] = static_cast<T>(acc);

        weights_ptr += cols_weights;
    }
}
} // namespace

template <typename T, typename TB>
SimpleTensor<T> fully_connected_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &dst_shape, QuantizationInfo out_quant_info)
{
    // if no explicit quantization has been set you the same as src
    if(out_quant_info == QuantizationInfo())
    {
        out_quant_info = src.quantization_info();
    }

    // Create reference
    SimpleTensor<T> dst{ TensorShape{ dst_shape }, src.data_type(), 1, out_quant_info };

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
                                  rows_weights);
    }

    return dst;
}

template SimpleTensor<float> fully_connected_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &dst_shape,
                                                   QuantizationInfo out_quant_info);
template SimpleTensor<half> fully_connected_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, const TensorShape &dst_shape,
                                                  QuantizationInfo out_quant_info);
template SimpleTensor<uint8_t> fully_connected_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, const TensorShape &dst_shape,
                                                     QuantizationInfo out_quant_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
