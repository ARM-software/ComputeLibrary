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
#include "ConvolutionLayer.h"

#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

#include "tests/framework/Asserts.h"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

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
inline bool is_valid_pixel(int i, int min, int max)
{
    return (i >= min && i < max);
}

// 3D convolution for floating point type
template < typename T, typename TB, typename std::enable_if < is_floating_point<T>::value &&is_floating_point<TB>::value, int >::type = 0 >
void convolution3d(const SimpleTensor<T> &in, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &out,
                   int i_offset, int w_offset, int b_offset, int o_offset,
                   int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights)
{
    const T *in_ptr  = in.data() + i_offset;
    const T *w_ptr   = weights.data() + w_offset;
    const TB *b_ptr   = bias.data() + b_offset;
    T        *out_ptr = out.data() + o_offset;

    const int half_width_weights  = width_weights / 2;
    const int half_height_weights = height_weights / 2;

    // Reset accumulator
    T acc(0);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights; yk <= half_height_weights; ++yk)
        {
            for(int xk = -half_width_weights; xk <= half_width_weights; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk, 0, width_in) && is_valid_pixel(yi + yk, 0, height_in))
                {
                    const int idx = xk + half_width_weights;
                    const int idy = yk + half_height_weights;

                    const T i_value = in_ptr[offset_slice_in + xk + yk * width_in];
                    const T w_value = w_ptr[idx + idy * width_weights + ifm * width_weights * height_weights];

                    acc += i_value * w_value;
                }
            }
        }
    }

    // Accumulate the bias and store the result
    *out_ptr = acc + (*b_ptr);
}

// 3D convolution for fixed point type
template < typename T, typename TB, typename std::enable_if < std::is_integral<T>::value &&std::is_integral<TB>::value, int >::type = 0 >
void convolution3d(const SimpleTensor<T> &in, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &out,
                   int i_offset, int w_offset, int b_offset, int o_offset,
                   int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights)
{
    const T *in_ptr               = in.data() + i_offset;
    const T *w_ptr                = weights.data() + w_offset;
    const T *b_ptr                = bias.data() + b_offset;
    T       *out_ptr              = out.data() + o_offset;
    int      fixed_point_position = in.fixed_point_position();

    const int half_width_weights  = width_weights / 2;
    const int half_height_weights = height_weights / 2;

    using namespace fixed_point_arithmetic;
    using promoted_type = fixed_point_arithmetic::traits::promote_t<T>;

    // Reset accumulator
    fixed_point<promoted_type> acc(0, fixed_point_position);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights; yk <= half_height_weights; ++yk)
        {
            for(int xk = -half_width_weights; xk <= half_width_weights; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk, 0, width_in) && is_valid_pixel(yi + yk, 0, height_in))
                {
                    const int idx = xk + half_width_weights;
                    const int idy = yk + half_height_weights;

                    const fixed_point<promoted_type> i_value(in_ptr[offset_slice_in + xk + yk * width_in], fixed_point_position, true);
                    const fixed_point<promoted_type> w_value(w_ptr[idx + idy * width_weights + ifm * width_weights * height_weights], fixed_point_position, true);
                    const fixed_point<promoted_type> iw = i_value * w_value;
                    acc                                 = iw + acc;
                }
            }
        }
    }

    // Get the bias
    const fixed_point<promoted_type> b(*b_ptr, fixed_point_position, true);

    // Accumulate the bias and covert back
    acc = acc + b;
    fixed_point<T> res(acc);
    *out_ptr = res.raw();
}

// 3D convolution for QASYMM8 type
template <>
void convolution3d(const SimpleTensor<uint8_t> &in, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, SimpleTensor<uint8_t> &out,
                   int i_offset, int w_offset, int b_offset, int o_offset,
                   int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights)
{
    const uint8_t *in_ptr  = in.data() + i_offset;
    const uint8_t *w_ptr   = weights.data() + w_offset;
    const int32_t *b_ptr   = bias.data() + b_offset;
    uint8_t       *out_ptr = out.data() + o_offset;

    const int   input_offset   = -in.quantization_info().offset;
    const float input_scale    = in.quantization_info().scale;
    const int   weights_offset = -weights.quantization_info().offset;
    const float weights_scale  = weights.quantization_info().scale;
    const int   output_offset  = out.quantization_info().offset;
    const float output_scale   = out.quantization_info().scale;

    int         output_multiplier = 0;
    int         output_shift      = 0;
    const float multiplier        = input_scale * weights_scale / output_scale;
    arm_compute::quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

    const int half_width_weights  = width_weights / 2;
    const int half_height_weights = height_weights / 2;

    // Reset accumulator
    int32_t acc(0);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights; yk <= half_height_weights; ++yk)
        {
            for(int xk = -half_width_weights; xk <= half_width_weights; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk, 0, width_in) && is_valid_pixel(yi + yk, 0, height_in))
                {
                    const int idx = xk + half_width_weights;
                    const int idy = yk + half_height_weights;

                    const uint8_t i_value = in_ptr[offset_slice_in + xk + yk * width_in];
                    const uint8_t w_value = w_ptr[idx + idy * width_weights + ifm * width_weights * height_weights];

                    acc += (i_value + input_offset) * (w_value + weights_offset);
                }
            }
        }
    }

    // Accumulate the bias
    acc += (*b_ptr);

    acc = asymm_rounding_divide_by_pow2(asymm_int_mult(acc, output_multiplier), output_shift);
    acc += output_offset;
    acc = utility::clamp<int32_t>(acc, 0, 255);

    // Store the result
    *out_ptr = acc;
}
} // namespace

template <typename T, typename TB>
SimpleTensor<T> convolution_layer(const SimpleTensor<T> &src, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, const TensorShape &output_shape, const PadStrideInfo &info)
{
    // Create reference
    SimpleTensor<T> dst{ output_shape, src.data_type(), 1, src.fixed_point_position(), src.quantization_info() };

    // Compute reference
    const int width_in       = src.shape().x();
    const int height_in      = src.shape().y();
    const int depth_in       = src.shape().z();
    const int width_out      = dst.shape().x();
    const int height_out     = dst.shape().y();
    const int depth_out      = dst.shape().z();
    const int width_weights  = weights.shape().x();
    const int height_weights = weights.shape().y();
    const int depth_weights  = weights.shape().z();
    const int pad_left       = std::min(static_cast<int>(info.pad_left()), width_weights / 2);
    const int pad_top        = std::min(static_cast<int>(info.pad_top()), height_weights / 2);
    const int pad_right      = std::min(static_cast<int>(info.pad_right()), width_weights / 2);
    const int pad_bottom     = std::min(static_cast<int>(info.pad_bottom()), height_weights / 2);

    const int start_xi    = width_weights / 2 - pad_left;
    const int start_yi    = height_weights / 2 - pad_top;
    const int end_xi      = width_in + pad_left - width_weights / 2 + pad_right - width_weights / 2;
    const int end_yi      = height_in + pad_top - height_weights / 2 + pad_bottom - height_weights / 2;
    const int stride_xi   = info.stride().first;
    const int stride_yi   = info.stride().second;
    const int num_batches = src.shape().total_size() / (width_in * height_in * depth_in);

    for(int r = 0; r < num_batches; ++r)
    {
        for(int yi = start_yi; yi < start_yi + end_yi; yi += stride_yi)
        {
            for(int xi = start_xi; xi < start_xi + end_xi; xi += stride_xi)
            {
                for(int ofm = 0; ofm < depth_out; ++ofm)
                {
                    // Compute input and output offsets
                    const int offset_in  = r * width_in * height_in * depth_in;
                    const int xo         = (xi - start_xi) / stride_xi;
                    const int yo         = (yi - start_yi) / stride_yi;
                    const int offset_out = xo + yo * width_out + ofm * width_out * height_out + r * width_out * height_out * depth_out;

                    ARM_COMPUTE_ASSERT(xo < width_out);
                    ARM_COMPUTE_ASSERT(yo < height_out);

                    // Compute 3D convolution
                    convolution3d(src, weights, bias, dst,
                                  offset_in, ofm * width_weights * height_weights * depth_weights, ofm, offset_out,
                                  xi, yi,
                                  width_in, height_in, depth_in,
                                  width_weights, height_weights);
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<float> convolution_layer(const SimpleTensor<float> &src, const SimpleTensor<float> &weights, const SimpleTensor<float> &bias, const TensorShape &output_shape,
                                               const PadStrideInfo &info);
template SimpleTensor<half> convolution_layer(const SimpleTensor<half> &src, const SimpleTensor<half> &weights, const SimpleTensor<half> &bias, const TensorShape &output_shape,
                                              const PadStrideInfo &info);
template SimpleTensor<qint8_t> convolution_layer(const SimpleTensor<qint8_t> &src, const SimpleTensor<qint8_t> &weights, const SimpleTensor<qint8_t> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info);
template SimpleTensor<qint16_t> convolution_layer(const SimpleTensor<qint16_t> &src, const SimpleTensor<qint16_t> &weights, const SimpleTensor<qint16_t> &bias, const TensorShape &output_shape,
                                                  const PadStrideInfo &info);
template SimpleTensor<uint8_t> convolution_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &weights, const SimpleTensor<int32_t> &bias, const TensorShape &output_shape,
                                                 const PadStrideInfo &info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
