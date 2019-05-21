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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_CONVOLUTION_H__
#define __ARM_COMPUTE_TEST_VALIDATION_CONVOLUTION_H__

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

namespace arm_compute
{
namespace test
{
namespace convolution_3d
{
namespace detail
{
inline bool is_valid_pixel(int i, int min, int max)
{
    return (i >= min && i < max);
}

// 3D convolution for floating point type
template < typename T, typename TB, typename std::enable_if < validation::is_floating_point<T>::value &&validation::is_floating_point<TB>::value, int >::type = 0 >
inline void convolution3d(const SimpleTensor<T> &in, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &out,
                          int i_offset, int w_offset, int b_offset, int o_offset,
                          int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights, int dilation_x = 1, int dilation_y = 1)
{
    const T *in_ptr  = in.data() + i_offset;
    const T *w_ptr   = weights.data() + w_offset;
    const TB *b_ptr   = bias.data() + b_offset;
    T        *out_ptr = out.data() + o_offset;

    const int half_width_weights_start  = width_weights / 2;
    const int half_width_weights_end    = ((width_weights % 2) == 0) ? (half_width_weights_start - 1) : half_width_weights_start;
    const int half_height_weights_start = height_weights / 2;
    const int half_height_weights_end   = ((height_weights % 2) == 0) ? (half_height_weights_start - 1) : half_height_weights_start;

    // Reset accumulator
    T acc(0);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights_start; yk <= half_height_weights_end; ++yk)
        {
            for(int xk = -half_width_weights_start; xk <= half_width_weights_end; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk * dilation_x, 0, width_in) && is_valid_pixel(yi + yk * dilation_y, 0, height_in))
                {
                    const int idx = xk + half_width_weights_start;
                    const int idy = yk + half_height_weights_start;

                    const T i_value = in_ptr[offset_slice_in + xk * dilation_x + yk * dilation_y * width_in];
                    const T w_value = w_ptr[idx + idy * width_weights + ifm * width_weights * height_weights];

                    acc += i_value * w_value;
                }
            }
        }
    }

    // Accumulate the bias and store the result
    *out_ptr = acc + (*b_ptr);
}

// 3D convolution for QASYMM8 type
template < typename T, typename TB, typename std::enable_if < std::is_same<T, uint8_t>::value &&std::is_same<TB, int32_t>::value, int >::type = 0 >
inline void convolution3d(const SimpleTensor<T> &in, const SimpleTensor<T> &weights, const SimpleTensor<TB> &bias, SimpleTensor<T> &out,
                          int i_offset, int w_offset, int b_offset, int o_offset,
                          int xi, int yi, int width_in, int height_in, int depth_in, int width_weights, int height_weights, int dilation_x = 1, int dilation_y = 1)
{
    const T *in_ptr  = in.data() + i_offset;
    const T *w_ptr   = weights.data() + w_offset;
    const TB *b_ptr   = bias.data() + b_offset;
    T        *out_ptr = out.data() + o_offset;

    const UniformQuantizationInfo iq_info = in.quantization_info().uniform();
    const UniformQuantizationInfo wq_info = weights.quantization_info().uniform();
    const UniformQuantizationInfo oq_info = out.quantization_info().uniform();

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

    const int half_width_weights_start  = width_weights / 2;
    const int half_width_weights_end    = ((width_weights % 2) == 0) ? (half_width_weights_start - 1) : half_width_weights_start;
    const int half_height_weights_start = height_weights / 2;
    const int half_height_weights_end   = ((height_weights % 2) == 0) ? (half_height_weights_start - 1) : half_height_weights_start;

    // Reset accumulator
    int32_t acc(0);

    // Compute a 2D convolution for each IFM and accumulate the result
    for(int ifm = 0; ifm < depth_in; ++ifm)
    {
        // Compute the offset for the input slice
        const int offset_slice_in = xi + yi * width_in + ifm * width_in * height_in;

        // Compute 2D convolution
        for(int yk = -half_height_weights_start; yk <= half_height_weights_end; ++yk)
        {
            for(int xk = -half_width_weights_start; xk <= half_width_weights_end; ++xk)
            {
                // Check if the pixel is out-of-bound
                if(is_valid_pixel(xi + xk * dilation_x, 0, width_in) && is_valid_pixel(yi + yk * dilation_y, 0, height_in))
                {
                    const int idx = xk + half_width_weights_start;
                    const int idy = yk + half_height_weights_start;

                    const uint8_t i_value = in_ptr[offset_slice_in + xk * dilation_x + yk * dilation_y * width_in];
                    const uint8_t w_value = w_ptr[idx + idy * width_weights + ifm * width_weights * height_weights];

                    acc += (i_value + input_offset) * (w_value + weights_offset);
                }
            }
        }
    }

    // Accumulate the bias
    acc += (*b_ptr);

    acc = validation::asymm_rounding_divide_by_pow2(validation::asymm_int_mult(acc, output_multiplier), output_shift);
    acc += output_offset;
    acc = utility::clamp<int32_t>(acc, 0, 255);

    // Store the result
    *out_ptr = acc;
}
} // namespace detail
} // namespace convolution_3d
} // namespace test
} // namespace arm_compute
#endif /*__ARM_COMPUTE_TEST_VALIDATION_CONVOLUTION_H__ */
