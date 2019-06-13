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
#include "GEMMLowp.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/reference/UtilsQuantizedAsymm.h"

#include <limits>

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
template <typename T>
void quantize_down_int32_to_uint8_scale(const SimpleTensor<T> *in, const SimpleTensor<T> *bias, SimpleTensor<uint8_t> *dst, int32_t result_offset, int32_t result_mult_int, int32_t result_shift,
                                        int32_t min, int32_t max)
{
    const int cols_in = in->shape().x();

    for(int i = 0; i < in->num_elements(); ++i)
    {
        int32_t result = ((*in)[i] + result_offset);

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        result *= result_mult_int;

        result >>= result_shift;

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<uint8_t>(std::max(0, std::min(255, result)));
    }
}

template <typename T>
void quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<T> *in, const SimpleTensor<T> *bias, SimpleTensor<uint8_t> *dst, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                      int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    const int cols_in = in->shape().x();

    for(int i = 0; i < in->num_elements(); ++i)
    {
        int32_t result = (*in)[i];

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        // Fixed point multiplication
        result = asymm_rounding_divide_by_pow2(asymm_int_mult(result, result_fixedpoint_multiplier), result_shift);
        result += result_offset_after_shift;

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<uint8_t>(std::max(0, std::min(255, result)));
    }
}

template <typename T>
void quantize_down_int32_to_int16_scale_by_fixedpoint(const SimpleTensor<T> *in, const SimpleTensor<T> *bias, SimpleTensor<int16_t> *dst, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                      int32_t min, int32_t max)
{
    const int cols_in = in->shape().x();

    for(int i = 0; i < in->num_elements(); ++i)
    {
        int32_t result = (*in)[i];

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        // Fixed point multiplication
        result = asymm_rounding_divide_by_pow2(asymm_int_mult(result, result_fixedpoint_multiplier), result_shift);

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<int16_t>(std::max(-32768, std::min(32767, result)));
    }
}
} // namespace

template <typename T_out, typename T_in>
SimpleTensor<T_out> gemmlowp_matrix_multiply_core(const SimpleTensor<T_in> &a, const SimpleTensor<T_in> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset)
{
    static_assert(std::is_same<typename std::decay<T_out>::type, int32_t>::value, "Only int32_t is allowed for the output");

    DataType            dt = std::is_same<T_out, int32_t>::value ? DataType::S32 : DataType::U32;
    SimpleTensor<T_out> c(shape_c, dt);

    const int K = a.shape().x();
    const int M = a.shape().y();
    const int N = b.shape().x();
    const int D = a.shape().z(); // Number of matrices in a batch

    const int a_stride_z = K * M;
    // Do not slide the matrix B along the 3rd dimension in case matrix B has less than 3 dimensions
    const int b_stride_z = b.shape().num_dimensions() > 2 ? N * K : 0;
    const int c_stride_z = N * M;

    std::vector<T_out> acc;
    acc.resize(N);

    for(int depth = 0; depth < D; ++depth)
    {
        const int base_addr_a = depth * a_stride_z;
        const int base_addr_b = depth * b_stride_z;
        const int base_addr_c = depth * c_stride_z;

        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                acc[j] = 0;
            }
            for(int k = 0; k < K; ++k)
            {
                const T_out tmp_a = a_offset + static_cast<T_out>(a[base_addr_a + k + i * K]);
                for(int j = 0; j < N; ++j)
                {
                    const T_out tmp_b       = b_offset + static_cast<T_out>(b[base_addr_b + j + k * N]);
                    const T_out mult_as_int = tmp_a * tmp_b;
                    acc[j] += mult_as_int;
                }
            }
            for(int j = 0; j < N; ++j)
            {
                c[base_addr_c + j + i * N] = acc[j];
            }
        }
    }

    return c;
}

// used to validate assembly kernels which don't know anything about offsets
template <typename T1, typename T2>
SimpleTensor<T1> gemmlowp(const SimpleTensor<T2> &a, const SimpleTensor<T2> &b, TensorShape shape_c)
{
    return gemmlowp_matrix_multiply_core<T1, T2>(a, b, shape_c, 0, 0);
}

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<T> &in, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min, int32_t max)
{
    SimpleTensor<uint8_t> dst(in.shape(), DataType::QASYMM8);

    quantize_down_int32_to_uint8_scale<T>(&in, nullptr, &dst, result_offset, result_mult_int, result_shift, min, max);

    return dst;
}

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<T> &in, const SimpleTensor<T> &bias, int32_t result_offset, int32_t result_mult_int, int32_t result_shift,
                                                                  int32_t min, int32_t max)
{
    SimpleTensor<uint8_t> dst(in.shape(), DataType::QASYMM8);

    quantize_down_int32_to_uint8_scale<T>(&in, &bias, &dst, result_offset, result_mult_int, result_shift, min, max);

    return dst;
}

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<T> &in, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                int32_t result_offset_after_shift, int32_t min,
                                                                                int32_t max)
{
    SimpleTensor<uint8_t> dst(in.shape(), DataType::QASYMM8);

    quantize_down_int32_to_uint8_scale_by_fixedpoint<T>(&in, nullptr, &dst, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

    return dst;
}

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<T> &in, const SimpleTensor<T> &bias, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    SimpleTensor<uint8_t> dst(in.shape(), DataType::QASYMM8);

    quantize_down_int32_to_uint8_scale_by_fixedpoint<T>(&in, &bias, &dst, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

    return dst;
}

template <typename T>
SimpleTensor<int16_t> gemmlowp_quantize_down_int32_to_int16_scale_by_fixedpoint(const SimpleTensor<T> &in, int32_t result_fixedpoint_multiplier, int32_t result_shift, int32_t min,
                                                                                int32_t max)
{
    SimpleTensor<int16_t> dst(in.shape(), DataType::QSYMM16);

    quantize_down_int32_to_int16_scale_by_fixedpoint<T>(&in, nullptr, &dst, result_fixedpoint_multiplier, result_shift, min, max);

    return dst;
}

template <typename T>
SimpleTensor<int16_t> gemmlowp_quantize_down_int32_to_int16_scale_by_fixedpoint(const SimpleTensor<T> &in, const SimpleTensor<T> &bias, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                int32_t min, int32_t max)
{
    SimpleTensor<int16_t> dst(in.shape(), DataType::QSYMM16);

    quantize_down_int32_to_int16_scale_by_fixedpoint<T>(&in, &bias, &dst, result_fixedpoint_multiplier, result_shift, min, max);

    return dst;
}

template SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                         int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b, int32_t result_fixedpoint_multiplier,
                                                                                         int32_t result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<int16_t> gemmlowp_quantize_down_int32_to_int16_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                         int32_t min, int32_t max);
template SimpleTensor<int16_t> gemmlowp_quantize_down_int32_to_int16_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b, int32_t result_fixedpoint_multiplier,
                                                                                         int32_t result_shift, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<int32_t> &a, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min,
                                                                           int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b, int32_t result_offset, int32_t result_mult_int,
                                                                           int32_t result_shift, int32_t min, int32_t max);
template SimpleTensor<int32_t> gemmlowp_matrix_multiply_core(const SimpleTensor<int8_t> &a, const SimpleTensor<int8_t> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset);
template SimpleTensor<int32_t> gemmlowp_matrix_multiply_core(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset);
template SimpleTensor<int32_t> gemmlowp(const SimpleTensor<int8_t> &a, const SimpleTensor<int8_t> &b, TensorShape shape_c);
template SimpleTensor<int32_t> gemmlowp(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, TensorShape shape_c);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
