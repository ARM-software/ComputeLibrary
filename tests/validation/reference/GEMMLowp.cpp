/*
 * Copyright (c) 2017-2020 ARM Limited.
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

#include "support/ToolchainSupport.h"

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
struct DataTypeExtractor
{
    static DataType data_type()
    {
        DataType data_type = DataType::UNKNOWN;
        if(std::is_same<T, int8_t>::value)
        {
            data_type = DataType::QASYMM8_SIGNED;
        }
        else if(std::is_same<T, uint8_t>::value)
        {
            data_type = DataType::QASYMM8;
        }
        else if(std::is_same<T, int16_t>::value)
        {
            data_type = DataType::QSYMM16;
        }
        return data_type;
    }
};

template <typename TIn, typename TOut>
void quantize_down_scale(const SimpleTensor<TIn> *in, const SimpleTensor<TIn> *bias, SimpleTensor<TOut> *dst, int32_t result_offset, std::vector<int32_t> result_mult_int,
                         std::vector<int32_t> result_shift, int32_t min, int32_t max)
{
    const int  cols_in        = in->shape().x();
    const bool is_per_channel = result_mult_int.size() > 1;

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < in->num_elements(); ++i)
    {
        int32_t result = ((*in)[i] + result_offset);

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        result *= (is_per_channel) ? result_mult_int[i % cols_in] : result_mult_int[0];

        result >>= (is_per_channel) ? result_shift[i % cols_in] : result_shift[0];

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<TOut>(std::max<TIn>(std::numeric_limits<TOut>::lowest(),
                                                    std::min<TIn>(std::numeric_limits<TOut>::max(), result)));
    }
}

template <typename TIn, typename TOut>
void quantize_down_scale_by_fixedpoint(const SimpleTensor<TIn> *in, const SimpleTensor<TIn> *bias, SimpleTensor<TOut> *dst, std::vector<int32_t> result_fixedpoint_multiplier,
                                       std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    const int  cols_in        = in->shape().x();
    const bool is_per_channel = result_fixedpoint_multiplier.size() > 1;

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < in->num_elements(); ++i)
    {
        TIn result = (*in)[i];

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        // Fixed point multiplication
        const int32_t multiplier = (is_per_channel) ? result_fixedpoint_multiplier[i % cols_in] : result_fixedpoint_multiplier[0];
        const int32_t shift      = (is_per_channel) ? result_shift[i % cols_in] : result_shift[0];

        if(shift < 0)
        {
            result = asymm_int_mult(result * (1 << (-shift)), multiplier);
        }
        else
        {
            result = asymm_rounding_divide_by_pow2(asymm_int_mult(result, multiplier), shift);
        }
        result += result_offset_after_shift;

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<TOut>(std::max<TIn>(std::numeric_limits<TOut>::lowest(),
                                                    std::min<TIn>(std::numeric_limits<TOut>::max(), result)));
    }
}

template <typename TIn, typename TOut>
void quantize_down_scale_by_float(const SimpleTensor<TIn> *in, const SimpleTensor<TIn> *bias, SimpleTensor<TOut> *dst, std::vector<float_t> result_real_multiplier,
                                  int32_t result_offset, int32_t min, int32_t max)
{
    const int  cols_in        = in->shape().x();
    const bool is_per_channel = result_real_multiplier.size() > 1;

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < in->num_elements(); ++i)
    {
        TIn result = (*in)[i];

        if(bias != nullptr)
        {
            result += (*bias)[i % cols_in];
        }

        // Float multiplication
        const float_t multiplier = (is_per_channel) ? result_real_multiplier[i % cols_in] : result_real_multiplier[0];

        float_t result_f = static_cast<float_t>(result) * multiplier + static_cast<float_t>(result_offset);
        result           = static_cast<TIn>(support::cpp11::round(result_f));

        // Bounded ReLu
        if(min != max)
        {
            result = std::max(min, std::min(max, result));
        }

        (*dst)[i] = static_cast<TOut>(std::max<TIn>(std::numeric_limits<TOut>::lowest(),
                                                    std::min<TIn>(std::numeric_limits<TOut>::max(), result)));
    }
}
} // namespace

template <typename T_out, typename T_in, typename T_in_1>
SimpleTensor<T_out> gemmlowp_matrix_multiply_core(const SimpleTensor<T_in> &a, const SimpleTensor<T_in_1> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset)
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
template <typename T1, typename T2, typename T3>
SimpleTensor<T1> gemmlowp(const SimpleTensor<T2> &a, const SimpleTensor<T3> &b, TensorShape shape_c)
{
    return gemmlowp_matrix_multiply_core<T1, T2, T3>(a, b, shape_c, 0, 0);
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale(const SimpleTensor<TIn> &in, int32_t result_offset, std::vector<int32_t> result_mult_int, std::vector<int32_t> result_shift,
                                                int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale<TIn, TOut>(&in, nullptr, &dst, result_offset, result_mult_int, result_shift, min, max);

    return dst;
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale(const SimpleTensor<TIn> &in, const SimpleTensor<TIn> &bias, int32_t result_offset, std::vector<int32_t> result_mult_int,
                                                std::vector<int32_t> result_shift, int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale<TIn, TOut>(&in, &bias, &dst, result_offset, result_mult_int, result_shift, min, max);

    return dst;
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<TIn> &in, std::vector<int32_t> result_fixedpoint_multiplier, std::vector<int32_t> result_shift,
                                                              int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale_by_fixedpoint<TIn, TOut>(&in, nullptr, &dst, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

    return dst;
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<TIn> &in, const SimpleTensor<TIn> &bias, std::vector<int32_t> result_fixedpoint_multiplier,
                                                              std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale_by_fixedpoint<TIn, TOut>(&in, &bias, &dst, result_fixedpoint_multiplier, result_shift, result_offset_after_shift, min, max);

    return dst;
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<TIn> &in, const SimpleTensor<TIn> &bias,
                                                         std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale_by_float<TIn, TOut>(&in, &bias, &dst, result_real_multiplier, result_offset, min, max);

    return dst;
}

template <typename TIn, typename TOut>
SimpleTensor<TOut> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<TIn> &in,
                                                         std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max)
{
    SimpleTensor<TOut> dst(in.shape(), DataTypeExtractor<TOut>::data_type());

    quantize_down_scale_by_float<TIn, TOut>(&in, nullptr, &dst, result_real_multiplier, result_offset, min, max);

    return dst;
}

template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b,
                                                                     std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<int32_t> &a,
                                                                     std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b,
                                                                    std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale_by_float(const SimpleTensor<int32_t> &a,
                                                                    std::vector<float_t> result_real_multiplier, int32_t result_offset, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, std::vector<int32_t> result_fixedpoint_multiplier,
                                                                          std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b,
                                                                          std::vector<int32_t> result_fixedpoint_multiplier,
                                                                          std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, std::vector<int32_t> result_fixedpoint_multiplier,
                                                                         std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b,
                                                                         std::vector<int32_t> result_fixedpoint_multiplier,
                                                                         std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<int16_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, std::vector<int32_t> result_fixedpoint_multiplier,
                                                                          std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<int16_t> gemmlowp_quantize_down_scale_by_fixedpoint(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b,
                                                                          std::vector<int32_t> result_fixedpoint_multiplier,
                                                                          std::vector<int32_t> result_shift, int32_t result_offset_after_shift, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale(const SimpleTensor<int32_t> &a, int32_t result_offset, std::vector<int32_t> result_mult_int,
                                                            std::vector<int32_t> result_shift, int32_t min, int32_t max);
template SimpleTensor<uint8_t> gemmlowp_quantize_down_scale(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b, int32_t result_offset, std::vector<int32_t> result_mult_int,
                                                            std::vector<int32_t> result_shift, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale(const SimpleTensor<int32_t> &a, int32_t result_offset, std::vector<int32_t> result_mult_int,
                                                           std::vector<int32_t> result_shift, int32_t min, int32_t max);
template SimpleTensor<int8_t> gemmlowp_quantize_down_scale(const SimpleTensor<int32_t> &a, const SimpleTensor<int32_t> &b, int32_t result_offset, std::vector<int32_t> result_mult_int,
                                                           std::vector<int32_t> result_shift, int32_t min, int32_t max);
template SimpleTensor<int32_t> gemmlowp_matrix_multiply_core(const SimpleTensor<int8_t> &a, const SimpleTensor<int8_t> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset);
template SimpleTensor<int32_t> gemmlowp_matrix_multiply_core(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, TensorShape shape_c, int32_t a_offset, int32_t b_offset);
template SimpleTensor<int32_t> gemmlowp<int32_t, int8_t, int8_t>(const SimpleTensor<int8_t> &a, const SimpleTensor<int8_t> &b, TensorShape shape_c);
template SimpleTensor<int32_t> gemmlowp<int32_t, uint8_t, uint8_t>(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, TensorShape shape_c);
template SimpleTensor<int32_t> gemmlowp<int32_t, uint8_t, int8_t>(const SimpleTensor<uint8_t> &a, const SimpleTensor<int8_t> &b, TensorShape shape_c);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
