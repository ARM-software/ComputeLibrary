/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "tests/validation/Helpers.h"
#include "tests/framework/Asserts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <>
SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<uint8_t> &src)
{
    const UniformQuantizationInfo &quantization_info = src.quantization_info().uniform();
    SimpleTensor<float>            dst{ src.shape(), DataType::F32, 1, QuantizationInfo(), src.data_layout() };
#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = dequantize_qasymm8(src[i], quantization_info);
    }
    return dst;
}

template <>
SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<int8_t> &src)
{
    const UniformQuantizationInfo &quantization_info = src.quantization_info().uniform();
    SimpleTensor<float>            dst{ src.shape(), DataType::F32, 1, QuantizationInfo(), src.data_layout() };

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = dequantize_qasymm8_signed(src[i], quantization_info);
    }
    return dst;
}

template <>
SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<uint16_t> &src)
{
    const UniformQuantizationInfo &quantization_info = src.quantization_info().uniform();
    SimpleTensor<float>            dst{ src.shape(), DataType::F32, 1, QuantizationInfo(), src.data_layout() };

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = dequantize_qasymm16(src[i], quantization_info);
    }
    return dst;
}

template <>
SimpleTensor<uint8_t> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info)
{
    SimpleTensor<uint8_t>          dst{ src.shape(), DataType::QASYMM8, 1, quantization_info };
    const UniformQuantizationInfo &qinfo = quantization_info.uniform();

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantize_qasymm8(src[i], qinfo);
    }
    return dst;
}

template <>
SimpleTensor<int8_t> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info)
{
    SimpleTensor<int8_t>           dst{ src.shape(), DataType::QASYMM8_SIGNED, 1, quantization_info };
    const UniformQuantizationInfo &qinfo = quantization_info.uniform();

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantize_qasymm8_signed(src[i], qinfo);
    }
    return dst;
}

template <>
SimpleTensor<uint16_t> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info)
{
    SimpleTensor<uint16_t>         dst{ src.shape(), DataType::QASYMM16, 1, quantization_info };
    const UniformQuantizationInfo &qinfo = quantization_info.uniform();

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantize_qasymm16(src[i], qinfo);
    }
    return dst;
}

template <>
SimpleTensor<int16_t> convert_to_symmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info)
{
    SimpleTensor<int16_t>          dst{ src.shape(), DataType::QSYMM16, 1, quantization_info };
    const UniformQuantizationInfo &qinfo = quantization_info.uniform();

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantize_qsymm16(src[i], qinfo);
    }
    return dst;
}

template <>
SimpleTensor<float> convert_from_symmetric(const SimpleTensor<int16_t> &src)
{
    const UniformQuantizationInfo &quantization_info = src.quantization_info().uniform();
    SimpleTensor<float>            dst{ src.shape(), DataType::F32, 1, QuantizationInfo(), src.data_layout() };

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = dequantize_qsymm16(src[i], quantization_info);
    }
    return dst;
}

template <typename T>
void matrix_multiply(const SimpleTensor<T> &a, const SimpleTensor<T> &b, SimpleTensor<T> &out)
{
    ARM_COMPUTE_ERROR_ON(a.shape()[0] != b.shape()[1]);
    ARM_COMPUTE_ERROR_ON(a.shape()[1] != out.shape()[1]);
    ARM_COMPUTE_ERROR_ON(b.shape()[0] != out.shape()[0]);

    const int M = a.shape()[1]; // Rows
    const int N = b.shape()[0]; // Cols
    const int K = b.shape()[1];

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif /* _OPENMP */
    for(int y = 0; y < M; ++y)
    {
        for(int x = 0; x < N; ++x)
        {
            float acc = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                acc += a[y * K + k] * b[x + k * N];
            }

            out[x + y * N] = acc;
        }
    }
}

template <typename T>
void transpose_matrix(const SimpleTensor<T> &in, SimpleTensor<T> &out)
{
    ARM_COMPUTE_ERROR_ON((in.shape()[0] != out.shape()[1]) || (in.shape()[1] != out.shape()[0]));

    const int width  = in.shape()[0];
    const int height = in.shape()[1];

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif /* _OPENMP */
    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            const T val = in[x + y * width];

            out[x * height + y] = val;
        }
    }
}

template <typename T>
void get_tile(const SimpleTensor<T> &in, SimpleTensor<T> &tile, const Coordinates &coord)
{
    ARM_COMPUTE_ERROR_ON(tile.shape().num_dimensions() > 2);

    const int w_tile = tile.shape()[0];
    const int h_tile = tile.shape()[1];

    // Fill the tile with zeros
    std::fill(tile.data() + 0, (tile.data() + (w_tile * h_tile)), static_cast<T>(0));

    // Check if with the dimensions greater than 2 we could have out-of-bound reads
    for(size_t d = 2; d < Coordinates::num_max_dimensions; ++d)
    {
        if(coord[d] < 0 || coord[d] >= static_cast<int>(in.shape()[d]))
        {
            ARM_COMPUTE_ERROR("coord[d] < 0 || coord[d] >= in.shape()[d] with d >= 2");
        }
    }

    // Since we could have out-of-bound reads along the X and Y dimensions,
    // we start calculating the input address with x = 0 and y = 0
    Coordinates start_coord = coord;
    start_coord[0]          = 0;
    start_coord[1]          = 0;

    // Get input and roi pointers
    auto in_ptr  = static_cast<const T *>(in(start_coord));
    auto roi_ptr = static_cast<T *>(tile.data());

    const int x_in_start = std::max(0, coord[0]);
    const int y_in_start = std::max(0, coord[1]);
    const int x_in_end   = std::min(static_cast<int>(in.shape()[0]), coord[0] + w_tile);
    const int y_in_end   = std::min(static_cast<int>(in.shape()[1]), coord[1] + h_tile);

    // Number of elements to copy per row
    const int n = x_in_end - x_in_start;

    // Starting coordinates for the ROI
    const int x_tile_start = coord[0] > 0 ? 0 : std::abs(coord[0]);
    const int y_tile_start = coord[1] > 0 ? 0 : std::abs(coord[1]);

    // Update input pointer
    in_ptr += x_in_start;
    in_ptr += (y_in_start * in.shape()[0]);

    // Update ROI pointer
    roi_ptr += x_tile_start;
    roi_ptr += (y_tile_start * tile.shape()[0]);

    for(int y = y_in_start; y < y_in_end; ++y)
    {
        // Copy per row
        std::copy(in_ptr, in_ptr + n, roi_ptr);

        in_ptr += in.shape()[0];
        roi_ptr += tile.shape()[0];
    }
}

template <typename T>
void zeros(SimpleTensor<T> &in, const Coordinates &anchor, const TensorShape &shape)
{
    ARM_COMPUTE_ERROR_ON(anchor.num_dimensions() != shape.num_dimensions());
    ARM_COMPUTE_ERROR_ON(in.shape().num_dimensions() > 2);
    ARM_COMPUTE_ERROR_ON(shape.num_dimensions() > 2);

    // Check if with the dimensions greater than 2 we could have out-of-bound reads
    for(size_t d = 0; d < Coordinates::num_max_dimensions; ++d)
    {
        if(anchor[d] < 0 || ((anchor[d] + shape[d]) > in.shape()[d]))
        {
            ARM_COMPUTE_ERROR("anchor[d] < 0 || (anchor[d] + shape[d]) > in.shape()[d]");
        }
    }

    // Get input pointer
    auto in_ptr = static_cast<T *>(in(anchor[0] + anchor[1] * in.shape()[0]));

    const unsigned int n = in.shape()[0];

    for(unsigned int y = 0; y < shape[1]; ++y)
    {
        std::fill(in_ptr, in_ptr + shape[0], 0);
        in_ptr += n;
    }
}

std::pair<int, int> get_quantized_bounds(const QuantizationInfo &quant_info, float min, float max)
{
    ARM_COMPUTE_ERROR_ON_MSG(min > max, "min must be lower equal than max");

    const int min_bound = quantize_qasymm8(min, quant_info.uniform());
    const int max_bound = quantize_qasymm8(max, quant_info.uniform());
    return std::pair<int, int> { min_bound, max_bound };
}

std::pair<int, int> get_quantized_qasymm8_signed_bounds(const QuantizationInfo &quant_info, float min, float max)
{
    ARM_COMPUTE_ERROR_ON_MSG(min > max, "min must be lower equal than max");

    const int min_bound = quantize_qasymm8_signed(min, quant_info.uniform());
    const int max_bound = quantize_qasymm8_signed(max, quant_info.uniform());
    return std::pair<int, int> { min_bound, max_bound };
}

std::pair<int, int> get_symm_quantized_per_channel_bounds(const QuantizationInfo &quant_info, float min, float max, size_t channel_id)
{
    ARM_COMPUTE_ERROR_ON_MSG(min > max, "min must be lower equal than max");

    const int min_bound = quantize_qsymm8_per_channel(min, quant_info, channel_id);
    const int max_bound = quantize_qsymm8_per_channel(max, quant_info, channel_id);
    return std::pair<int, int> { min_bound, max_bound };
}

void add_padding_x(std::initializer_list<ITensor *> tensors, const DataLayout &data_layout, bool only_right_pad)
{
    if(data_layout == DataLayout::NHWC)
    {
        constexpr unsigned int lower = 1U;
        constexpr unsigned int upper = 16U;

        std::uniform_int_distribution<unsigned int> distribution(lower, upper);
        size_t                                      seed_offset = 0;

        for(ITensor *tensor : tensors)
        {
            ARM_COMPUTE_ERROR_ON(!tensor->info()->is_resizable());

            std::mt19937 gen(library->seed() + seed_offset++);

            const unsigned int right = distribution(gen);
            const unsigned int left  = only_right_pad ? 0 : distribution(gen);

            tensor->info()->extend_padding(PaddingSize(0U, right, 0U, left));
        }
    }
}

QuantizationHint suggest_conv_dst_q_info_and_bias(const QuantizationInfo &in_q_info,
                                                  const QuantizationInfo &weight_q_info,
                                                  int32_t height,
                                                  int32_t width,
                                                  int32_t channels,
                                                  DataType data_type,
                                                  float bias_fraction)
{
    /**  Quantization Setup of convolution
     *
     *  Just like any other multiply-accummulate, convolution (2D) operation
     *  multiplies and accumulates the input and weight tensors. This operation
     *  takes place in three dimensions: height, width and channels. All of them
     *  belong to the weight tensor.
     *
     *  The formula for simple convolution can be written as:
     *      C = sum_h sum_w sum_c(I[h_offset + h, w_offset + w, c] * W[h, w, c])
     *
     *  Here, h_offset and w_offset are the starting positions in the image. Effects
     *  of paddings are ignored. This accumulation reduces to something like
     *
     *  C = sum_m(I_index * W_hwc)
     *      where m is height x width x channels.
     *
     *  Non-unit strides and/or dilations do not change the probabilistic nature of
     *  this sum because we always iterate as the size of the weight tensor.
     *
     *  Paddings may affect this summation, but it's a boundary condition and so is
     *  neglected for brevity.
     */

    return suggest_mac_dst_q_info_and_bias(in_q_info, weight_q_info, height * width * channels, data_type, bias_fraction);
}

QuantizationHint suggest_matmul_dst_q_info_and_bias(const QuantizationInfo &lhs_q_info,
                                                    const QuantizationInfo &rhs_q_info,
                                                    int32_t m, int32_t n, int32_t k, DataType data_type,
                                                    float bias_fraction)
{
    ARM_COMPUTE_UNUSED(m, n);

    /**  Quantization Setup of matrix multiplication
     *
     *  We have a matrix multiplication of the form C = A * B + D
     *  where A is (m X k), B is (k x n) and C is therefore (m x n).
     *  The bias, D is (1 x n).
     *
     *  If we have some distributional statistics of A, B and D, i.e. mean and variance,
     *  we can estimate the mean and variance of a single value in C matrix and pick
     *  good scale and offset values for the output and have non-saturated tests.
     *
     *  Each element in the output matrix can be calculated as follows:
     *      C_ij = sum_k(A_ik * B_kj) + D_j
     *
     * Note: All possible A_ik, B_kj, D_j random variables are assumed mutually independent.
     * Note: In quantized operators, bias is an integer. But, its quantization scale is
     *       assumed to be equal to lhs_scale * rhs_scale, and offset equal to 0.
     * Note: Since, bias is an integer that should be given as input, we need to pick responsible
     *       values when adding it on top of the summation. This is where "bias_fraction" comes
     *       into play. Based on the fraction given, we also return suggested bias range (min/max)
     *       for not saturating the output.
     *
     * Because all random variables are mutually independent, any C_ij has the same statistics,
     * which is why we return a single destination quantization info object; which is why we can
     * resort to a more general calculation explained in suggest_mac_dst_q_info_and_bias().
     *
     * From a probabilistic perspective, the above calculation reduces to
     *      c = sum_k (a_k * b_k) + d
     */

    return suggest_mac_dst_q_info_and_bias(lhs_q_info, rhs_q_info, k, data_type, bias_fraction);
}

QuantizationHint suggest_mac_dst_q_info_and_bias(
    const QuantizationInfo &a_q_info, const QuantizationInfo &b_q_info, int32_t K, DataType data_type, float bias_fraction)
{
    QuantizationInfo c_q_info;

    ARM_COMPUTE_ASSERT(data_type == DataType::QASYMM8 || data_type == DataType::QASYMM8_SIGNED);

    const int32_t t_max = static_cast<int32_t>(data_type == DataType::QASYMM8 ? std::numeric_limits<uint8_t>::max() : std::numeric_limits<int8_t>::max());
    const int32_t t_min = static_cast<int32_t>(data_type == DataType::QASYMM8 ? std::numeric_limits<uint8_t>::min() : std::numeric_limits<int8_t>::min());

    /**  Quantization Setup of multiply-accummulate
     *
     * Expression (in float):
     *    C = sum_k ( A_k * B_k ) + D
     *
     * Lemma: An affine transformation (i.e. aX + b) to a discrete uniform random variable
     *        creates another discrete uniform random variable.
     *
     * Terminology:
     *  E[X]: Mean of the random variable X (sometimes referred as mu_x)
     *  var(X): Variance of the random variable X (someimes referred as sigma^2_x)
     *  std(X): sqrt(var(X)), standard deviation of X
     *
     * 1) Calculate the mean:
     *      E[C] = sum_k( E[A_k] * E[B_k] ) + D = K * mean_a * mean_b + mean_d
     *
     *      Since elements of A and B are uniformly distributed random variables, we have
     *          mean_a = (max_a + min_a) / 2, mean_b = (max_b + min_b ) / 2
     *          max_a and min_a can be calculated with the scale_a/b and offset_a/b
     *              by replacing data type minimum and maximums in the equations
     *
     *    We don't know mean_d because we have to choose it based on bias_fraction. If we call
     *    the summation as M_int, similar to above, we have:
     *
     *      E[C_int] = sum_k( E[A_k_int] * E[B_k_int] ) + E[D_int] = K * mean_a_int * mean_b_int + mean_d_int
     *                  \___________________________/
     *                             E[M_int]
     *
     *      We choose a bias mean proportional to the integer summation. This proportion is "bias_fraction".
     *      So, we have D_int = f * M_int (f: fraction), and
     *          E[D_int] = mean_d_int = f * E[M_int]
     *
     *      This also means, for floating point value of D, the following:
     *          E[D] = mean_d = E[D_int] * a_scale * b_scale
     *
     * 2) Calculate the variance:
     *      var(C)    = sum_k( var(A_k * B_k) ) + var(D)
     *                = sum_k ( E[A_k^2 * B_k^2] - E[A_k]^2E[B_k^2] )
     *                = ...
     *                = K * (var_a * var_b + var_a * mean^2_b + var_b * mean^2_a) + var_d
     *
     *      Similarly, due to uniform random variable properties, we have
     *          var_a = (max_a - min_a)^2 / 12
     *          var_b = (max_b - min_b)^2 / 12
     *
     *      Again, we don't know var_d as we don't know the bias. As set out in the previous section, we have
     *              var(D_int) = var(f * M_int) = f^2 * var(M_int)
     *
     *      Using the same expression, we can find var(M_int):
     *      var(C_int)    = sum_k( var(A_k_int * B_k_int) ) + var(D_int)
     *                    = sum_k ( E[A_k_int^2 * B_k_int^2] - E[A_k_int]^2E[B_k_int^2] )
     *                    = ...
     *                    = K * (var_a_int * var_b_int + var_a_int * mean^2_b_int + var_b_int * mean^2_a_int) + var_d_int
     *                      \_______________________________________________________________________________/
     *                                                          var(M_int)
     *
     *      Now, we know mean and variance of D_int, we can return a suitable bias range with
     *          [mean_d_int +/- 2 * std_d_int]
     *
     *      This also means, for floating point value of D, the following:
     *          var(D) = var_d = var(D_int) * a_scale^2 * b_scale^2
     *
     *      E[D] and var(D) calculated in steps (1) and (2) can be substituted into E[C] and var(C) calculatons.
     *
     * 3) Now, we have an idea of what would an average C will look like and how much deviation
     *    is present around it. The exact distribution of C is difficult to come up with dependent on K.
     *    But, as K increases, due to Central Limit Theorem, it'll look more like a bell shaped figure,
     *    approaching normal distribution.
     *
     *    This is useful because, in normal distribution, we know that values +- 2 std_deviation around
     *    the mean constitute 95% of the values. Therefore, setting a plausible range for us:
     *      C_range = [C_min, C_max] = [mean_c - 2 * std_c, mean_c + 2 * std_c]
     *
     * 4)
     *    If we map this [C_min, C_max] to [0, 255] or [-128, 127] depending on the signedness of the
     *    data type, we can find a suitable scale and offset for the output. On average, it's expected
     *    that 5% of the output values will saturate and 95% will remain in the range.
     *
     *    The equations to be solved for offset_c and scale_c are:
     *          C_min = scale_c * (type_min - offset_c)
     *          C_max = scale_c * (type_max - offset_c)
     */

    const int32_t a_offset = a_q_info.uniform().offset;
    const float   a_scale  = a_q_info.uniform().scale;
    const int32_t b_offset = b_q_info.uniform().offset;
    const float   b_scale  = b_q_info.uniform().scale;

    // Integer value statistics. Valid for both Lhs/A and Rhs/B
    const float     mean_a_int = (t_max + t_min) / 2.f;
    constexpr float var_a_int  = (256 * 256 - 1) / 12.f; // Discrete uniform RV variance
    const float     mean_b_int = mean_a_int;             // A_int and B_int has the same stats
    constexpr float var_b_int  = var_a_int;

    // Lhs/A stats
    const float max_a  = (t_max - a_offset) * a_scale;
    const float min_a  = (t_min - a_offset) * a_scale;
    const float mean_a = (max_a + min_a) / 2;
    const float var_a  = (max_a - min_a) * (max_a - min_a) / 12;

    // Rhs/B stats
    const float max_b  = (t_max - b_offset) * b_scale;
    const float min_b  = (t_min - b_offset) * b_scale;
    const float mean_b = (max_b + min_b) / 2;
    const float var_b  = (max_b - min_b) * (max_b - min_b) / 12;

    // Integer multiplication output/M stats
    const float mean_m_int = K * mean_a_int * mean_b_int;
    const float var_m_int  = K * (var_a_int * var_b_int + mean_a_int * var_b_int + mean_b_int + var_a_int);
    const float std_m_int  = sqrt(var_m_int);

    // Bias/D both Int and Float statistics
    const float mean_d_int = bias_fraction * mean_m_int;
    const float std_d_int  = bias_fraction * std_m_int;
    const float mean_d     = a_scale * b_scale * mean_d_int;
    const float std_d      = a_scale * b_scale * std_d_int;
    const float var_d      = std_d * std_d;

    // Also calculate the suggested bias range
    const int32_t min_bias = mean_d_int - 2 * std_d_int;
    const int32_t max_bias = mean_d_int + 2 * std_d_int;

    // Output/C stats
    const float mean_out = K * mean_a * mean_b + mean_d;
    const float var_out  = K * (var_a * var_b + var_a * mean_b * mean_b + var_b * mean_a * mean_a) + var_d;
    const float std_out  = sqrt(var_out);

    // Output quantization setup
    const float   scale_out  = 4 * std_out / 255;
    const int32_t offset_out = static_cast<int32_t>(t_min - (mean_out - 2.f * std_out) / scale_out);

    c_q_info = QuantizationInfo(scale_out, offset_out);

    return { c_q_info, min_bias, max_bias };
}

template void get_tile(const SimpleTensor<float> &in, SimpleTensor<float> &roi, const Coordinates &coord);
template void get_tile(const SimpleTensor<half> &in, SimpleTensor<half> &roi, const Coordinates &coord);
template void get_tile(const SimpleTensor<int> &in, SimpleTensor<int> &roi, const Coordinates &coord);
template void get_tile(const SimpleTensor<short> &in, SimpleTensor<short> &roi, const Coordinates &coord);
template void get_tile(const SimpleTensor<char> &in, SimpleTensor<char> &roi, const Coordinates &coord);
template void zeros(SimpleTensor<float> &in, const Coordinates &anchor, const TensorShape &shape);
template void zeros(SimpleTensor<half> &in, const Coordinates &anchor, const TensorShape &shape);
template void transpose_matrix(const SimpleTensor<float> &in, SimpleTensor<float> &out);
template void transpose_matrix(const SimpleTensor<half> &in, SimpleTensor<half> &out);
template void transpose_matrix(const SimpleTensor<int> &in, SimpleTensor<int> &out);
template void transpose_matrix(const SimpleTensor<short> &in, SimpleTensor<short> &out);
template void transpose_matrix(const SimpleTensor<char> &in, SimpleTensor<char> &out);
template void transpose_matrix(const SimpleTensor<int8_t> &in, SimpleTensor<int8_t> &out);
template void transpose_matrix(const SimpleTensor<uint8_t> &in, SimpleTensor<uint8_t> &out);
template void matrix_multiply(const SimpleTensor<float> &a, const SimpleTensor<float> &b, SimpleTensor<float> &out);
template void matrix_multiply(const SimpleTensor<half> &a, const SimpleTensor<half> &b, SimpleTensor<half> &out);

} // namespace validation
} // namespace test
} // namespace arm_compute
