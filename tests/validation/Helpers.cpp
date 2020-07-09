/*
 * Copyright (c) 2017-2020 Arm Limited.
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

#include <algorithm>
#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
void fill_mask_from_pattern(uint8_t *mask, int cols, int rows, MatrixPattern pattern)
{
    unsigned int                v = 0;
    std::mt19937                gen(library->seed());
    std::bernoulli_distribution dist(0.5);

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < cols; ++c, ++v)
        {
            uint8_t val = 0;

            switch(pattern)
            {
                case MatrixPattern::BOX:
                    val = 255;
                    break;
                case MatrixPattern::CROSS:
                    val = ((r == (rows / 2)) || (c == (cols / 2))) ? 255 : 0;
                    break;
                case MatrixPattern::DISK:
                    val = (((r - rows / 2.0f + 0.5f) * (r - rows / 2.0f + 0.5f)) / ((rows / 2.0f) * (rows / 2.0f)) + ((c - cols / 2.0f + 0.5f) * (c - cols / 2.0f + 0.5f)) / ((cols / 2.0f) *
                            (cols / 2.0f))) <= 1.0f ? 255 : 0;
                    break;
                case MatrixPattern::OTHER:
                    val = (dist(gen) ? 0 : 255);
                    break;
                default:
                    return;
            }

            mask[v] = val;
        }
    }

    if(pattern == MatrixPattern::OTHER)
    {
        std::uniform_int_distribution<uint8_t> distribution_u8(0, ((cols * rows) - 1));
        mask[distribution_u8(gen)] = 255;
    }
}

HarrisCornersParameters harris_corners_parameters()
{
    HarrisCornersParameters params;

    std::mt19937                           gen(library->seed());
    std::uniform_real_distribution<float>  threshold_dist(0.f, 0.001f);
    std::uniform_real_distribution<float>  sensitivity(0.04f, 0.15f);
    std::uniform_real_distribution<float>  euclidean_distance(0.f, 30.f);
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);

    params.threshold             = threshold_dist(gen);
    params.sensitivity           = sensitivity(gen);
    params.min_dist              = euclidean_distance(gen);
    params.constant_border_value = int_dist(gen);

    return params;
}

CannyEdgeParameters canny_edge_parameters()
{
    CannyEdgeParameters params;

    std::mt19937                           gen(library->seed());
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);
    std::uniform_int_distribution<uint8_t> threshold_dist(2, 255);

    params.constant_border_value = int_dist(gen);
    params.upper_thresh          = threshold_dist(gen); // upper_threshold >= 2
    threshold_dist               = std::uniform_int_distribution<uint8_t>(1, params.upper_thresh - 1);
    params.lower_thresh          = threshold_dist(gen); // lower_threshold >= 1 && lower_threshold < upper_threshold

    return params;
}

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
template void matrix_multiply(const SimpleTensor<float> &a, const SimpleTensor<float> &b, SimpleTensor<float> &out);
template void matrix_multiply(const SimpleTensor<half> &a, const SimpleTensor<half> &b, SimpleTensor<half> &out);

} // namespace validation
} // namespace test
} // namespace arm_compute
