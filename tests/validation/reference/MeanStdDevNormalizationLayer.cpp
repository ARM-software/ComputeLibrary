/*
 * Copyright (c) 2019, 2022, 2024-2026 Arm Limited.
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
#include "MeanStdDevNormalizationLayer.h"

#include "arm_compute/core/Types.h"

#include <cmath>

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
/** Normalises one row in double precision, with the variance computed from centred values */
template <typename T>
void mean_std_normalize_row(const T *src, T *dst, int cols, float epsilon)
{
    double sum = 0.0;
    for (int j = 0; j < cols; ++j)
    {
        sum += static_cast<double>(src[j]);
    }
    const double mean = sum / cols;

    double sum_sq = 0.0;
    for (int j = 0; j < cols; ++j)
    {
        const double centred = static_cast<double>(src[j]) - mean;
        sum_sq += centred * centred;
    }
    const double stddev_inv = 1.0 / std::sqrt(sum_sq / cols + epsilon);

    for (int j = 0; j < cols; ++j)
    {
        dst[j] = static_cast<T>((static_cast<double>(src[j]) - mean) * stddev_inv);
    }
}
} // namespace

template <typename T>
SimpleTensor<T> mean_std_normalization_layer(const SimpleTensor<T> &src, float epsilon, const QuantizationInfo &oq_info)
{
    ARM_COMPUTE_UNUSED(oq_info);

    SimpleTensor<T> dst{src.shape(), src.data_type(), 1};
    const int       cols       = src.shape()[0];
    const int       batch_size = src.shape()[1];
    for (int i = 0; i < batch_size; ++i)
    {
        mean_std_normalize_row(src.data() + i * cols, dst.data() + i * cols, cols, epsilon);
    }
    return dst;
}

template <>
SimpleTensor<uint8_t>
mean_std_normalization_layer(const SimpleTensor<uint8_t> &src, float epsilon, const QuantizationInfo &oq_info)
{
    const QuantizationInfo dst_qinfo = oq_info.empty() ? src.quantization_info() : oq_info;

    SimpleTensor<float>   src_tmp = convert_from_asymmetric(src);
    SimpleTensor<float>   dst_tmp = mean_std_normalization_layer<float>(src_tmp, epsilon);
    SimpleTensor<uint8_t> dst     = convert_to_asymmetric<uint8_t>(dst_tmp, dst_qinfo);
    return dst;
}

template SimpleTensor<float>
mean_std_normalization_layer(const SimpleTensor<float> &src, float epsilon, const QuantizationInfo &oq_info);
template SimpleTensor<half>
mean_std_normalization_layer(const SimpleTensor<half> &src, float epsilon, const QuantizationInfo &oq_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
