/*
 * Copyright (c) 2019 ARM Limited.
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

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> mean_std_normalization_layer(const SimpleTensor<T> &src, float epsilon)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), 1 };

    const int cols       = src.shape()[0];
    const int batch_size = src.shape()[1];

    for(int i = 0; i < batch_size; ++i)
    {
        T sum    = static_cast<T>(0.f);
        T sum_sq = static_cast<T>(0.f);
        for(int j = 0; j < cols; ++j)
        {
            const T value = src[j + i * cols];
            sum += value;
            sum_sq += value * value;
        }
        const T mean       = sum / static_cast<T>(cols);
        const T var        = ((sum_sq / static_cast<T>(cols)) - (mean * mean)) + static_cast<T>(epsilon);
        const T stddev_inv = static_cast<T>(1.f) / static_cast<T>(std::sqrt(var));
        for(int j = 0; j < cols; ++j)
        {
            dst[j + i * cols] = (src[j + i * cols] - mean) * stddev_inv;
        }
    }
    return dst;
}

template SimpleTensor<float> mean_std_normalization_layer(const SimpleTensor<float> &src, float epsilon);
template SimpleTensor<half> mean_std_normalization_layer(const SimpleTensor<half> &src, float epsilon);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
