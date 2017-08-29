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
#include "DequantizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
SimpleTensor<float> dequantization_layer(const SimpleTensor<T> &src, const SimpleTensor<float> &min_max)
{
    // Create reference
    SimpleTensor<float> dst{ src.shape(), DataType::F32 };

    // Compute reference
    const int width       = src.shape().x();
    const int height      = src.shape().y();
    const int depth       = src.shape().z();
    const int stride_w    = width * height * depth;
    const int num_batches = min_max.shape().total_size_upper(1);

    for(int k = 0; k < num_batches; ++k)
    {
        const float min     = min_max[k * 2 + 0];
        const float max     = min_max[k * 2 + 1];
        const float range   = max - min;
        const float scaling = range / 255.0f;

        for(int i = 0; i < stride_w; ++i)
        {
            dst[i + k * stride_w] = (static_cast<float>(src[i + k * stride_w]) * scaling) + min;
        }
    }

    return dst;
}

template SimpleTensor<float> dequantization_layer(const SimpleTensor<uint8_t> &src, const SimpleTensor<float> &min_max);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
