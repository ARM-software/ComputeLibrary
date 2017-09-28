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
#include "QuantizationLayer.h"

#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<T> &src)
{
    // Create reference
    SimpleTensor<uint8_t> dst{ src.shape(), DataType::U8 };

    const int width       = src.shape().x();
    const int height      = src.shape().y();
    const int depth       = src.shape().z();
    const int stride_w    = width * height * depth;
    const int num_batches = src.shape().total_size_upper(3);

    for(int k = 0; k < num_batches; ++k)
    {
        // Compute min and max of the 3D tensor
        float min = src[k * stride_w];
        float max = src[k * stride_w];

        // Look for min and max values
        for(int i = 1; i < stride_w; ++i)
        {
            float val = src[i + k * stride_w];
            min       = std::min(min, val);
            max       = std::max(max, val);
        }

        // Saturate the result in case min = max
        if(min == max)
        {
            min = 0.0f;
            max = 1.0f;
        }

        const float range = max - min;

        for(int i = 0; i < stride_w; ++i)
        {
            // map values to range [0.0, 1.0]
            float       val        = src[i + k * stride_w];
            const float normalized = (val - min) / range;
            dst[i + k * stride_w]  = static_cast<uint8_t>(std::min(255.0f, normalized * 256.0f));
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<float> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
