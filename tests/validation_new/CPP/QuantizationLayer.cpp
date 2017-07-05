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

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
void compute_min_max(const SimpleTensor<float> &src, float *min, float *max)
{
    // Set min and max to first pixel
    float tmp_min = src[0];
    float tmp_max = src[0];

    // Look for min and max values
    for(int i = 1; i < src.num_elements(); ++i)
    {
        if(src[i] < tmp_min)
        {
            tmp_min = src[i];
        }
        if(src[i] > tmp_max)
        {
            tmp_max = src[i];
        }
    }

    *min = tmp_min;
    *max = tmp_max;
}

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<T> &src)
{
    // Create reference
    SimpleTensor<uint8_t> dst{ src.shape(), DataType::U8 };

    // Compute min and max of the tensor using Min-Max layer
    float min = 0.f;
    float max = 0.f;

    compute_min_max(src, &min, &max);

    const float range = max - min;

    for(int i = 0; i < src.num_elements(); ++i)
    {
        // map values to range [0.0, 1.0]
        const float normalized = (src[i] - min) / range;
        dst[i]                 = static_cast<uint8_t>(std::min(255.0f, normalized * 256.0f));
    }

    return dst;
}

template SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<float> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
