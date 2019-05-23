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
template <typename T>
SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<T> &src, const QuantizationInfo quantization_info)
{
    // Create reference
    SimpleTensor<uint8_t> dst{ src.shape(), DataType::QASYMM8, 1, quantization_info };

    for(int i = 0; i < src.num_elements(); ++i)
    {
#ifdef __aarch64__
        dst[i] = quantization_info.quantize((src[i]), RoundingPolicy::TO_NEAREST_EVEN);
#else  // __aarch64__
        dst[i] = quantization_info.quantize((src[i]), RoundingPolicy::TO_ZERO);
#endif // __aarch64__
    }
    return dst;
}

template SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<half> &src, const QuantizationInfo quantization_info);
template SimpleTensor<uint8_t> quantization_layer(const SimpleTensor<float> &src, const QuantizationInfo quantization_info);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
