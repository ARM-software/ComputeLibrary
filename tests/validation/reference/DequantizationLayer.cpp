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
#include "DequantizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> dequantization_layer(const SimpleTensor<uint8_t> &src)
{
    const DataType          dst_data_type     = std::is_same<T, float>::value ? DataType::F32 : DataType::F16;
    const QuantizationInfo &quantization_info = src.quantization_info();

    SimpleTensor<T> dst{ src.shape(), dst_data_type };

    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = static_cast<T>(quantization_info.dequantize(src[i]));
    }

    return dst;
}

template SimpleTensor<half> dequantization_layer(const SimpleTensor<uint8_t> &src);
template SimpleTensor<float> dequantization_layer(const SimpleTensor<uint8_t> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
