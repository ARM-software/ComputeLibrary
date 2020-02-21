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

#include "Permute.h"

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
template <typename TOut>
TOut dequantize(int8_t val, const UniformQuantizationInfo qinfo, DataType dt)
{
    if(dt == DataType::QSYMM8 || dt == DataType::QSYMM8_PER_CHANNEL)
    {
        return static_cast<TOut>(dequantize_qsymm8(val, qinfo));
    }
    else
    {
        return static_cast<TOut>(dequantize_qasymm8_signed(val, qinfo));
    }
}
template <typename TOut>
TOut dequantize(uint8_t val, const UniformQuantizationInfo qinfo, DataType dt)
{
    ARM_COMPUTE_UNUSED(dt);
    return static_cast<TOut>(dequantize_qasymm8(val, qinfo));
}
template <typename TOut>
TOut dequantize(int16_t val, const UniformQuantizationInfo qinfo, DataType dt)
{
    ARM_COMPUTE_UNUSED(dt);
    return static_cast<TOut>(dequantize_qsymm16(val, qinfo));
}
} // namespace
template <typename TOut, typename TIn>
SimpleTensor<TOut> dequantization_layer(const SimpleTensor<TIn> &src)
{
    const DataType src_data_type = src.data_type();
    const DataType dst_data_type = std::is_same<TOut, float>::value ? DataType::F32 : DataType::F16;

    SimpleTensor<TOut> dst{ src.shape(), dst_data_type };

    if(is_data_type_quantized_per_channel(src_data_type))
    {
        const int WH = src.shape().x() * src.shape().y();
        const int C  = src.shape().z();
        const int N  = src.shape().total_size() / (WH * C);

        const std::vector<float> qscales = src.quantization_info().scale();

        for(int n = 0; n < N; ++n)
        {
            for(int c = 0; c < C; ++c)
            {
                const size_t                  idx           = n * C * WH + c * WH;
                const UniformQuantizationInfo channel_qinfo = { qscales[c], 0 };

                // Dequantize slice
                for(int s = 0; s < WH; ++s)
                {
                    dst[idx + s] = dequantize<TOut>(static_cast<TIn>(src[idx + s]), channel_qinfo, src_data_type);
                }
            }
        }
    }
    else
    {
        const UniformQuantizationInfo &quantization_info = src.quantization_info().uniform();
        ARM_COMPUTE_ERROR_ON(quantization_info.offset != 0 && src_data_type == DataType::QSYMM8);

        for(int i = 0; i < src.num_elements(); ++i)
        {
            dst[i] = static_cast<TOut>(dequantize<TOut>(static_cast<TIn>(src[i]), quantization_info, src_data_type));
        }
    }

    return dst;
}

template SimpleTensor<half> dequantization_layer(const SimpleTensor<uint8_t> &src);
template SimpleTensor<float> dequantization_layer(const SimpleTensor<uint8_t> &src);
template SimpleTensor<half> dequantization_layer(const SimpleTensor<int8_t> &src);
template SimpleTensor<float> dequantization_layer(const SimpleTensor<int8_t> &src);
template SimpleTensor<half> dequantization_layer(const SimpleTensor<int16_t> &src);
template SimpleTensor<float> dequantization_layer(const SimpleTensor<int16_t> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
