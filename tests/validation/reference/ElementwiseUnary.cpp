/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#include "ElementwiseUnary.h"
#include "tests/validation/Helpers.h"
#include "utils/TypePrinter.h"
namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> elementwise_unary(const SimpleTensor<T> &src, SimpleTensor<T> &dst, ElementWiseUnary op)
{
    for(int i = 0; i < src.num_elements(); ++i)
    {
        switch(op)
        {
            case ElementWiseUnary::RSQRT:
                dst[i] = 1.f / std::sqrt(src[i]);
                break;
            case ElementWiseUnary::EXP:
                dst[i] = std::exp(src[i]);
                break;
            case ElementWiseUnary::NEG:
                dst[i] = -src[i];
                break;
            case ElementWiseUnary::LOG:
                dst[i] = std::log(src[i]);
                break;
            case ElementWiseUnary::ABS:
                dst[i] = std::abs(src[i]);
                break;
            case ElementWiseUnary::SIN:
                dst[i] = std::sin(src[i]);
                break;
            case ElementWiseUnary::ROUND:
                dst[i] = arm_compute::support::cpp11::nearbyint(src[i]);
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
    }
    return dst;
}
template <>
SimpleTensor<int8_t> elementwise_unary(const SimpleTensor<int8_t> &src, SimpleTensor<int8_t> &dst, ElementWiseUnary op)
{
    if(dst.data_type() == DataType::QASYMM8_SIGNED)
    {
        SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
        SimpleTensor<float> dst_tmp(src.shape(), DataType::F32);
        for(int i = 0; i < src.num_elements(); ++i)
        {
            switch(op)
            {
                case ElementWiseUnary::RSQRT:
                    if(src_tmp[i] != 0)
                    {
                        dst_tmp[i] = 1.f / std::sqrt(src_tmp[i]);
                    }
                    else
                    {
                       // rsqrt(0) give 'inf' so set to the maximum in int8: 127
                       dst_tmp[i] = (127.0f - dst.quantization_info().uniform().offset)  * dst.quantization_info().uniform().scale ;
                    }
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
        }
        dst = convert_to_asymmetric<int8_t>(dst_tmp, dst.quantization_info());
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
    return dst;
}
template <>
SimpleTensor<uint8_t> elementwise_unary(const SimpleTensor<uint8_t> &src, SimpleTensor<uint8_t> &dst, ElementWiseUnary op)
{
    if(dst.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> src_tmp = convert_from_asymmetric(src);
        SimpleTensor<float> dst_tmp(src.shape(), DataType::F32);
        for(int i = 0; i < src.num_elements(); ++i)
        {
            switch(op)
            {
                case ElementWiseUnary::RSQRT:
                    if(src_tmp[i] != 0)
                    {
                        dst_tmp[i] = 1.f / std::sqrt(src_tmp[i]);
                    }
                    else
                    {
                        // rsqrt(0) give 'inf' so set to the maximum in uint8: 255
                        dst_tmp[i] = (255.0f - dst.quantization_info().uniform().offset)* dst.quantization_info().uniform().scale;
                    }
                    break;
                default:
                    ARM_COMPUTE_ERROR("Not implemented");
            }
        }
        dst = convert_to_asymmetric<uint8_t>(dst_tmp, dst.quantization_info());
    }
    else
    {
        ARM_COMPUTE_ERROR("Not implemented");
    }
    return dst;
}

template SimpleTensor<float> elementwise_unary(const SimpleTensor<float> &src, SimpleTensor<float> &dst, ElementWiseUnary op);
template SimpleTensor<half> elementwise_unary(const SimpleTensor<half> &src, SimpleTensor<half> &dst, ElementWiseUnary op);
template SimpleTensor<int32_t> elementwise_unary(const SimpleTensor<int32_t> &src, SimpleTensor<int32_t> &dst, ElementWiseUnary op);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
