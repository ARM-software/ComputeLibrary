/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "DepthConvertLayer.h"

#include "tests/validation/Helpers.h"

#include "arm_compute/core/utils/misc/Rounding.h"
#include "arm_compute/core/utils/misc/SaturateCast.h"

#include "tests/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&!std::is_same<T1, T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    SimpleTensor<T2> result(src.shape(), dt_out);

    // Up-casting
    if(element_size_from_data_type(src.data_type()) < element_size_from_data_type(dt_out))
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            if(is_data_type_quantized(src.data_type()))
            {
                result[i] = scvt_f32_qasymm8(src[i], src.quantization_info().scale, src.quantization_info().offset);
            }
            else
            {
                result[i] = src[i] << shift;
            }
        }
    }
    // Down-casting
    else
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            T1 val    = src[i] >> shift;
            result[i] = (policy == ConvertPolicy::SATURATE) ? utils::cast::saturate_cast<T2>(val) : static_cast<T2>(val);
        }
    }
    return result;
}

template < typename T1, typename T2, typename std::enable_if < is_floating_point<T1>::value &&!std::is_same<T1, T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    SimpleTensor<T2> result(src.shape(), dt_out);
    ARM_COMPUTE_ERROR_ON(shift != 0);
    ARM_COMPUTE_UNUSED(policy, shift);

    if(!is_floating_point<T2>::value)
    {
        // Always saturate on floats
        for(int i = 0; i < src.num_elements(); ++i)
        {
            if(is_data_type_quantized(dt_out))
            {
                T1 val    = utils::rounding::round_half_away_from_zero(src[i]);
                result[i] = sqcvt_qasymm8_f32(val, src.quantization_info().scale, src.quantization_info().offset);
            }
            else
            {
                T1 val    = utils::rounding::round_half_away_from_zero(src[i]);
                result[i] = utils::cast::saturate_cast<T2>(val);
            }
        }
    }
    else
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            result[i] = static_cast<T2>(src[i]);
        }
    }
    return result;
}

// U8
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// S8
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// U16
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// S16
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// U32
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<uint32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// S32
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<int32_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// F16
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<half> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

// F32
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<half> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
