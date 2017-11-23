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
#include "DepthConvertLayer.h"

#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"

#include "tests/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&std::is_floating_point<T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_UNUSED(shift);

    using namespace fixed_point_arithmetic;
    SimpleTensor<T2> result(src.shape(), dt_out);

    const int fixed_point_position = src.fixed_point_position();

    for(int i = 0; i < src.num_elements(); ++i)
    {
        result[i] = static_cast<float>(fixed_point<T1>(src[i], fixed_point_position, true));
    }

    return result;
}

template < typename T1, typename T2, typename std::enable_if < std::is_floating_point<T1>::value &&std::is_integral<T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_UNUSED(shift);

    using namespace fixed_point_arithmetic;
    SimpleTensor<T2> result(src.shape(), dt_out, 1, src.fixed_point_position());

    const int fixed_point_position = result.fixed_point_position();

    for(int i = 0; i < src.num_elements(); ++i)
    {
        result[i] = fixed_point<T2>(src[i], fixed_point_position).raw();
    }

    return result;
}

template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&std::is_integral<T2>::value &&!std::is_same<T1, T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    SimpleTensor<T2> result(src.shape(), dt_out);

    // Up-casting
    if(src.data_type() <= dt_out)
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            result[i] = src[i] << shift;
        }
    }
    // Down-casting
    else
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            T1 val    = src[i] >> shift;
            result[i] = (policy == ConvertPolicy::SATURATE) ? saturate_cast<T2>(val) : static_cast<T2>(val);
        }
    }
    return result;
}

template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&std::is_integral<T2>::value &&std::is_same<T1, T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(policy);

    using namespace fixed_point_arithmetic;

    SimpleTensor<T2> result(src.shape(), dt_out);

    bool is_in_place = (&src == &result);

    const int fixed_point_position_in  = src.fixed_point_position();
    const int fixed_point_position_out = (is_in_place) ? static_cast<int>(shift) : result.fixed_point_position();

    if(!is_in_place || (fixed_point_position_in != fixed_point_position_out))
    {
        for(int i = 0; i < src.num_elements(); ++i)
        {
            auto x = fixed_point<T2>(src[i], fixed_point_position_in, true);
            x.resacle(fixed_point_position_out);
            result[i] = x.raw();
        }
    }

    return result;
}

template < typename T1, typename T2, typename std::enable_if < std::is_floating_point<T1>::value &&is_floating_point<T2>::value, int >::type >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_UNUSED(shift);

    SimpleTensor<T2> result(src.shape(), dt_out);

    for(int i = 0; i < src.num_elements(); ++i)
    {
        result[i] = static_cast<T2>(src[i]);
    }
}

template SimpleTensor<uint16_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<uint8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint32_t> depth_convert(const SimpleTensor<uint16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<uint8_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int32_t> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<int8_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<float> depth_convert(const SimpleTensor<int16_t> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int8_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
template SimpleTensor<int16_t> depth_convert(const SimpleTensor<float> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
