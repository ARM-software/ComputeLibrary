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
#include "ArithmeticSubtraction.h"

#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T1, typename T2, typename T3>
SimpleTensor<T3> arithmetic_subtraction(const SimpleTensor<T1> &src1, const SimpleTensor<T2> &src2, DataType dst_data_type, ConvertPolicy convert_policy)
{
    SimpleTensor<T3> result(src1.shape(), dst_data_type);

    using intermediate_type = typename common_promoted_signed_type<typename std::conditional<sizeof(T1) >= sizeof(T2), T1, T2>::type >::intermediate_type;

    for(int i = 0; i < src1.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(src1[i]) - static_cast<intermediate_type>(src2[i]);
        result[i]             = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T3>(val) : static_cast<T3>(val);
    }

    return result;
}

template SimpleTensor<uint8_t> arithmetic_subtraction(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_subtraction(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_subtraction(const SimpleTensor<uint8_t> &src1, const SimpleTensor<int16_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_subtraction(const SimpleTensor<int16_t> &src1, const SimpleTensor<uint8_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_subtraction(const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<int8_t> arithmetic_subtraction(const SimpleTensor<int8_t> &src1, const SimpleTensor<int8_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<half> arithmetic_subtraction(const SimpleTensor<half> &src1, const SimpleTensor<half> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<float> arithmetic_subtraction(const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
