/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTH_CONVERT_H
#define ARM_COMPUTE_TEST_DEPTH_CONVERT_H

#include "tests/SimpleTensor.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template < typename T1, typename T2, typename std::enable_if < std::is_integral<T1>::value &&!std::is_same<T1, T2>::value, int >::type = 0 >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

template < typename T1, typename T2, typename std::enable_if < is_floating_point<T1>::value &&(!std::is_same<T1, T2>::value &&!std::is_same<T2, bfloat16>::value), int >::type = 0 >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);

template < typename T1, typename T2, typename std::enable_if < std::is_same<T1, bfloat16>::value || std::is_same<T2, bfloat16>::value, int >::type = 0 >
SimpleTensor<T2> depth_convert(const SimpleTensor<T1> &src, DataType dt_out, ConvertPolicy policy, uint32_t shift);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTH_CONVERT_H */
