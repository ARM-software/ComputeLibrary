/*
 * Copyright (c) 2017-2020, 2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_REFERENCE_SOFTMAXLAYER_H
#define ACL_TESTS_VALIDATION_REFERENCE_SOFTMAXLAYER_H

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
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type = 0>
SimpleTensor<T> softmax_layer_generic(const SimpleTensor<T> &src, float beta, int32_t axis, bool is_log = false);

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type = 0>
SimpleTensor<T> softmax_layer_bfloat16(const SimpleTensor<T> &src, float beta, int32_t axis, bool is_log = false);

template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type = 0>
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t axis = 0, bool is_log = false);

template < typename T, typename std::enable_if < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int >::type = 0 >
SimpleTensor<T> softmax_layer(const SimpleTensor<T> &src, float beta, int32_t axis = 0, bool is_log = false);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_REFERENCE_SOFTMAXLAYER_H
