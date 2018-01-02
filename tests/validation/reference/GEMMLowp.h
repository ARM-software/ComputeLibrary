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
#ifndef __ARM_COMPUTE_TEST_GEMMLOWP_H__
#define __ARM_COMPUTE_TEST_GEMMLOWP_H__

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
template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<T> &in, int32_t result_offset, int32_t result_mult_int, int32_t result_shift, int32_t min = 0, int32_t max = 0);
template <typename T1, typename T2>
SimpleTensor<T1> gemmlowp_matrix_multiply_core(const SimpleTensor<T2> &a, const SimpleTensor<T2> &b, int32_t a_offset, int32_t b_offset);

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<T> &in, int32_t result_offset, int32_t result_mult_int, int32_t result_shift);

template <typename T1, typename T2>
SimpleTensor<T1> gemmlowp(const SimpleTensor<T2> &a, const SimpleTensor<T2> &b);

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale(const SimpleTensor<T> &in, const SimpleTensor<T> &bias, int32_t result_offset, int32_t result_mult_int, int32_t result_shift,
                                                                  int32_t min = 0, int32_t max = 0);

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<T> &in, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                int32_t result_offset_after_shift,
                                                                                int32_t min = 0, int32_t max = 0);

template <typename T>
SimpleTensor<uint8_t> gemmlowp_quantize_down_int32_to_uint8_scale_by_fixedpoint(const SimpleTensor<T> &in, const SimpleTensor<T> &bias, int32_t result_fixedpoint_multiplier, int32_t result_shift,
                                                                                int32_t result_offset_after_shift, int32_t min = 0, int32_t max = 0);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_GEMMLOWP_H__ */
