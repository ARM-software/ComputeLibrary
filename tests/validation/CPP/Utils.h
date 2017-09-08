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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_UTILS_H__
#define __ARM_COMPUTE_TEST_VALIDATION_UTILS_H__

#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/ILutAccessor.h"
#include "tests/Types.h"

#include <array>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename T>
T tensor_elem_at(const SimpleTensor<T> &in, Coordinates coord, BorderMode border_mode, T constant_border_value);

template <typename T>
T bilinear_policy(const SimpleTensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, uint8_t constant_border_value);

template <typename T1, typename T2, typename T3>
void apply_2d_spatial_filter(Coordinates coord, const SimpleTensor<T1> &in, SimpleTensor<T3> &out, const TensorShape &filter_shape, const T2 *filter_itr, float scale, BorderMode border_mode,
                             T1 constant_border_value = 0);

RawTensor transpose(const RawTensor &src, int chunk_width = 1);
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_VALIDATION_UTILS_H__ */
