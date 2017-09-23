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
T bilinear_policy(const SimpleTensor<T> &in, Coordinates id, float xn, float yn, BorderMode border_mode, T constant_border_value);

/* Apply 2D spatial filter on a single element of @p in at coordinates @p coord
 *
 * - filter sizes have to be odd number
 * - Row major order of filter assumed
 * - TO_ZERO rounding policy assumed
 * - SATURATE convert policy assumed
 */
template <typename T, typename U, typename V>
void apply_2d_spatial_filter(Coordinates coord, const SimpleTensor<T> &src, SimpleTensor<U> &dst, const TensorShape &filter_shape, const V *filter_itr, double scale, BorderMode border_mode,
                             T constant_border_value = T(0))
{
    double    val = 0.;
    const int x   = coord.x();
    const int y   = coord.y();
    for(int j = y - static_cast<int>(filter_shape[1] / 2); j <= y + static_cast<int>(filter_shape[1] / 2); ++j)
    {
        for(int i = x - static_cast<int>(filter_shape[0] / 2); i <= x + static_cast<int>(filter_shape[0] / 2); ++i)
        {
            coord.set(0, i);
            coord.set(1, j);
            val += static_cast<double>(*filter_itr) * tensor_elem_at(src, coord, border_mode, constant_border_value);
            ++filter_itr;
        }
    }
    coord.set(0, x);
    coord.set(1, y);
    dst[coord2index(src.shape(), coord)] = saturate_cast<U>(support::cpp11::trunc(val * scale));
}

RawTensor transpose(const RawTensor &src, int chunk_width = 1);
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_VALIDATION_UTILS_H__ */
