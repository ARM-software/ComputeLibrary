/*
 * Copyright (c) 2018 ARM Limited.
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
#include "Select.h"

#include "arm_compute/core/Types.h"
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
SimpleTensor<T> select(const SimpleTensor<uint8_t> &c, const SimpleTensor<T> &x, const SimpleTensor<T> &y)
{
    // Check if condition has the same rank as c
    const bool is_same_rank = (c.shape().num_dimensions() == x.shape().num_dimensions());

    // Check shapes
    ARM_COMPUTE_ERROR_ON(x.shape() != y.shape());
    ARM_COMPUTE_ERROR_ON(is_same_rank && (x.shape() != c.shape()));
    ARM_COMPUTE_ERROR_ON(!is_same_rank && (c.shape().num_dimensions() > 1) && (c.shape().x() != x.shape()[x.shape().num_dimensions() - 1]));

    // Create reference
    SimpleTensor<T> dst{ x.shape(), x.data_type(), 1 };

    // Run select core
    if(is_same_rank)
    {
        for(int i = 0; i < x.num_elements(); ++i)
        {
            dst[i] = c[i] > 0 ? x[i] : y[i];
        }
    }
    else
    {
        T *output_ptr = dst.data();

        const int outer_size = c.num_elements();
        const int inner_size = x.num_elements() / outer_size;
        size_t    offset     = 0;

        for(int i = 0; i < outer_size; ++i)
        {
            const T *input_ptr = c[i] > 0 ? x.data() : y.data();
            memcpy(output_ptr + offset, input_ptr + offset, inner_size * sizeof(T));
            offset += inner_size;
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> select(const SimpleTensor<uint8_t> &c, const SimpleTensor<uint8_t> &x, const SimpleTensor<uint8_t> &y);
template SimpleTensor<half> select(const SimpleTensor<uint8_t> &c, const SimpleTensor<half> &x, const SimpleTensor<half> &y);
template SimpleTensor<float> select(const SimpleTensor<uint8_t> &c, const SimpleTensor<float> &x, const SimpleTensor<float> &y);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
