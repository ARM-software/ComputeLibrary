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
#include "WeightsReshape.h"

#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> weights_reshape(const SimpleTensor<T> &src, const SimpleTensor<T> &biases, const TensorShape &dst_shape, const unsigned int num_groups)
{
    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1 };

    // Compute reference
    const bool   has_bias  = biases.size() > 0;
    const size_t linear_sz = src.shape().total_size_lower(3);
    const size_t group_sz  = src.shape()[3] / num_groups;

    for(size_t g = 0; g < num_groups; ++g)
    {
        for(size_t w = 0; w < group_sz; ++w)
        {
            const size_t curr_weight = g * group_sz + w;

            size_t i = 0;
            for(; i < linear_sz; ++i)
            {
                dst[coord2index(dst.shape(), Coordinates(w, i, g))] = src[curr_weight * linear_sz + i];
            }
            if(has_bias)
            {
                dst[coord2index(dst.shape(), Coordinates(w, i, g))] = static_cast<T>(biases[curr_weight]);
            }
        }
    }

    return dst;
}

template SimpleTensor<float> weights_reshape(const SimpleTensor<float> &src, const SimpleTensor<float> &biases, const TensorShape &dst_shape, const unsigned int num_groups);
template SimpleTensor<half> weights_reshape(const SimpleTensor<half> &src, const SimpleTensor<half> &biases, const TensorShape &dst_shape, const unsigned int num_groups);
template SimpleTensor<uint8_t> weights_reshape(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint8_t> &biases, const TensorShape &dst_shape, const unsigned int num_groups);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
