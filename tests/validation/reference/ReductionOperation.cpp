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
#include "ReductionOperation.h"

#include "tests/validation/Helpers.h"

#include <algorithm>
#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
template <typename T>
struct square
{
    T operator()(const T &lhs, const T &rhs) const
    {
        return (lhs + rhs * rhs);
    }
};

template <typename T>
T reduce_operation(T *ptr, int reduce_elements, ReductionOperation op)
{
    switch(op)
    {
        case ReductionOperation::SUM_SQUARE:
            return std::accumulate(ptr, ptr + reduce_elements, 0.f, square<T>());
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation");
    }
}
} // namespace

template <typename T>
SimpleTensor<T> reduction_operation(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op)
{
    // Create reference
    SimpleTensor<T> dst{ dst_shape, src.data_type() };

    // Compute reference
    const int reduce_elems = src.shape()[axis];
    const int upper_dims   = src.shape().total_size_upper(axis + 1);

    for(int du = 0; du < upper_dims; ++du)
    {
        if(axis == 0)
        {
            const T *src_row_ptr = src.data() + du * reduce_elems;
            dst[du]              = reduce_operation(src_row_ptr, reduce_elems, op);
        }
        else
        {
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
        }
    }

    return dst;
}

template SimpleTensor<float> reduction_operation(const SimpleTensor<float> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
