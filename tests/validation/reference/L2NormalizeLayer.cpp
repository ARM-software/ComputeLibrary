/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "L2NormalizeLayer.h"
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
TensorShape get_output_shape(TensorShape shape, unsigned int axis)
{
    TensorShape output_shape(shape);
    output_shape.set(axis, 1);
    return output_shape;
}
} // namespace

template <typename T>
SimpleTensor<T> l2_normalize(const SimpleTensor<T> &src, unsigned int axis, float epsilon)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type() };

    // Reduce across given axis
    SimpleTensor<T> sum = reduction_operation<T, T>(src, get_output_shape(src.shape(), axis), axis, ReductionOperation::SUM_SQUARE);

    // Compute reference
    const int upper_dims     = src.shape().total_size_upper(axis + 1);
    const int lower_dims     = src.shape().total_size_lower(axis + 1);
    const int lower_dims_sum = sum.shape().total_size_lower(axis + 1);

    for(int du = 0; du < upper_dims; ++du)
    {
        const T *src_row_ptr = src.data() + du * lower_dims;
        T       *dst_row_ptr = dst.data() + du * lower_dims;
        switch(axis)
        {
            case 0:
            {
                const int elems               = src.shape()[0];
                const T   normalization_value = sqrt(std::max(sum[du], static_cast<T>(epsilon)));
                std::transform(src_row_ptr, src_row_ptr + elems, dst_row_ptr, [normalization_value](T val)
                {
                    return val / normalization_value;
                });
            }
            break;
            case 1:
            case 2:
            {
                for(int ld = 0; ld < lower_dims; ++ld)
                {
                    const T normalization_value = sqrt(std::max(sum[ld % lower_dims_sum + du * lower_dims_sum], static_cast<T>(epsilon)));
                    dst_row_ptr[ld]             = src_row_ptr[ld] / normalization_value;
                }
            }
            break;
            default:
                ARM_COMPUTE_ERROR("Axis not supported");
        }
    }

    return dst;
}

template SimpleTensor<float> l2_normalize(const SimpleTensor<float> &src, unsigned int axis, float epsilon);
template SimpleTensor<half> l2_normalize(const SimpleTensor<half> &src, unsigned int axis, float epsilon);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
