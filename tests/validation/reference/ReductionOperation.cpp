/*
 * Copyright (c) 2017-2018 ARM Limited.
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
T reduce_operation(T *ptr, int reduce_elements, ReductionOperation op, int stride)
{
    using type = typename std::remove_cv<T>::type;
    auto res   = type(0);

    if(std::is_integral<type>::value)
    {
        uint32_t int_res = 0;
        for(int i = 0; i < reduce_elements; ++i)
        {
            auto elem = static_cast<uint32_t>(*(ptr + stride * i));
            int_res += (op == ReductionOperation::SUM_SQUARE) ? elem * elem : elem;
        }
        if(op == ReductionOperation::MEAN_SUM && reduce_elements > 0)
        {
            int_res /= reduce_elements;
        }
        res = saturate_cast<type>(int_res);
    }
    else
    {
        for(int i = 0; i < reduce_elements; ++i)
        {
            auto elem = *(ptr + stride * i);
            res += (op == ReductionOperation::SUM_SQUARE) ? elem * elem : elem;
        }
        if(op == ReductionOperation::MEAN_SUM && reduce_elements > 0)
        {
            res /= reduce_elements;
        }
    }

    return res;
}
} // namespace

template <typename T>
SimpleTensor<T> reduction_operation(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op)
{
    // Create reference
    SimpleTensor<T>    dst{ dst_shape, src.data_type(), 1, src.quantization_info() };
    const unsigned int src_width    = src.shape().x();
    const unsigned int src_height   = src.shape().y();
    const unsigned int src_depth    = src.shape().z();
    const unsigned int src_batch    = src.shape()[3];
    const int          reduce_elems = src.shape()[axis];

    switch(axis)
    {
        case 0:
        {
            const unsigned int upper_dims = src.shape().total_size_upper(1);
            for(unsigned int du = 0; du < upper_dims; ++du)
            {
                const T *src_row_ptr = src.data() + du * reduce_elems;
                auto     res         = reduce_operation(src_row_ptr, reduce_elems, op, 1);
                dst[du]              = res;
            }
        }
        break;
        case 1:
        {
            const unsigned int upper_dims = src.shape().total_size_upper(2);
            for(unsigned int du = 0; du < upper_dims; ++du)
            {
                for(unsigned int x = 0; x < src_width; ++x)
                {
                    const int in_offset   = du * src_height * src_width + x;
                    const int out_offset  = du * src_width + x;
                    const T *src_row_ptr = src.data() + in_offset;
                    auto      res         = reduce_operation(src_row_ptr, reduce_elems, op, src_width);
                    dst[out_offset]       = res;
                }
            }
        }
        break;
        case 2:
        {
            const unsigned int upper_dims = src.shape().total_size_upper(3);
            for(unsigned int du = 0; du < upper_dims; ++du)
            {
                for(unsigned int x = 0; x < src_width; ++x)
                {
                    for(unsigned int y = 0; y < src_height; ++y)
                    {
                        const int in_offset   = du * src_depth * src_height * src_width + y * src_width + x;
                        const int out_offset  = du * src_width * src_height + y * src_width + x;
                        const T *src_row_ptr = src.data() + in_offset;
                        auto      res         = reduce_operation(src_row_ptr, reduce_elems, op, src_height * src_width);
                        dst[out_offset]       = res;
                    }
                }
            }
        }
        break;
        case 3:
        {
            const unsigned int upper_dims = src.shape().total_size_upper(4);
            for(unsigned int du = 0; du < upper_dims; ++du)
            {
                for(unsigned int z = 0; z < src_depth; ++z)
                {
                    for(unsigned int y = 0; y < src_height; ++y)
                    {
                        for(unsigned int x = 0; x < src_width; ++x)
                        {
                            const int in_offset   = du * src_batch * src_depth * src_height * src_width + z * src_width * src_height + y * src_width + x;
                            const int out_offset  = du * src_depth * src_height * src_width + z * src_width * src_height + y * src_width + x;
                            const T *src_row_ptr = src.data() + in_offset;
                            auto      res         = reduce_operation(src_row_ptr, reduce_elems, op, src_width * src_height * src_depth);
                            dst[out_offset]       = res;
                        }
                    }
                }
            }
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction axis");
    }

    return dst;
}

template SimpleTensor<float> reduction_operation(const SimpleTensor<float> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op);
template SimpleTensor<half> reduction_operation(const SimpleTensor<half> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op);
template SimpleTensor<uint8_t> reduction_operation(const SimpleTensor<uint8_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
