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
struct square
{
    T operator()(const T &lhs, const T &rhs) const
    {
        return (lhs + rhs * rhs);
    }
};

template <typename T>
struct sum
{
    T operator()(const T &lhs, const T &rhs) const
    {
        return (lhs + rhs);
    }
};

template <typename T>
T reduce_operation(T *ptr, int reduce_elements, ReductionOperation op)
{
    switch(op)
    {
        case ReductionOperation::SUM_SQUARE:
            return std::accumulate(ptr, ptr + reduce_elements, static_cast<T>(0), square<T>());
        case ReductionOperation::SUM:
        case ReductionOperation::MEAN_SUM:
            return std::accumulate(ptr, ptr + reduce_elements, static_cast<T>(0), sum<T>());
        default:
            ARM_COMPUTE_ERROR("Unsupported reduction operation");
    }
}
} // namespace

template <typename T>
SimpleTensor<T> reduction_operation(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op)
{
    // Create reference
    SimpleTensor<T>    dst{ dst_shape, src.data_type() };
    const unsigned int src_width  = src.shape().x();
    const unsigned int src_height = src.shape().y();
    const unsigned int src_depth  = src.shape().z();
    const unsigned int src_batch  = src.shape()[3];
    const bool         mean       = op == ReductionOperation::MEAN_SUM;

    switch(axis)
    {
        case 0:
        {
            const int          reduce_elems = src.shape()[axis];
            const unsigned int upper_dims   = src.shape().total_size_upper(1);
            for(unsigned int du = 0; du < upper_dims; ++du)
            {
                if(std::is_integral<T>::value)
                {
                    uint32_t res = 0;
                    for(unsigned int x = 0; x < src_width; ++x)
                    {
                        res += static_cast<uint32_t>(src[du * src_width + x]);
                    }
                    if(mean && src_width > 0)
                    {
                        res /= src_width;
                    }
                    dst[du] = static_cast<uint8_t>(res);
                }
                else
                {
                    const T *src_row_ptr = src.data() + du * reduce_elems;

                    auto res = reduce_operation(src_row_ptr, reduce_elems, op);
                    if(mean && src_width > 0)
                    {
                        res /= src_width;
                    }
                    dst[du] = res;
                }
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
                    if(std::is_integral<T>::value)
                    {
                        uint32_t res = 0;
                        for(unsigned int y = 0; y < src_height; ++y)
                        {
                            res += static_cast<uint32_t>(src[du * src_height * src_width + y * src_width + x]);
                        }
                        if(mean && src_height > 0)
                        {
                            res /= src_height;
                        }
                        dst[du * src_width + x] = static_cast<uint8_t>(res);
                    }
                    else
                    {
                        auto res = T(0);
                        for(unsigned int y = 0; y < src_height; ++y)
                        {
                            res += src[du * src_height * src_width + y * src_width + x];
                        }
                        if(mean && src_height > 0)
                        {
                            res /= src_height;
                        }
                        dst[du * src_width + x] = res;
                    }
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
                        if(std::is_integral<T>::value)
                        {
                            uint32_t res = T(0);
                            for(unsigned int z = 0; z < src_depth; ++z)
                            {
                                res += static_cast<uint32_t>(src[du * src_depth * src_height * src_width + z * src_height * src_width + y * src_width + x]);
                            }
                            if(mean && src_depth > 0)
                            {
                                res /= src_depth;
                            }
                            dst[du * src_width * src_height + y * src_width + x] = static_cast<uint8_t>(res);
                        }
                        else
                        {
                            auto res = T(0);
                            for(unsigned int z = 0; z < src_depth; ++z)
                            {
                                res += src[du * src_depth * src_height * src_width + z * src_height * src_width + y * src_width + x];
                            }
                            if(mean && src_depth > 0)
                            {
                                res /= src_depth;
                            }
                            dst[du * src_width * src_height + y * src_width + x] = res;
                        }
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
                            if(std::is_integral<T>::value)
                            {
                                uint32_t res = 0;
                                for(unsigned int w = 0; w < src_batch; ++w)
                                {
                                    res += static_cast<uint32_t>(src[du * src_batch * src_depth * src_height * src_width + w * src_width * src_height * src_depth + z * src_width * src_height + y * src_width + x]);
                                }
                                if(mean && src_batch > 0)
                                {
                                    res /= src_batch;
                                }

                                dst[du * src_depth * src_height * src_width + z * src_width * src_height + y * src_width + x] = static_cast<uint8_t>(res);
                            }
                            else
                            {
                                auto res = T(0);
                                for(unsigned int w = 0; w < src_batch; ++w)
                                {
                                    res += src[du * src_batch * src_depth * src_height * src_width + w * src_width * src_height * src_depth + z * src_width * src_height + y * src_width + x];
                                }
                                if(mean && src_batch > 0)
                                {
                                    res /= src_batch;
                                }

                                dst[du * src_depth * src_height * src_width + z * src_width * src_height + y * src_width + x] = res;
                            }
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
