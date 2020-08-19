/*
 * Copyright (c) 2017-2020 Arm Limited.
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
template <typename T, typename OT>
OT reduce_operation(const T *ptr, int reduce_elements, ReductionOperation op, int stride)
{
    using type = typename std::remove_cv<OT>::type;
    T res;
    switch(op)
    {
        case ReductionOperation::PROD:
        {
            res = type(1);
        }
        break;
        case ReductionOperation::MIN:
        case ReductionOperation::MAX:
        {
            res = *ptr;
        }
        break;
        default:
        {
            res = type(0);
        }
    }

    if(std::is_integral<type>::value)
    {
        auto int_res = static_cast<int32_t>(res);
        for(int i = 0; i < reduce_elements; ++i)
        {
            auto elem = *(ptr + stride * i);

            switch(op)
            {
                case ReductionOperation::MIN:
                    if(static_cast<T>(int_res) > elem)
                    {
                        int_res = elem;
                    }
                    break;
                case ReductionOperation::MAX:
                    if(static_cast<T>(int_res) < elem)
                    {
                        int_res = elem;
                    }
                    break;
                case ReductionOperation::SUM_SQUARE:
                    int_res += elem * elem;
                    break;
                case ReductionOperation::MEAN_SUM:
                case ReductionOperation::SUM:
                    int_res += elem;
                    break;
                case ReductionOperation::PROD:
                    int_res *= elem;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Operation not supported");
            }
        }
        if(op == ReductionOperation::MEAN_SUM && reduce_elements > 0)
        {
            int_res /= reduce_elements;
        }
        res = static_cast<type>(int_res);
    }
    else
    {
        for(int i = 0; i < reduce_elements; ++i)
        {
            auto elem = *(ptr + stride * i);
            switch(op)
            {
                case ReductionOperation::MIN:
                    if(res > elem)
                    {
                        res = elem;
                    }
                    break;
                case ReductionOperation::MAX:
                    if(res < elem)
                    {
                        res = elem;
                    }
                    break;
                case ReductionOperation::SUM_SQUARE:
                    res += elem * elem;
                    break;
                case ReductionOperation::MEAN_SUM:
                case ReductionOperation::SUM:
                    res += elem;
                    break;
                case ReductionOperation::PROD:
                    res *= elem;
                    break;
                default:
                    ARM_COMPUTE_ERROR("Operation not supported");
            }
        }
        if(op == ReductionOperation::MEAN_SUM && reduce_elements > 0)
        {
            res /= reduce_elements;
        }
    }
    return res;
}

template <typename T, typename OT>
OT reduce_operation_arg_min_max(const T *ptr, int reduce_elements, ReductionOperation op, int stride)
{
    uint32_t res = 0;
    for(int i = 0; i < reduce_elements; ++i)
    {
        auto elem = *(ptr + stride * i);
        switch(op)
        {
            case ReductionOperation::ARG_IDX_MIN:
                if(*(ptr + stride * res) > elem)
                {
                    res = static_cast<uint32_t>(i);
                }
                break;
            case ReductionOperation::ARG_IDX_MAX:
                if(*(ptr + stride * res) < elem)
                {
                    res = static_cast<uint32_t>(i);
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Operation not supported");
        }
    }
    return static_cast<OT>(res);
}

} // namespace

template <typename T, typename OT>
SimpleTensor<OT> compute_reduction_operation(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op)
{
    // Create reference
    const bool         is_arg_min_max   = (op == ReductionOperation::ARG_IDX_MIN || op == ReductionOperation::ARG_IDX_MAX);
    DataType           output_data_type = is_arg_min_max ? DataType::S32 : src.data_type();
    SimpleTensor<OT>   dst{ dst_shape, output_data_type, 1, src.quantization_info() };
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
                dst[du]              = is_arg_min_max ?
                                       reduce_operation_arg_min_max<T, OT>(src_row_ptr, reduce_elems, op, 1) :
                                       reduce_operation<T, OT>(src_row_ptr, reduce_elems, op, 1);
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
                    dst[out_offset]       = is_arg_min_max ?
                                            reduce_operation_arg_min_max<T, OT>(src_row_ptr, reduce_elems, op, src_width) :
                                            reduce_operation<T, OT>(src_row_ptr, reduce_elems, op, src_width);
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
                        dst[out_offset]       = is_arg_min_max ?
                                                reduce_operation_arg_min_max<T, OT>(src_row_ptr, reduce_elems, op, src_width * src_height) :
                                                reduce_operation<T, OT>(src_row_ptr, reduce_elems, op, src_width * src_height);
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
                            dst[out_offset]       = is_arg_min_max ?
                                                    reduce_operation_arg_min_max<T, OT>(src_row_ptr, reduce_elems, op, src_width * src_height * src_depth) :
                                                    reduce_operation<T, OT>(src_row_ptr, reduce_elems, op, src_width * src_height * src_depth);
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

template <typename T, typename OT>
SimpleTensor<OT> reduction_operation(const SimpleTensor<T> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info_output)
{
    ARM_COMPUTE_UNUSED(quantization_info_output);
    return compute_reduction_operation<T, OT>(src, dst_shape, axis, op);
}

template <>
SimpleTensor<uint8_t> reduction_operation(const SimpleTensor<uint8_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info_output)
{
    if(src.data_type() == DataType::QASYMM8)
    {
        // If the operation is MEAN_SUM, we can directly use the uint8 implementation without taking into account scale and offset
        if(op == ReductionOperation::MEAN_SUM && src.quantization_info() == quantization_info_output)
        {
            return compute_reduction_operation<uint8_t, uint8_t>(src, dst_shape, axis, op);
        }
        else
        {
            SimpleTensor<float> src_f = convert_from_asymmetric(src);
            SimpleTensor<float> dst_f = reference::reduction_operation<float, float>(src_f, dst_shape, axis, op);
            return convert_to_asymmetric<uint8_t>(dst_f, quantization_info_output);
        }
    }
    else
    {
        return compute_reduction_operation<uint8_t, uint8_t>(src, dst_shape, axis, op);
    }
}

template <>
SimpleTensor<int8_t> reduction_operation(const SimpleTensor<int8_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info_output)
{
    if(src.data_type() == DataType::QASYMM8_SIGNED)
    {
        // If the operation is MEAN_SUM, we can directly use the int8 implementation without taking into account scale and offset
        if(op == ReductionOperation::MEAN_SUM && src.quantization_info() == quantization_info_output)
        {
            return compute_reduction_operation<int8_t, int8_t>(src, dst_shape, axis, op);
        }
        else
        {
            SimpleTensor<float> src_f = convert_from_asymmetric(src);
            SimpleTensor<float> dst_f = reference::reduction_operation<float, float>(src_f, dst_shape, axis, op);
            return convert_to_asymmetric<int8_t>(dst_f, quantization_info_output);
        }
    }
    else
    {
        return compute_reduction_operation<int8_t, int8_t>(src, dst_shape, axis, op);
    }
}

template SimpleTensor<float> reduction_operation(const SimpleTensor<float> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                 QuantizationInfo quantization_info_output = QuantizationInfo());
template SimpleTensor<half> reduction_operation(const SimpleTensor<half> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                QuantizationInfo quantization_info_output = QuantizationInfo());

template SimpleTensor<int32_t> reduction_operation(const SimpleTensor<float> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                   QuantizationInfo quantization_info_output = QuantizationInfo());
template SimpleTensor<int32_t> reduction_operation(const SimpleTensor<int32_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                   QuantizationInfo quantization_info_output = QuantizationInfo());
template SimpleTensor<int32_t> reduction_operation(const SimpleTensor<half> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                   QuantizationInfo quantization_info_output = QuantizationInfo());
template SimpleTensor<int32_t> reduction_operation(const SimpleTensor<uint8_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                   QuantizationInfo quantization_info_output = QuantizationInfo());
template SimpleTensor<int32_t> reduction_operation(const SimpleTensor<int8_t> &src, const TensorShape &dst_shape, unsigned int axis, ReductionOperation op,
                                                   QuantizationInfo quantization_info_output = QuantizationInfo());

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
