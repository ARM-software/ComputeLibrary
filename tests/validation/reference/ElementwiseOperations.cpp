/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "ElementwiseOperations.h"

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
namespace
{
template <typename T>
T arithm_op(ArithmeticOperation op, T src1, T src2, ConvertPolicy convert_policy)
{
    using intermediate_type = typename common_promoted_signed_type<T>::intermediate_type;

    intermediate_type val;

    if(op == ArithmeticOperation::ADD)
    {
        val = static_cast<intermediate_type>(src1) + static_cast<intermediate_type>(src2);
    }
    else if(op == ArithmeticOperation::SUB)
    {
        val = static_cast<intermediate_type>(src1) - static_cast<intermediate_type>(src2);
    }
    else if(op == ArithmeticOperation::MIN)
    {
        val = std::min(static_cast<intermediate_type>(src1), static_cast<intermediate_type>(src2));
    }
    else if(op == ArithmeticOperation::MAX)
    {
        val = std::max(static_cast<intermediate_type>(src1), static_cast<intermediate_type>(src2));
    }
    else if(op == ArithmeticOperation::SQUARED_DIFF)
    {
        intermediate_type tmp = (static_cast<intermediate_type>(src1) - static_cast<intermediate_type>(src2));
        val                   = tmp * tmp;
    }
    else if(op == ArithmeticOperation::DIV)
    {
        val = (static_cast<intermediate_type>(src1) / static_cast<intermediate_type>(src2));
    }
    else
    {
        ARM_COMPUTE_ERROR("Not handled");
    }

    T result;
    if(op == ArithmeticOperation::ADD || op == ArithmeticOperation::SUB || op == ArithmeticOperation::DIV)
    {
        result = (convert_policy == ConvertPolicy::SATURATE) ? saturate_cast<T>(val) : static_cast<T>(val);
    }
    else
    {
        result = static_cast<T>(val);
    }
    return result;
}

template <size_t dim>
struct BroadcastUnroll
{
    template <typename T>
    static void unroll(ArithmeticOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       ConvertPolicy convert_policy, Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        const bool src1_is_broadcast = (src1.shape()[dim - 1] != dst.shape()[dim - 1]);
        const bool src2_is_broadcast = (src2.shape()[dim - 1] != dst.shape()[dim - 1]);

        id_src1.set(dim - 1, 0);
        id_src2.set(dim - 1, 0);
        id_dst.set(dim - 1, 0);

        for(size_t i = 0; i < dst.shape()[dim - 1]; ++i, ++id_dst[dim - 1])
        {
            BroadcastUnroll < dim - 1 >::unroll(op, src1, src2, dst, convert_policy, id_src1, id_src2, id_dst);

            id_src1[dim - 1] += !src1_is_broadcast;
            id_src2[dim - 1] += !src2_is_broadcast;
        }
    }
};

template <>
struct BroadcastUnroll<0>
{
    template <typename T>
    static void unroll(ArithmeticOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       ConvertPolicy convert_policy, Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        dst[coord2index(dst.shape(), id_dst)] = arithm_op(op, src1[coord2index(src1.shape(), id_src1)], src2[coord2index(src2.shape(), id_src2)], convert_policy);
    }
};
} // namespace

template <typename T>
SimpleTensor<T> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst, ConvertPolicy convert_policy)
{
    Coordinates id_src1, id_src2, id_dst;

    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1, src2, dst, convert_policy, id_src1, id_src2, id_dst);

    return dst;
}

template <>
SimpleTensor<uint8_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2, SimpleTensor<uint8_t> &dst, ConvertPolicy convert_policy)
{
    if(dst.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> src1_tmp = convert_from_asymmetric(src1);
        SimpleTensor<float> src2_tmp = convert_from_asymmetric(src2);
        SimpleTensor<float> dst_tmp(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dst.data_type());

        Coordinates id_src1, id_src2, id_dst;

        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1_tmp, src2_tmp, dst_tmp, convert_policy, id_src1, id_src2, id_dst);

        dst = convert_to_asymmetric(dst_tmp, dst.quantization_info());
        return dst;
    }
    else
    {
        // DataType::U8
        Coordinates id_src1, id_src2, id_dst;

        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1, src2, dst, convert_policy, id_src1, id_src2, id_dst);

        return dst;
    }
}

template SimpleTensor<int32_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int32_t> &src1, const SimpleTensor<int32_t> &src2, SimpleTensor<int32_t> &dst,
                                                    ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, SimpleTensor<int16_t> &dst,
                                                    ConvertPolicy convert_policy);
template SimpleTensor<int8_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int8_t> &src1, const SimpleTensor<int8_t> &src2, SimpleTensor<int8_t> &dst,
                                                   ConvertPolicy convert_policy);
template SimpleTensor<half> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<half> &src1, const SimpleTensor<half> &src2, SimpleTensor<half> &dst, ConvertPolicy convert_policy);
template SimpleTensor<float> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, SimpleTensor<float> &dst, ConvertPolicy convert_policy);

template <typename T>
SimpleTensor<T> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, DataType dst_data_type, ConvertPolicy convert_policy)
{
    ARM_COMPUTE_ERROR_ON_MSG(dst_data_type == DataType::QASYMM8, "For QASYMM8, the quantized output tensor should be passed directly.");

    SimpleTensor<T> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), dst_data_type);
    arithmetic_operation<T>(op, src1, src2, dst, convert_policy);
    return dst;
}

template SimpleTensor<int32_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int32_t> &src1, const SimpleTensor<int32_t> &src2, DataType dst_data_type,
                                                    ConvertPolicy convert_policy);
template SimpleTensor<int16_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int16_t> &src1, const SimpleTensor<int16_t> &src2, DataType dst_data_type,
                                                    ConvertPolicy convert_policy);
template SimpleTensor<int8_t> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<int8_t> &src1, const SimpleTensor<int8_t> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<half> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<half> &src1, const SimpleTensor<half> &src2, DataType dst_data_type, ConvertPolicy convert_policy);
template SimpleTensor<float> arithmetic_operation(ArithmeticOperation op, const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, DataType dst_data_type, ConvertPolicy convert_policy);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
