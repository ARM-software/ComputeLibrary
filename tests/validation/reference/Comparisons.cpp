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
#include "Comparisons.h"

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
uint8_t compare_op(ComparisonOperation op, T src1, T src2)
{
    uint8_t result = 0;
    switch(op)
    {
        case ComparisonOperation::Equal:
            result = static_cast<uint8_t>(src1 == src2);
            break;
        case ComparisonOperation::NotEqual:
            result = static_cast<uint8_t>(src1 != src2);
            break;
        case ComparisonOperation::GreaterEqual:
            result = static_cast<uint8_t>(src1 >= src2);
            break;
        case ComparisonOperation::Greater:
            result = static_cast<uint8_t>(src1 > src2);
            break;
        case ComparisonOperation::LessEqual:
            result = static_cast<uint8_t>(src1 <= src2);
            break;
        case ComparisonOperation::Less:
            result = static_cast<uint8_t>(src1 < src2);
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported operation");
    }
    return (result != 0) ? 255 : 0;
}

template <size_t dim>
struct BroadcastUnroll
{
    template <typename T>
    static void unroll(ComparisonOperation    op,
                       const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<uint8_t> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        const bool src1_is_broadcast = (src1.shape()[dim - 1] != dst.shape()[dim - 1]);
        const bool src2_is_broadcast = (src2.shape()[dim - 1] != dst.shape()[dim - 1]);

        id_src1.set(dim - 1, 0);
        id_src2.set(dim - 1, 0);
        id_dst.set(dim - 1, 0);

        for(size_t i = 0; i < dst.shape()[dim - 1]; ++i, ++id_dst[dim - 1])
        {
            BroadcastUnroll < dim - 1 >::unroll(op, src1, src2, dst, id_src1, id_src2, id_dst);

            id_src1[dim - 1] += !src1_is_broadcast;
            id_src2[dim - 1] += !src2_is_broadcast;
        }
    }
};

template <>
struct BroadcastUnroll<0>
{
    template <typename T>
    static void unroll(ComparisonOperation    op,
                       const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<uint8_t> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        dst[coord2index(dst.shape(), id_dst)] = compare_op(op, src1[coord2index(src1.shape(), id_src1)], src2[coord2index(src2.shape(), id_src2)]);
    }
};
} // namespace

template <typename T>
SimpleTensor<uint8_t> compare(ComparisonOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2)
{
    SimpleTensor<uint8_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), DataType::U8);

    Coordinates id_src1{};
    Coordinates id_src2{};
    Coordinates id_dst{};
    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1, src2, dst, id_src1, id_src2, id_dst);
    return dst;
}

template <>
SimpleTensor<uint8_t> compare(ComparisonOperation op, const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2)
{
    SimpleTensor<uint8_t> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), DataType::U8);

    Coordinates id_src1{};
    Coordinates id_src2{};
    Coordinates id_dst{};

    if(src1.data_type() == DataType::QASYMM8)
    {
        SimpleTensor<float> src1_tmp = convert_from_asymmetric(src1);
        SimpleTensor<float> src2_tmp = convert_from_asymmetric(src2);
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1_tmp, src2_tmp, dst, id_src1, id_src2, id_dst);
    }
    else
    {
        // DataType::U8
        BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(op, src1, src2, dst, id_src1, id_src2, id_dst);
    }
    return dst;
}

template SimpleTensor<uint8_t> compare(ComparisonOperation op, const SimpleTensor<half> &src1, const SimpleTensor<half> &src2);
template SimpleTensor<uint8_t> compare(ComparisonOperation op, const SimpleTensor<float> &src1, const SimpleTensor<float> &src2);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
