/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "tests/validation/reference/Logical.h"
#include "src/core/KernelTypes.h"
#include "tests/framework/Asserts.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
T logical_binary_op(arm_compute::LogicalOperation op, T src1, T src2)
{
    switch(op)
    {
        case arm_compute::LogicalOperation::And:
            return src1 && src2;
        case arm_compute::LogicalOperation::Or:
            return src1 || src2;
        // The following operators are either invalid or not binary operator
        case arm_compute::LogicalOperation::Not:
        // fall through
        case arm_compute::LogicalOperation::Unknown:
        // fall through
        default:
            ARM_COMPUTE_ASSERT(true);
    }
    return T{};
}

template <size_t dim>
struct BroadcastUnroll
{
    template <typename T>
    static void unroll(arm_compute::LogicalOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        const bool src1_is_broadcast = (src1.shape()[dim - 1] != dst.shape()[dim - 1]);
        const bool src2_is_broadcast = (src2.shape()[dim - 1] != dst.shape()[dim - 1]);

        id_src1.set(dim - 1, 0);
        id_src2.set(dim - 1, 0);
        id_dst.set(dim - 1, 0);
#if defined(_OPENMP)
        #pragma omp parallel for
#endif /* _OPENMP */
        for(size_t i = 0; i < dst.shape()[dim - 1]; ++i)
        {
            BroadcastUnroll < dim - 1 >::unroll(op, src1, src2, dst, id_src1, id_src2, id_dst);

            id_src1[dim - 1] += !src1_is_broadcast;
            id_src2[dim - 1] += !src2_is_broadcast;
            ++id_dst[dim - 1];
        }
    }
};

template <>
struct BroadcastUnroll<0>
{
    template <typename T>
    static void unroll(arm_compute::LogicalOperation op, const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        dst[coord2index(dst.shape(), id_dst)] = logical_binary_op(op, src1[coord2index(src1.shape(), id_src1)], src2[coord2index(src2.shape(), id_src2)]);
    }
};

template <typename T>
SimpleTensor<T> logical_or(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2)
{
    Coordinates     id_src1{};
    Coordinates     id_src2{};
    Coordinates     id_dst{};
    SimpleTensor<T> dst{ TensorShape::broadcast_shape(src1.shape(), src2.shape()), src1.data_type() };

    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(arm_compute::LogicalOperation::Or, src1, src2, dst, id_src1, id_src2, id_dst);

    return dst;
}

template <typename T>
SimpleTensor<T> logical_and(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2)
{
    Coordinates     id_src1{};
    Coordinates     id_src2{};
    Coordinates     id_dst{};
    SimpleTensor<T> dst{ TensorShape::broadcast_shape(src1.shape(), src2.shape()), src1.data_type() };

    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(arm_compute::LogicalOperation::And, src1, src2, dst, id_src1, id_src2, id_dst);

    return dst;
}

template <typename T>
SimpleTensor<T> logical_not(const SimpleTensor<T> &src)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());
#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = !src[i];
    }

    return dst;
}

template SimpleTensor<uint8_t> logical_or(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2);
template SimpleTensor<uint8_t> logical_and(const SimpleTensor<uint8_t> &src1, const SimpleTensor<uint8_t> &src2);
template SimpleTensor<uint8_t> logical_not(const SimpleTensor<uint8_t> &src1);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
