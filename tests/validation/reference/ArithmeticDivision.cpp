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
#include "ArithmeticDivision.h"

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
template <size_t dim>
struct BroadcastUnroll
{
    template <typename T>
    static void unroll(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        const bool src1_is_broadcast = (src1.shape()[dim - 1] != dst.shape()[dim - 1]);
        const bool src2_is_broadcast = (src2.shape()[dim - 1] != dst.shape()[dim - 1]);

        id_src1.set(dim - 1, 0);
        id_src2.set(dim - 1, 0);
        id_dst.set(dim - 1, 0);

        for(size_t i = 0; i < dst.shape()[dim - 1]; ++i, ++id_dst[dim - 1])
        {
            BroadcastUnroll < dim - 1 >::unroll(src1, src2, dst, id_src1, id_src2, id_dst);

            id_src1[dim - 1] += !src1_is_broadcast;
            id_src2[dim - 1] += !src2_is_broadcast;
        }
    }
};

template <>
struct BroadcastUnroll<0>
{
    template <typename T>
    static void unroll(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, SimpleTensor<T> &dst,
                       Coordinates &id_src1, Coordinates &id_src2, Coordinates &id_dst)
    {
        dst[coord2index(dst.shape(), id_dst)] = src1[coord2index(src1.shape(), id_src1)] / src2[coord2index(src2.shape(), id_src2)];
    }
};
} // namespace

template <typename T>
SimpleTensor<T> arithmetic_division(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, DataType data_type)
{
    SimpleTensor<T> dst(TensorShape::broadcast_shape(src1.shape(), src2.shape()), data_type);

    Coordinates id_src1{};
    Coordinates id_src2{};
    Coordinates id_dst{};

    BroadcastUnroll<Coordinates::num_max_dimensions>::unroll(src1, src2, dst, id_src1, id_src2, id_dst);

    return dst;
}

template SimpleTensor<half> arithmetic_division(const SimpleTensor<half> &src1, const SimpleTensor<half> &src2, DataType data_type);
template SimpleTensor<float> arithmetic_division(const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, DataType data_type);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
