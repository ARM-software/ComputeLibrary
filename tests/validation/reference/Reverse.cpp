/*
 * Copyright (c) 2018-2020, 2023 Arm Limited.
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
#include "Reverse.h"

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
template <typename T>
SimpleTensor<T> reverse(const SimpleTensor<T> &src, const SimpleTensor<int32_t> &axis, bool use_inverted_axis)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(axis.shape().num_dimensions() > 1);
    ARM_COMPUTE_ERROR_ON(axis.shape().x() > 4);

    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type(), src.num_channels(), src.quantization_info() };

    const unsigned int width   = src.shape()[0];
    const unsigned int height  = src.shape()[1];
    const unsigned int depth   = src.shape()[2];
    const unsigned int batches = src.shape()[3];

    const int rank = src.shape().num_dimensions();

    std::array<bool, 4> to_reverse = { { false, false, false, false } };
    for(int i = 0; i < axis.num_elements(); ++i)
    {
        int axis_i = axis[i];

        // The values of axis tensor must be between [-rank, rank-1].
        if((axis_i < -rank) || (axis_i >= rank))
        {
            ARM_COMPUTE_ERROR("the values of the axis tensor must be within [-rank, rank-1].");
        }

        // In case of negative axis value i.e targeted axis(i) = rank + axis(i)
        if(axis_i < 0)
        {
            axis_i = rank + axis_i;
        }

        // Reverse ACL axis indices convention i.e. (inverted)axis = (tensor_rank - 1) - axis
        if(use_inverted_axis)
        {
            axis_i = (rank - 1) - axis_i;
        }

        to_reverse[axis_i] = true;
    }

    const uint32_t num_elements = src.num_elements();

#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(uint32_t i = 0; i < num_elements; ++i)
    {
        const Coordinates  src_coord = index2coord(src.shape(), i);
        const unsigned int dst_x     = to_reverse[0] ? width - src_coord[0] - 1 : src_coord[0];
        const unsigned int dst_y     = to_reverse[1] ? height - src_coord[1] - 1 : src_coord[1];
        const unsigned int dst_z     = to_reverse[2] ? depth - src_coord[2] - 1 : src_coord[2];
        const unsigned int dst_w     = to_reverse[3] ? batches - src_coord[3] - 1 : src_coord[3];

        dst[coord2index(src.shape(), Coordinates(dst_x, dst_y, dst_z, dst_w))] = src[i];
    }

    return dst;
}

template SimpleTensor<uint8_t> reverse(const SimpleTensor<uint8_t> &src, const SimpleTensor<int32_t> &axis, bool use_inverted_axis);
template SimpleTensor<half> reverse(const SimpleTensor<half> &src, const SimpleTensor<int32_t> &axis, bool use_inverted_axis);
template SimpleTensor<float> reverse(const SimpleTensor<float> &src, const SimpleTensor<int32_t> &axis, bool use_inverted_axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
