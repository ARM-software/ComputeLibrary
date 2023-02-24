/*
 * Copyright (c) 2018-2019, 2022-2023 Arm Limited.
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

#include "Gather.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
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
SimpleTensor<T> gather(const SimpleTensor<T> &src, const SimpleTensor<uint32_t> &indices, uint32_t actual_axis)
{
    const TensorShape dst_shape   = arm_compute::misc::shape_calculator::compute_gather_shape(src.shape(), indices.shape(), actual_axis);
    SimpleTensor<T>   dst(dst_shape, src.data_type());

    const auto        src_ptr     = static_cast<const T *>(src.data());
    const auto        indices_ptr = static_cast<const uint32_t *>(indices.data());
    const auto        dst_ptr     = static_cast<T *>(dst.data());

    Window win;
    win.use_tensor_dimensions(dst_shape);

    execute_window_loop(win, [&](const Coordinates &dst_coords) {
        // Calculate the coordinates of the index value.
        Coordinates idx_coords;

        for(size_t i = 0; i < indices.shape().num_dimensions(); ++i)
        {
            idx_coords.set(i, dst_coords[i + actual_axis]);
        }

        // Calculate the coordinates of the source data.
        Coordinates src_coords;

        for(size_t i = 0; i < actual_axis; ++i)
        {
            src_coords.set(i, dst_coords[i]);
        }

        src_coords.set(actual_axis, indices_ptr[coords2index(indices.shape(), idx_coords)]);

        for(size_t i = actual_axis + 1; i < src.shape().num_dimensions(); ++i)
        {
            src_coords.set(i, dst_coords[i + indices.shape().num_dimensions() - 1]);
        }

        // Copy the data.
        dst_ptr[coords2index(dst.shape(), dst_coords)] = src_ptr[coords2index(src.shape(), src_coords)];
    });

    return dst;
}

template SimpleTensor<float> gather(const SimpleTensor<float> &src, const SimpleTensor<uint32_t> &indices, uint32_t actual_axis);
template SimpleTensor<half> gather(const SimpleTensor<half> &src, const SimpleTensor<uint32_t> &indices, uint32_t actual_axis);
template SimpleTensor<uint16_t> gather(const SimpleTensor<uint16_t> &src, const SimpleTensor<uint32_t> &indices, uint32_t actual_axis);
template SimpleTensor<uint8_t> gather(const SimpleTensor<uint8_t> &src, const SimpleTensor<uint32_t> &indices, uint32_t actual_axis);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
