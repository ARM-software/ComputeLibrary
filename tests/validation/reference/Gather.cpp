/*
 * Copyright (c) 2018-2019, 2022 Arm Limited.
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
    const auto       *indices_ptr = static_cast<const uint32_t *>(indices.data());
    const TensorShape dst_shape   = arm_compute::misc::shape_calculator::compute_gather_shape(src.shape(), indices.shape(), actual_axis);
    SimpleTensor<T>   dst(dst_shape, src.data_type());

    Window win;
    win.use_tensor_dimensions(dst_shape);
    if(indices.shape().num_dimensions() == 1u)
    {
        execute_window_loop(win, [&](const Coordinates & id)
        {
            Coordinates offset;
            for(unsigned int dim = 0; dim < id.num_dimensions(); ++dim)
            {
                if(dim == actual_axis)
                {
                    offset.set(dim, indices_ptr[id[dim]]);
                }
                else
                {
                    offset.set(dim, id[dim]);
                }
            }
            *reinterpret_cast<T *>(dst(id)) = *reinterpret_cast<const T *>(src(offset));
        });
    }
    else
    {
        if(actual_axis == 0)
        {
            win.set(Window::DimX, Window::Dimension(0, 1, 1));
            uint32_t index = 0;
            execute_window_loop(win, [&](const Coordinates & id)
            {
                auto     *dst_ptr     = reinterpret_cast<T *>(dst(id));
                const int row_to_copy = indices[index++];
                std::copy_n(src.data() + row_to_copy * src.shape()[0], src.shape()[0], dst_ptr);
            });
        }
    }

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
