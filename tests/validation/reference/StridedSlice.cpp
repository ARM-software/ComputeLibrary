/*
 * Copyright (c) 2018 ARM Limited.
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
#include "StridedSlice.h"

#include "arm_compute/core/utils/helpers/tensor_transform.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> strided_slice(const SimpleTensor<T> &src,
                              Coordinates starts, Coordinates ends, BiStrides strides,
                              int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    using namespace arm_compute::helpers::tensor_transform;

    // Validation checks
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON(starts.num_dimensions() > src.shape().num_dimensions());
    ARM_COMPUTE_ERROR_ON(ends.num_dimensions() > src.shape().num_dimensions());
    ARM_COMPUTE_ERROR_ON(strides.num_dimensions() > src.shape().num_dimensions());
    for(unsigned int i = 0; i < strides.num_dimensions(); ++i)
    {
        ARM_COMPUTE_ERROR_ON(strides[i] == 0);
    }

    // Get source shape
    const TensorShape &src_shape = src.shape();

    // Get actual start, end coordinates and strides
    const Coordinates final_strides = strided_slice_strides(src_shape, strides);
    const Coordinates starts_abs    = strided_slice_absolute_start_coords(src_shape, starts, final_strides, begin_mask);
    const Coordinates ends_abs      = strided_slice_absolute_end_coords(src_shape, starts_abs, ends, final_strides, end_mask, shrink_axis_mask);

    // Get destination shape
    const TensorShape dst_shape = compute_strided_slice_output_shape(src_shape, starts_abs, ends_abs, final_strides);

    // Create destination tensor
    SimpleTensor<T> dst{ dst_shape, src.data_type(), 1 };

    // Perform strided slice
    Window win;
    win.use_tensor_dimensions(dst_shape);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        Coordinates offset;
        for(unsigned int i = 0; i < id.num_dimensions(); ++i)
        {
            offset.set(i, starts_abs[i] + id[i] * final_strides[i]);
        }
        *reinterpret_cast<T *>(dst(id)) = *reinterpret_cast<const T *>(src(offset));
    });

    return dst;
}

template SimpleTensor<float> strided_slice(const SimpleTensor<float> &src,
                                           Coordinates starts, Coordinates ends, BiStrides strides,
                                           int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask);
template SimpleTensor<half_float::half> strided_slice(const SimpleTensor<half_float::half> &src,
                                                      Coordinates starts, Coordinates ends, BiStrides strides,
                                                      int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
