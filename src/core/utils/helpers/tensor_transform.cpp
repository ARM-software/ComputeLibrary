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
#include "arm_compute/core/utils/helpers/tensor_transform.h"

#include "arm_compute/core/utils/helpers/bit_ops.h"

namespace arm_compute
{
namespace helpers
{
namespace tensor_transform
{
int calculate_stride_on_index(int index, Coordinates strides)
{
    return index >= static_cast<int>(strides.num_dimensions()) ? 1 : strides[index];
}

int calculate_start_on_index(TensorShape input_shape, int index, Coordinates starts, Coordinates strides, int32_t begin_mask)
{
    // Early exit
    if(index >= static_cast<int>(starts.num_dimensions()))
    {
        return 0;
    }

    // Get stride
    const int stride = calculate_stride_on_index(index, strides);

    // Calculate start
    int start = starts[index];

    // Reset in case of begin mask present
    if(arm_compute::helpers::bit_ops::is_bit_set(begin_mask, index))
    {
        start = stride > 0 ? std::numeric_limits<int>::lowest() : std::numeric_limits<int>::max();
    }

    // Account negative start points
    const int dim_size = input_shape[index];
    if(start < 0)
    {
        start += dim_size;
    }

    // Final clamp
    start = utility::clamp(start, 0, dim_size - 1);

    return start;
}

int calculate_end_on_index(TensorShape input_shape, int index, int start_on_index,
                           Coordinates ends, Coordinates strides,
                           int32_t end_mask, int32_t shrink_axis_mask)
{
    // Early exit
    if(index >= static_cast<int>(ends.num_dimensions()))
    {
        return input_shape[index];
    }

    const int  stride      = calculate_stride_on_index(index, strides);
    const bool shrink_axis = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, index);

    // Calculate start
    int stop = ends[index];

    // Shrink dimension
    if(shrink_axis)
    {
        stop = start_on_index + 1;
    }

    // Reset in case of begin mask present
    if(arm_compute::helpers::bit_ops::is_bit_set(end_mask, index) && !shrink_axis)
    {
        stop = (stride > 0) ? std::numeric_limits<int>::max() : std::numeric_limits<int>::lowest();
    }

    // Account negative end points
    const int dim_size = input_shape[index];
    if(stop < 0)
    {
        stop += dim_size;
    }

    // Final clamp
    stop = (stride > 0) ? utility::clamp(stop, 0, dim_size) : utility::clamp(stop, -1, dim_size - 1);

    return stop;
}

std::tuple<Coordinates, Coordinates, Coordinates> calculate_strided_slice_coords(TensorShape input_shape,
                                                                                 Coordinates starts, Coordinates ends, Coordinates strides,
                                                                                 int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    Coordinates starts_abs, ends_abs, final_strides;
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        const int start_i = calculate_start_on_index(input_shape, i, starts, strides, begin_mask);
        starts_abs.set(i, start_i);
        ends_abs.set(i, calculate_end_on_index(input_shape, i, start_i, ends, strides, end_mask, shrink_axis_mask));
        final_strides.set(i, calculate_stride_on_index(i, strides));
    }

    return std::make_tuple(starts_abs, ends_abs, final_strides);
}

TensorShape compute_strided_slice_output_shape(TensorShape input_shape, Coordinates starts, Coordinates ends, Coordinates strides,
                                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask, bool return_unshrinked)
{
    unsigned int index = 0;

    TensorShape output_shape;
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        const int stride = calculate_stride_on_index(index, strides);
        const int start  = calculate_start_on_index(input_shape, i, starts, strides, begin_mask);
        const int end    = calculate_end_on_index(input_shape, i, start, ends, strides, end_mask, shrink_axis_mask);
        const int range  = end - start;

        const bool is_shrink = arm_compute::helpers::bit_ops::is_bit_set(shrink_axis_mask, i);
        if(return_unshrinked || !is_shrink)
        {
            if((range == 0) ||               // Zero range
               (range < 0 && stride >= 0) || // Negative range with positive stride
               (range > 0 && stride <= 0))   // Positive range with negative stride
            {
                output_shape.set(index, 0);
                return output_shape;
            }
            else
            {
                int dim = range / stride + (range % stride != 0 ? 1 : 0);
                output_shape.set(index++, dim);
            }
        }
    }
    return output_shape;
}

int32_t construct_slice_end_mask(Coordinates ends)
{
    // Create end mask
    int32_t end_mask = 0;
    for(unsigned int i = 0; i < ends.num_dimensions(); ++i)
    {
        if(ends[i] < 0)
        {
            end_mask |= 1 << i;
        }
    }

    return end_mask;
}
} // namespace tensor_transform
} // namespace helpers
} // namespace arm_compute
