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

namespace arm_compute
{
namespace helpers
{
namespace tensor_transform
{
Coordinates slice_absolute_end_coords(TensorShape input_shape, Coordinates ends)
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
    // Get unit strides
    const BiStrides unit_strides = strided_slice_strides(input_shape, BiStrides());

    return strided_slice_absolute_end_coords(input_shape, Coordinates(), ends, unit_strides, end_mask);
}

TensorShape compute_slice_output_shape(TensorShape input_shape, Coordinates starts, Coordinates ends_abs)
{
    // Get unit strides
    const BiStrides unit_strides = strided_slice_strides(input_shape, BiStrides());
    return compute_strided_slice_output_shape(input_shape, starts, ends_abs, unit_strides);
}

Coordinates strided_slice_absolute_start_coords(TensorShape input_shape, Coordinates starts, Coordinates strides, int32_t begin_mask)
{
    Coordinates starts_abs;
    for(unsigned int i = 0; i < starts.num_dimensions(); ++i)
    {
        // Get start index
        int start_i = starts[i];

        // Reset in case of begin mask present
        if((begin_mask & 1 << i) != 0)
        {
            start_i = strides[i] > 0 ? std::numeric_limits<int>::lowest() : std::numeric_limits<int>::max();
        }

        // Account negative start points
        const int dim_size = input_shape[i];
        if(start_i < 0)
        {
            start_i += dim_size;
        }

        // Final clamp
        start_i = utility::clamp(start_i, 0, dim_size - 1);
        starts_abs.set(i, start_i);
    }

    // Fill remaining
    for(unsigned int i = starts_abs.num_dimensions(); i < input_shape.num_dimensions(); ++i)
    {
        starts_abs.set(i, 0);
    }

    return starts_abs;
}

Coordinates strided_slice_absolute_end_coords(TensorShape input_shape, Coordinates starts_abs, Coordinates ends, Coordinates strides,
                                              int32_t end_mask, int32_t shrink_axis_mask)
{
    Coordinates ends_abs;
    for(unsigned int i = 0; i < ends.num_dimensions(); ++i)
    {
        // Get end index
        int stop_i = ends[i];

        // Shrink dimension
        if((shrink_axis_mask & (1 << i)) != 0)
        {
            stop_i = starts_abs[i] + 1;
        }

        // Reset in case of begin mask present
        if((end_mask & 1 << i) != 0)
        {
            stop_i = (strides[i] > 0) ? std::numeric_limits<int>::max() : std::numeric_limits<int>::lowest();
        }

        // Account negative end points
        const int dim_size = input_shape[i];
        if(stop_i < 0)
        {
            stop_i += dim_size;
        }

        // Final clamp
        stop_i = (strides[i] > 0) ? utility::clamp(stop_i, 0, dim_size) : utility::clamp(stop_i, -1, dim_size - 1);
        ends_abs.set(i, stop_i);
    }

    // Fill remaining ends
    for(unsigned int i = ends_abs.num_dimensions(); i < input_shape.num_dimensions(); ++i)
    {
        ends_abs.set(i, input_shape[i]);
    }

    return ends_abs;
}

Coordinates strided_slice_strides(TensorShape input_shape, Coordinates strides)
{
    for(unsigned int i = strides.num_dimensions(); i < input_shape.num_dimensions(); ++i)
    {
        strides.set(i, 1);
    }
    return strides;
}

TensorShape compute_strided_slice_output_shape(TensorShape input_shape, Coordinates starts_abs, Coordinates ends_abs, Coordinates final_strides)
{
    TensorShape output_shape = input_shape;
    for(unsigned int i = 0; i < input_shape.num_dimensions(); ++i)
    {
        const int stride_i = final_strides[i];
        const int range    = ends_abs[i] - starts_abs[i];
        if((range == 0) ||                 // Zero range
           (range < 0 && stride_i >= 0) || // Negative range with positive stride
           (range > 0 && stride_i <= 0))   // Positive range with negative stride
        {
            output_shape.set(i, 0);
            return output_shape;
        }
        else
        {
            int dim = range / stride_i + (range % stride_i != 0 ? 1 : 0);
            output_shape.set(i, dim);
        }
    }
    return output_shape;
}
} // namespace tensor_transform
} // namespace helpers
} // namespace arm_compute
