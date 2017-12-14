/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/AccessWindowStatic.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

AccessWindowStatic::AccessWindowStatic(ITensorInfo *info, int start_x, int start_y, int end_x, int end_y)
    : _info(info), _start_x(start_x), _start_y(start_y), _end_x(end_x), _end_y(end_y)
{
}

ValidRegion AccessWindowStatic::compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const
{
    ARM_COMPUTE_UNUSED(border_undefined);
    ARM_COMPUTE_UNUSED(border_size);

    return compute_valid_region(window, input_valid_region);
}

ValidRegion AccessWindowStatic::compute_valid_region(const Window &window, ValidRegion input_valid_region) const
{
    if(_info == nullptr)
    {
        return input_valid_region;
    }

    Coordinates &anchor = input_valid_region.anchor;
    TensorShape &shape  = input_valid_region.shape;

    // Start of the valid region is equal to the start of the static access but
    // never outside of the tensor.
    anchor.set(0, std::max<int>(0, _start_x));
    if(_info->num_dimensions() > 1)
    {
        anchor.set(1, std::max<int>(0, _start_y));
    }

    // End of the valid region is equal to the end of the static access but
    // never outside of the tensor.
    shape.set(0, std::min<int>(_end_x, _info->tensor_shape()[0]));
    if(_info->num_dimensions() > 1)
    {
        shape.set(1, std::min<int>(_end_y, _info->tensor_shape()[1]));
    }

    // For higher dimension use the intersection of the window size and the
    // valid region of the input
    for(size_t d = 2; d < _info->num_dimensions(); ++d)
    {
        anchor.set(d, std::max(window[d].start(), input_valid_region.anchor[d]));
        shape.set(d, std::min<int>(window[d].end(), input_valid_region.shape[d]) - anchor[d]);
    }

    return input_valid_region;
}

void AccessWindowStatic::set_valid_region(const Window &window, const ValidRegion &input_valid_region)
{
    if(_info != nullptr)
    {
        _info->set_valid_region(compute_valid_region(window, input_valid_region));
    }
}

bool AccessWindowStatic::update_window_if_needed(Window &window) const
{
    // If the padding is not enough and the tensor is not resizable, shrink the window to size 0
    if(_info == nullptr || _info->is_resizable())
    {
        return false;
    }

    const TensorShape &shape                = _info->tensor_shape();
    const Strides     &strides              = _info->strides_in_bytes();
    const size_t       offset_first_element = _info->offset_first_element_in_bytes();

    bool window_modified = false;

    // Calculate if padding is enough
    if(_start_y < 0)
    {
        const int front_pad_y_available = -static_cast<int>(offset_first_element / strides[1]);

        if(_start_y < front_pad_y_available)
        {
            window_modified = true;
        }
    }

    if(!window_modified)
    {
        if(_end_y > static_cast<int>(shape[1]))
        {
            const int stride_z             = _info->num_dimensions() > 2 ? strides[2] : _info->total_size();
            const int tail_pad_y_available = (stride_z / strides[1]) - shape[1];

            if(static_cast<int>(shape[1]) + tail_pad_y_available < _end_y)
            {
                window_modified = true;
            }
        }

        if(!window_modified)
        {
            const int stride_y = _info->num_dimensions() > 1 ? strides[1] : _info->total_size();

            if(_start_x < 0)
            {
                const int front_pad_x_available = -std::min<int>(static_cast<int>(offset_first_element), stride_y - shape[0] * strides[0]) / static_cast<int>(strides[0]);

                if(_start_x < front_pad_x_available)
                {
                    window_modified = true;
                }
            }

            if(!window_modified && _end_x > static_cast<int>(shape[0]))
            {
                const int tail_pad_x_available = (stride_y / strides[0]) - shape[0];

                if(static_cast<int>(shape[0]) + tail_pad_x_available < _end_x)
                {
                    window_modified = true;
                }
            }
        }
    }

    // If padding is not enough
    if(window_modified)
    {
        for(size_t i = 0; i < Coordinates::num_max_dimensions; ++i)
        {
            window.set(i, Window::Dimension(0, 0, 1));
        }
    }

    return window_modified;
}

bool AccessWindowStatic::update_padding_if_needed(const Window &window) const
{
    ARM_COMPUTE_UNUSED(window);

    // Only update the padding if the tensor allows it
    if(_info == nullptr || !_info->is_resizable())
    {
        return false;
    }

    const TensorShape &shape = _info->tensor_shape();

    PaddingSize padding;
    padding.left   = std::max(0, -_start_x);
    padding.right  = std::max<int>(0, _end_x - shape[0]);
    padding.top    = std::max(0, -_start_y);
    padding.bottom = std::max<int>(0, _end_y - shape[1]);

    // Update strides in tensor info
    return _info->extend_padding(padding);
}
