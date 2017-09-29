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
#include "arm_compute/core/IAccessWindow.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

ValidRegion AccessWindowRectangle::compute_valid_region(const Window &window, const ValidRegion &input_valid_region) const
{
    return compute_valid_region(window, input_valid_region, false, BorderSize(0));
}

ValidRegion AccessWindowRectangle::compute_valid_region(const Window &window, ValidRegion input_valid_region, bool border_undefined, BorderSize border_size) const
{
    if(_info == nullptr)
    {
        return input_valid_region;
    }

    Coordinates &anchor = input_valid_region.anchor;
    Coordinates  old_anchor(anchor);
    TensorShape &shape = input_valid_region.shape;

    if(!border_undefined)
    {
        border_size = BorderSize(0);
    }

    // Start of the valid region is equal to the start of the window. But it
    // cannot be less than the start of the input's valid region plus the border
    // size required by this kernel (if undefined).
    // Additionally the valid region is shifted by the offset that is used by
    // the kernel to write back output values.
    anchor.set(0, std::max<int>(window.x().start() * _scale_x, anchor[0] + border_size.left) + _x);
    if(_info->num_dimensions() > 1)
    {
        anchor.set(1, std::max<int>(window.y().start() * _scale_y, anchor[1] + border_size.top) + _y);
    }

    // End of the valid region is equal to the start of the last write of the
    // kernel plus the number of written elements. (This assumes that all
    // written elements are valid). Nevertheless the end cannot be larger than
    // the end of the input's valid region minus the border size.
    // Note: not the end points of the region are stored but its size. Thus the
    // old size is first converted into end points to compared against the
    // execution window. Afterwards the new end points are converted back into
    // a size of the region.
    shape.set(0, std::min<int>(old_anchor[0] + shape[0] - border_size.right, (window.x().end() - window.x().step()) * _scale_x + _width) - anchor[0]);
    if(_info->num_dimensions() > 1)
    {
        shape.set(1, std::min<int>(old_anchor[1] + shape[1] - border_size.bottom, (window.y().end() - window.y().step()) * _scale_y + _height) - anchor[1]);
    }

    // For higher dimensions use the intersection of the window size and the
    // valid region of the input
    for(size_t d = 2; d < _info->num_dimensions(); ++d)
    {
        anchor.set(d, std::max(window[d].start(), input_valid_region.anchor[d]));
        shape.set(d, std::min<int>(window[d].end(), input_valid_region.shape[d]) - anchor[d]);
    }

    return input_valid_region;
}

void AccessWindowRectangle::set_valid_region(const Window &window, const ValidRegion &input_valid_region, bool border_undefined, const BorderSize &border_size)
{
    if(_info != nullptr)
    {
        _info->set_valid_region(compute_valid_region(window, input_valid_region, border_undefined, border_size));
    }
}

bool AccessWindowRectangle::update_window_if_needed(Window &window) const
{
    // Only update the window size if we can't use padding
    if(_info == nullptr || _info->is_resizable())
    {
        return false;
    }

    const TensorShape &shape                = _info->tensor_shape();
    const Strides     &strides              = _info->strides_in_bytes();
    const size_t       offset_first_element = _info->offset_first_element_in_bytes();

    bool window_modified = false;

    int front_pad_y = 0;

    const int min_y = window.y().start() * _scale_y + _y;
    const int max_y = (window.y().end() - window.y().step()) * _scale_y + _y + _height;

    // Adjust window start for Y dimension
    if(min_y < 0)
    {
        // Calculate rows available above the tensor
        const int front_pad_y_available = -static_cast<int>(offset_first_element / strides[1]);

        if(min_y < front_pad_y_available)
        {
            // Not enough padding available, need to shrink the window
            const int start = adjust_up(min_y, front_pad_y_available, window.y().step() * _scale_y) - _y;

            window.set(1, Window::Dimension(start / _scale_y, window.y().end(), window.y().step()));
            window_modified = true;
        }

        // Update front padding with reconstructed value
        front_pad_y = std::max(0, static_cast<int>(std::floor(-window.y().start() * _scale_y)) - _y);
    }

    // Adjust window end for Y dimension
    if(max_y > static_cast<int>(shape[1]))
    {
        const int stride_z = _info->num_dimensions() > 2 ? strides[2] : _info->total_size();

        // Calculate rows available below the tensor
        const int tail_pad_y_available = (stride_z / strides[1]) - shape[1] - front_pad_y;

        if(static_cast<int>(shape[1]) + tail_pad_y_available < max_y)
        {
            // Not enough padding available, need to shrink the window
            const int end = adjust_down(max_y, shape[1] + tail_pad_y_available, window.y().step() * _scale_y) + window.y().step() * _scale_y - _y - _height;
            window.set(1, Window::Dimension(window.y().start(), end / _scale_y, window.y().step()));
            window_modified = true;
        }
    }

    int front_pad_x = 0;

    const int min_x = window.x().start() * _scale_x + _x;
    const int max_x = (window.x().end() - window.x().step()) * _scale_x + _x + _width;

    const int stride_y = _info->num_dimensions() > 1 ? strides[1] : _info->total_size();

    // Adjust window start for X dimension
    if(min_x < 0)
    {
        const int front_pad_x_available = -std::min<int>(static_cast<int>(offset_first_element) - front_pad_y * strides[1], stride_y - shape[0] * strides[0]) / static_cast<int>(strides[0]);

        if(min_x < front_pad_x_available)
        {
            // Not enough padding available, need to shrink the window
            const int start = adjust_up(min_x, front_pad_x_available, window.x().step() * _scale_x) - _x;
            window.set(0, Window::Dimension(start / _scale_x, window.x().end(), window.x().step()));
            window_modified = true;
        }

        // Update front padding with reconstructed value
        front_pad_x = std::max(0, static_cast<int>(std::floor(-window.x().start() * _scale_x)) - _x);
    }

    // Adjust window end for X dimension
    if(max_x > static_cast<int>(shape[0]))
    {
        const int tail_pad_x_available = (stride_y / strides[0]) - shape[0] - front_pad_x;

        if(static_cast<int>(shape[0]) + tail_pad_x_available < max_x)
        {
            // Not enough padding available, need to shrink the window
            const int end = adjust_down(max_x, shape[0] + tail_pad_x_available, window.x().step() * _scale_x) + window.x().step() * _scale_x - _x - _width;
            window.set(0, Window::Dimension(window.x().start(), end / _scale_x, window.x().step()));
            window_modified = true;
        }
    }

    window.validate();

    return window_modified;
}

bool AccessWindowRectangle::update_padding_if_needed(const Window &window) const
{
    // Only update the padding if the tensor allows it
    if(_info == nullptr || !_info->is_resizable())
    {
        return false;
    }

    ARM_COMPUTE_ERROR_ON(window.x().step() * _scale_x == 0);
    ARM_COMPUTE_ERROR_ON(window.y().step() * _scale_y == 0);

    const int min_x = window.x().start() * _scale_x + _x;
    const int max_x = (window.x().end() - window.x().step()) * _scale_x + _x + _width;
    const int min_y = window.y().start() * _scale_y + _y;
    const int max_y = (window.y().end() - window.y().step()) * _scale_y + _y + _height;

    const TensorShape &shape = _info->tensor_shape();

    PaddingSize padding;
    padding.left   = std::max(0, -min_x);
    padding.right  = std::max<int>(0, max_x - shape[0]);
    padding.top    = std::max(0, -min_y);
    padding.bottom = std::max<int>(0, max_y - shape[1]);

    // Update strides in tensor info
    return _info->extend_padding(padding);
}
