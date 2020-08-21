/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
Window calculate_max_window(const ValidRegion &valid_region, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if(!skip_border)
    {
        border_size = BorderSize(0);
    }

    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                   // Skip the border left of the image
                   anchor[0] + border_size.left,
                   // Skip the border right of the image
                   // Make sure the window width is a multiple of the step size
                   anchor[0] + border_size.left + ceil_to_multiple(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) - static_cast<int>(border_size.right)), steps[0]),
                   steps[0]));

    size_t n = 1;

    if(anchor.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                       // Skip the border above the image
                       anchor[1] + border_size.top,
                       // Skip the border below the image
                       anchor[1] + border_size.top + ceil_to_multiple(std::max(0, static_cast<int>(shape[1]) - static_cast<int>(border_size.top) - static_cast<int>(border_size.bottom)), steps[1]),
                       steps[1]));

        ++n;
    }

    if(anchor.num_dimensions() > 2)
    {
        window.set(2, Window::Dimension(anchor[2], std::max<size_t>(1, shape[2]), steps[2]));

        ++n;
    }

    for(; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for(; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

Window calculate_max_enlarged_window(const ValidRegion &valid_region, const Steps &steps, BorderSize border_size)
{
    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                   // move the anchor to the start from the border
                   anchor[0] - border_size.left,
                   // move the anchor to include the right end border
                   // Make sure the window width is a multiple of the step size
                   anchor[0] - border_size.left + ceil_to_multiple(shape[0] + border_size.left + border_size.right, steps[0]),
                   steps[0]));

    size_t n = 1;

    if(anchor.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                       // Include the border above the image
                       anchor[1] - border_size.top,
                       // Include the border below the image
                       anchor[1] - border_size.top + ceil_to_multiple(shape[1] + border_size.top + border_size.bottom, steps[1]),
                       steps[1]));

        ++n;
    }

    if(anchor.num_dimensions() > 2)
    {
        window.set(2, Window::Dimension(0, std::max<size_t>(1, shape[n]), steps[2]));

        ++n;
    }

    for(; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for(; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

Window calculate_max_window_horizontal(const ValidRegion &valid_region, const Steps &steps, bool skip_border, BorderSize border_size)
{
    if(skip_border)
    {
        border_size.top    = 0;
        border_size.bottom = 0;
    }
    else
    {
        border_size.left  = 0;
        border_size.right = 0;
    }

    const Coordinates &anchor = valid_region.anchor;
    const TensorShape &shape  = valid_region.shape;

    Window window;

    window.set(0, Window::Dimension(
                   // Skip the border left of the image
                   anchor[0] + border_size.left,
                   // Skip the border right of the image
                   // Make sure the window width is a multiple of the step size
                   anchor[0] + border_size.left + ceil_to_multiple(std::max(0, static_cast<int>(shape[0]) - static_cast<int>(border_size.left) - static_cast<int>(border_size.right)), steps[0]),
                   steps[0]));

    size_t n = 1;

    if(anchor.num_dimensions() > 1)
    {
        window.set(1, Window::Dimension(
                       // Skip the border above the image
                       anchor[1] - border_size.top,
                       // Skip the border below the image
                       anchor[1] + shape[1] + border_size.bottom,
                       1));

        ++n;
    }

    for(; n < anchor.num_dimensions(); ++n)
    {
        window.set(n, Window::Dimension(anchor[n], std::max<size_t>(1, shape[n])));
    }

    for(; n < Coordinates::num_max_dimensions; ++n)
    {
        window.set(n, Window::Dimension(0, 1));
    }

    return window;
}

ValidRegion calculate_valid_region_scale(const ITensorInfo &src_info, const TensorShape &dst_shape,
                                         InterpolationPolicy interpolate_policy, SamplingPolicy sampling_policy, bool border_undefined)
{
    const DataLayout data_layout = src_info.data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    const float scale_x        = static_cast<float>(dst_shape[idx_width]) / src_info.tensor_shape()[idx_width];
    const float scale_y        = static_cast<float>(dst_shape[idx_height]) / src_info.tensor_shape()[idx_height];
    const float sampling_point = (sampling_policy == SamplingPolicy::CENTER) ? 0.5f : 0.0f;

    // Get input's valid region start and end points
    const int valid_start_in_x = src_info.valid_region().anchor[idx_width];
    const int valid_start_in_y = src_info.valid_region().anchor[idx_height];
    const int valid_end_in_x   = src_info.valid_region().anchor[idx_width] + src_info.valid_region().shape[idx_width];
    const int valid_end_in_y   = src_info.valid_region().anchor[idx_height] + src_info.valid_region().shape[idx_height];

    // Initialize output's valid region start and end points
    auto valid_start_out_x = static_cast<int>(valid_start_in_x * scale_x);
    auto valid_start_out_y = static_cast<int>(valid_start_in_y * scale_y);
    auto valid_end_out_x   = std::min<int>(std::ceil(valid_end_in_x * scale_x), dst_shape[idx_width]);
    auto valid_end_out_y   = std::min<int>(std::ceil(valid_end_in_y * scale_y), dst_shape[idx_height]);

    // Handle valid points in case of the bi-linear interpolation
    if(border_undefined)
    {
        switch(interpolate_policy)
        {
            case InterpolationPolicy::NEAREST_NEIGHBOR:
            {
                // (start_out + sampling_point) >= (start_in * scale)
                // start_out = ceil((start_in * scale) - sampling_point)
                valid_start_out_x = std::ceil(valid_start_in_x * scale_x - sampling_point);
                valid_start_out_y = std::ceil(valid_start_in_y * scale_y - sampling_point);

                // (end_out - 1 + sampling_point) < (end_in * scale)
                // end_out   = ceil((end_in * scale) - sampling_point); // <-- ceil(x - 1) strictly less
                valid_end_out_x = std::ceil(valid_end_in_x * scale_x - sampling_point);
                valid_end_out_y = std::ceil(valid_end_in_y * scale_y - sampling_point);
                break;
            }
            case InterpolationPolicy::BILINEAR:
            {
                // (start_out + sampling_point) >= ((start_in + sampling_point) * scale)
                // start_out = ceil(((start_in + sampling_point) * scale) - sampling_point)
                valid_start_out_x = std::ceil((valid_start_in_x + sampling_point) * scale_x - sampling_point);
                valid_start_out_y = std::ceil((valid_start_in_y + sampling_point) * scale_y - sampling_point);

                // (end_out - 1 + sampling_point) <= ((end_in - 1 + sampling_point) * scale)
                // end_out   = floor(((end_in - 1 + sampling_point) * scale) - sampling_point + 1)
                valid_end_out_x = std::floor((valid_end_in_x - 1.f + sampling_point) * scale_x - sampling_point + 1.f);
                valid_end_out_y = std::floor((valid_end_in_y - 1.f + sampling_point) * scale_y - sampling_point + 1.f);
                break;
            }
            case InterpolationPolicy::AREA:
                break;
            default:
            {
                ARM_COMPUTE_ERROR("Invalid InterpolationPolicy");
                break;
            }
        }
    }

    // Setup output valid region
    ValidRegion valid_region{ Coordinates(), dst_shape, src_info.tensor_shape().num_dimensions() };

    valid_region.anchor.set(idx_width, std::max(0, valid_start_out_x));
    valid_region.anchor.set(idx_height, std::max(0, valid_start_out_y));

    valid_region.shape.set(idx_width, std::min<size_t>(valid_end_out_x - valid_start_out_x, dst_shape[idx_width]));
    valid_region.shape.set(idx_height, std::min<size_t>(valid_end_out_y - valid_start_out_y, dst_shape[idx_height]));

    return valid_region;
}

PermutationVector get_permutation_vector_from_softmax_axis(size_t actual_axis)
{
    switch(actual_axis)
    {
        case 1:
            return PermutationVector(1U, 0U, 2U, 3U);
        case 2:
            return PermutationVector(2U, 1U, 0U, 3U);
        case 3:
            return PermutationVector(3U, 1U, 2U, 0U);
        default:
            ARM_COMPUTE_ERROR("Axis not supported");
    }
}
} // namespace arm_compute