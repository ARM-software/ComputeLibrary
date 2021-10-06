/*
 * Copyright (c) 2016-2021 Arm Limited.
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
    ValidRegion valid_region{ Coordinates(), dst_shape, dst_shape.num_dimensions() };

    valid_region.anchor.set(idx_width, std::max(0, valid_start_out_x));
    valid_region.anchor.set(idx_height, std::max(0, valid_start_out_y));

    valid_region.shape.set(idx_width, std::min<size_t>(valid_end_out_x - valid_start_out_x, dst_shape[idx_width]));
    valid_region.shape.set(idx_height, std::min<size_t>(valid_end_out_y - valid_start_out_y, dst_shape[idx_height]));

    return valid_region;
}

const std::map<DataLayout, std::vector<DataLayoutDimension>> &get_layout_map()
{
    constexpr DataLayoutDimension W = DataLayoutDimension::WIDTH;
    constexpr DataLayoutDimension H = DataLayoutDimension::HEIGHT;
    constexpr DataLayoutDimension C = DataLayoutDimension::CHANNEL;
    constexpr DataLayoutDimension D = DataLayoutDimension::DEPTH;
    constexpr DataLayoutDimension N = DataLayoutDimension::BATCHES;

    static const std::map<DataLayout, std::vector<DataLayoutDimension>> layout_map =
    {
        { DataLayout::NDHWC, { C, W, H, D, N } },
        { DataLayout::NCDHW, { W, H, D, C, N } },
        { DataLayout::NHWC, { C, W, H, N } },
        { DataLayout::NCHW, { W, H, C, N } }
    };

    return layout_map;
}
} // namespace arm_compute