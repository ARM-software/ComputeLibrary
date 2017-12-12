/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/ITensor.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Window.h"

#include <cstring>
#include <limits>

using namespace arm_compute;

void ITensor::copy_from(const ITensor &src)
{
    if(&src == this)
    {
        return;
    }

    const ITensorInfo *src_info = src.info();
    ITensorInfo       *dst_info = this->info();

    ARM_COMPUTE_ERROR_ON(src_info->num_dimensions() > dst_info->num_dimensions());
    ARM_COMPUTE_ERROR_ON(src_info->num_channels() != dst_info->num_channels());
    ARM_COMPUTE_ERROR_ON(src_info->element_size() != dst_info->element_size());

    for(size_t d = 0; d < src_info->num_dimensions(); d++)
    {
        ARM_COMPUTE_ERROR_ON(src_info->dimension(d) > dst_info->dimension(d));
    }

    // Copy information about valid region
    dst_info->set_valid_region(src_info->valid_region());

    Window win_src;
    win_src.use_tensor_dimensions(src_info->tensor_shape(), Window::DimY);
    Window win_dst;
    win_dst.use_tensor_dimensions(dst_info->tensor_shape(), Window::DimY);

    Iterator src_it(&src, win_src);
    Iterator dst_it(this, win_dst);

    const size_t line_size = src_info->num_channels() * src_info->element_size() * src_info->dimension(0);

    execute_window_loop(win_src, [&](const Coordinates & id)
    {
        memcpy(dst_it.ptr(), src_it.ptr(), line_size);
    },
    src_it, dst_it);
}

void ITensor::print(std::ostream &s, IOFormatInfo io_fmt) const
{
    ARM_COMPUTE_ERROR_ON(this->buffer() == nullptr);

    const DataType    dt           = this->info()->data_type();
    const size_t      slices2D     = this->info()->tensor_shape().total_size_upper(2);
    const Strides     strides      = this->info()->strides_in_bytes();
    const PaddingSize padding      = this->info()->padding();
    const size_t      num_channels = this->info()->num_channels();

    // Set precision
    if(is_data_type_float(dt) && (io_fmt.precision_type != IOFormatInfo::PrecisionType::Default))
    {
        int precision = io_fmt.precision;
        if(io_fmt.precision_type == IOFormatInfo::PrecisionType::Full)
        {
            precision = std::numeric_limits<float>().max_digits10;
        }
        s.precision(precision);
    }

    // Define region to print
    size_t print_width  = 0;
    size_t print_height = 0;
    int    start_offset = 0;
    switch(io_fmt.print_region)
    {
        case IOFormatInfo::PrintRegion::NoPadding:
            print_width  = this->info()->dimension(0);
            print_height = this->info()->dimension(1);
            start_offset = this->info()->offset_first_element_in_bytes();
            break;
        case IOFormatInfo::PrintRegion::ValidRegion:
            print_width  = this->info()->valid_region().shape.x();
            print_height = this->info()->valid_region().shape.y();
            start_offset = this->info()->offset_element_in_bytes(Coordinates(this->info()->valid_region().anchor.x(),
                                                                             this->info()->valid_region().anchor.y()));
            break;
        case IOFormatInfo::PrintRegion::Full:
            print_width  = padding.left + this->info()->dimension(0) + padding.right;
            print_height = padding.top + this->info()->dimension(1) + padding.bottom;
            start_offset = static_cast<int>(this->info()->offset_first_element_in_bytes()) - padding.top * strides[1] - padding.left * strides[0];
            break;
        default:
            break;
    }

    print_width = print_width * num_channels;

    // Set pointer to start
    const uint8_t *ptr = this->buffer() + start_offset;

    // Start printing
    for(size_t i = 0; i < slices2D; ++i)
    {
        // Find max_width of elements in slice to align columns
        int max_element_width = 0;
        if(io_fmt.align_columns)
        {
            size_t offset = i * strides[2];
            for(size_t h = 0; h < print_height; ++h)
            {
                max_element_width = std::max<int>(max_element_width, max_consecutive_elements_display_width(s, dt, ptr + offset, print_width));
                offset += strides[1];
            }
        }

        // Print slice
        {
            size_t offset = i * strides[2];
            for(size_t h = 0; h < print_height; ++h)
            {
                print_consecutive_elements(s, dt, ptr + offset, print_width, max_element_width, io_fmt.element_delim);
                offset += strides[1];
                s << io_fmt.row_delim;
            }
            s << io_fmt.row_delim;
        }
    }
}
