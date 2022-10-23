/*
 * Copyright (c) 2017-2018, 2021-2022 Arm Limited.
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

#ifndef ARM_COMPUTE_TEST_SIMPLE_TENSOR_PRINTER
#define ARM_COMPUTE_TEST_SIMPLE_TENSOR_PRINTER

#include "arm_compute/core/Error.h"

#include "tests/RawTensor.h"
#include "tests/SimpleTensor.h"

#include <iostream>
#include <sstream>

namespace arm_compute
{
namespace test
{
template <typename T>
inline std::string prettify_tensor(const SimpleTensor<T> &input, const IOFormatInfo &io_fmt = IOFormatInfo{ IOFormatInfo::PrintRegion::NoPadding })
{
    ARM_COMPUTE_ERROR_ON(input.data() == nullptr);

    RawTensor tensor(std::move(SimpleTensor<T>(input)));

    TensorInfo info(tensor.shape(), tensor.num_channels(), tensor.data_type());

    const DataType    dt           = info.data_type();
    const size_t      slices2D     = info.tensor_shape().total_size_upper(2);
    const Strides     strides      = info.strides_in_bytes();
    const PaddingSize padding      = info.padding();
    const size_t      num_channels = info.num_channels();

    std::ostringstream os;

    // Set precision
    if(is_data_type_float(dt) && (io_fmt.precision_type != IOFormatInfo::PrecisionType::Default))
    {
        int precision = io_fmt.precision;
        if(io_fmt.precision_type == IOFormatInfo::PrecisionType::Full)
        {
            precision = std::numeric_limits<float>().max_digits10;
        }
        os.precision(precision);
    }

    // Define region to print
    size_t print_width  = 0;
    size_t print_height = 0;
    int    start_offset = 0;
    switch(io_fmt.print_region)
    {
        case IOFormatInfo::PrintRegion::NoPadding:
            print_width  = info.dimension(0);
            print_height = info.dimension(1);
            start_offset = info.offset_first_element_in_bytes();
            break;
        case IOFormatInfo::PrintRegion::ValidRegion:
            print_width  = info.valid_region().shape.x();
            print_height = info.valid_region().shape.y();
            start_offset = info.offset_element_in_bytes(Coordinates(info.valid_region().anchor.x(),
                                                                    info.valid_region().anchor.y()));
            break;
        case IOFormatInfo::PrintRegion::Full:
            print_width  = padding.left + info.dimension(0) + padding.right;
            print_height = padding.top + info.dimension(1) + padding.bottom;
            start_offset = static_cast<int>(info.offset_first_element_in_bytes()) - padding.top * strides[1] - padding.left * strides[0];
            break;
        default:
            break;
    }

    print_width = print_width * num_channels;

    // Set pointer to start
    const uint8_t *ptr = tensor.data() + start_offset;

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
                max_element_width = std::max<int>(max_element_width, max_consecutive_elements_display_width(os, dt, ptr + offset, print_width));
                offset += strides[1];
            }
        }

        // Print slice
        {
            size_t offset = i * strides[2];
            for(size_t h = 0; h < print_height; ++h)
            {
                print_consecutive_elements(os, dt, ptr + offset, print_width, max_element_width, io_fmt.element_delim);
                offset += strides[1];
                os << io_fmt.row_delim;
            }
            os << io_fmt.row_delim;
        }
    }

    return os.str();
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const SimpleTensor<T> &tensor)
{
    os << prettify_tensor(tensor, IOFormatInfo{ IOFormatInfo::PrintRegion::NoPadding });
    return os;
}

template <typename T>
inline std::string to_string(const SimpleTensor<T> &tensor)
{
    std::stringstream ss;
    ss << tensor;
    return ss.str();
}

#if PRINT_TENSOR_LIMIT
template <typename T>
void print_simpletensor(const SimpleTensor<T> &tensor, const std::string &title, const IOFormatInfo::PrintRegion &region = IOFormatInfo::PrintRegion::NoPadding)
{
    if(tensor.num_elements() < PRINT_TENSOR_LIMIT)
    {
        std::cout << title << ":" << std::endl;
        std::cout << prettify_tensor(tensor, IOFormatInfo{ region });
    }
}
#endif // PRINT_TENSOR_LIMIT
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SIMPLE_TENSOR_PRINTER */
