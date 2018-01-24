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
#include "Derivative.h"

#include "Utils.h"
#include "tests/Types.h"

#include <array>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
namespace
{
const std::array<int8_t, 9> derivative_3_x{ { 0, 0, 0, -1, 0, 1, 0, 0, 0 } };
const std::array<int8_t, 9> derivative_3_y{ { 0, -1, 0, 0, 0, 0, 0, 1, 0 } };

template <typename T>
struct data_type;

template <>
struct data_type<int16_t>
{
    const static DataType value = DataType::S16;
};
} // namespace

template <typename T, typename U>
std::pair<SimpleTensor<T>, SimpleTensor<T>> derivative(const SimpleTensor<U> &src, BorderMode border_mode, uint8_t constant_border_value, GradientDimension gradient_dimension)
{
    const unsigned int filter_size = 3;

    SimpleTensor<T> dst_x(src.shape(), data_type<T>::value, src.num_channels());
    SimpleTensor<T> dst_y(src.shape(), data_type<T>::value, src.num_channels());

    ValidRegion valid_region = shape_to_valid_region(src.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(filter_size / 2));

    for(int i = 0; i < src.num_elements(); ++i)
    {
        Coordinates coord = index2coord(src.shape(), i);

        if(!is_in_valid_region(valid_region, coord))
        {
            continue;
        }

        switch(gradient_dimension)
        {
            case GradientDimension::GRAD_X:
                apply_2d_spatial_filter(coord, src, dst_x, TensorShape{ filter_size, filter_size }, derivative_3_x.data(), 1.f, border_mode,
                                        constant_border_value);
                break;
            case GradientDimension::GRAD_Y:
                apply_2d_spatial_filter(coord, src, dst_y, TensorShape{ filter_size, filter_size }, derivative_3_y.data(), 1.f, border_mode,
                                        constant_border_value);
                break;
            case GradientDimension::GRAD_XY:
                apply_2d_spatial_filter(coord, src, dst_x, TensorShape{ filter_size, filter_size }, derivative_3_x.data(), 1.f, border_mode,
                                        constant_border_value);
                apply_2d_spatial_filter(coord, src, dst_y, TensorShape{ filter_size, filter_size }, derivative_3_y.data(), 1.f, border_mode,
                                        constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Gradient dimension not supported");
        }
    }

    return std::make_pair(dst_x, dst_y);
}

template std::pair<SimpleTensor<int16_t>, SimpleTensor<int16_t>> derivative(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value,
                                                                            GradientDimension gradient_dimension);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
