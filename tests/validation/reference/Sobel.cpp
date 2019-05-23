/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "Sobel.h"

#include "Utils.h"
#include "tests/validation/Helpers.h"

#include <array>
#include <map>
#include <utility>

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
const std::array<int8_t, 9> sobel_3_x{ { -1, 0, 1, -2, 0, 2, -1, 0, 1 } };
const std::array<int8_t, 9> sobel_3_y{ { -1, -2, -1, 0, 0, 0, 1, 2, 1 } };

const std::array<int8_t, 25> sobel_5_x{ {
        -1, -2, 0, 2, 1,
        -4, -8, 0, 8, 4,
        -6, -12, 0, 12, 6,
        -4, -8, 0, 8, 4,
        -1, -2, 0, 2, 1
    } };

const std::array<int8_t, 25> sobel_5_y{ {
        -1, -4, -6, -4, -1,
        -2, -8, -12, -8, -2,
        0, 0, 0, 0, 0,
        2, 8, 12, 8, 2,
        1, 4, 6, 4, 1
    } };

const std::array<int8_t, 49> sobel_7_x{ {
        -1, -4, -5, 0, 5, 4, 1,
        -6, -24, -30, 0, 30, 24, 6,
        -15, -60, -75, 0, 75, 60, 15,
        -20, -80, -100, 0, 100, 80, 20,
        -15, -60, -75, 0, 75, 60, 15,
        -6, -24, -30, 0, 30, 24, 6,
        -1, -4, -5, 0, 5, 4, 1
    } };

const std::array<int8_t, 49> sobel_7_y{ {
        -1, -6, -15, -20, -15, -6, -1,
        -4, -24, -60, -80, -60, -24, -4,
        -5, -30, -75, -100, -75, -30, -5,
        0, 0, 0, 0, 0, 0, 0,
        5, 30, 75, 100, 75, 30, 5,
        4, 24, 60, 80, 60, 24, 4,
        1, 6, 15, 20, 15, 6, 1
    } };

const std::map<int, std::pair<const int8_t *, const int8_t *>> masks
{
    { 3, { sobel_3_x.data(), sobel_3_y.data() } },
    { 5, { sobel_5_x.data(), sobel_5_y.data() } },
    { 7, { sobel_7_x.data(), sobel_7_y.data() } },
};

template <typename T>
struct data_type;

template <>
struct data_type<int16_t>
{
    const static DataType value = DataType::S16;
};

template <>
struct data_type<int>
{
    const static DataType value = DataType::S32;
};
} // namespace

template <typename T, typename U>
std::pair<SimpleTensor<T>, SimpleTensor<T>> sobel(const SimpleTensor<U> &src, int filter_size, BorderMode border_mode, uint8_t constant_border_value, GradientDimension gradient_dimension)
{
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
                apply_2d_spatial_filter(coord, src, dst_x, TensorShape{ static_cast<unsigned int>(filter_size), static_cast<unsigned int>(filter_size) }, masks.at(filter_size).first, 1.f, border_mode,
                                        constant_border_value);
                break;
            case GradientDimension::GRAD_Y:
                apply_2d_spatial_filter(coord, src, dst_y, TensorShape{ static_cast<unsigned int>(filter_size), static_cast<unsigned int>(filter_size) }, masks.at(filter_size).second, 1.f, border_mode,
                                        constant_border_value);
                break;
            case GradientDimension::GRAD_XY:
                apply_2d_spatial_filter(coord, src, dst_x, TensorShape{ static_cast<unsigned int>(filter_size), static_cast<unsigned int>(filter_size) }, masks.at(filter_size).first, 1.f, border_mode,
                                        constant_border_value);
                apply_2d_spatial_filter(coord, src, dst_y, TensorShape{ static_cast<unsigned int>(filter_size), static_cast<unsigned int>(filter_size) }, masks.at(filter_size).second, 1.f, border_mode,
                                        constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Gradient dimension not supported");
        }
    }

    return std::make_pair(dst_x, dst_y);
}

template std::pair<SimpleTensor<int16_t>, SimpleTensor<int16_t>> sobel(const SimpleTensor<uint8_t> &src, int filter_size, BorderMode border_mode, uint8_t constant_border_value,
                                                                       GradientDimension gradient_dimension);
template std::pair<SimpleTensor<int>, SimpleTensor<int>> sobel(const SimpleTensor<uint8_t> &src, int filter_size, BorderMode border_mode, uint8_t constant_border_value,
                                                               GradientDimension gradient_dimension);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
