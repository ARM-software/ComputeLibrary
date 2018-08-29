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
#include "LaplacianPyramid.h"

#include "tests/validation/reference/ArithmeticSubtraction.h"
#include "tests/validation/reference/DepthConvertLayer.h"
#include "tests/validation/reference/Gaussian5x5.h"
#include "tests/validation/reference/GaussianPyramidHalf.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename U>
std::vector<SimpleTensor<U>> laplacian_pyramid(const SimpleTensor<T> &src, SimpleTensor<U> &dst, size_t num_levels, BorderMode border_mode, uint8_t constant_border_value)
{
    std::vector<SimpleTensor<T>> pyramid_conv;
    std::vector<SimpleTensor<U>> pyramid_dst;

    // First, a Gaussian pyramid with SCALE_PYRAMID_HALF is created
    std::vector<SimpleTensor<T>> gaussian_level_pyramid = reference::gaussian_pyramid_half(src, border_mode, constant_border_value, num_levels);

    // For each level i, the corresponding image Ii is blurred with Gaussian 5x5
    // filter, and the difference between the two images is the corresponding
    // level Li of the Laplacian pyramid
    for(size_t i = 0; i < num_levels; ++i)
    {
        const SimpleTensor<T> level_filtered = reference::gaussian5x5(gaussian_level_pyramid[i], border_mode, constant_border_value);
        pyramid_conv.push_back(level_filtered);

        const SimpleTensor<U> level_sub = reference::arithmetic_subtraction<T, T, U>(gaussian_level_pyramid[i], level_filtered, dst.data_type(), ConvertPolicy::WRAP);
        pyramid_dst.push_back(level_sub);
    }

    // Return the lowest resolution image and the pyramid
    dst = depth_convert<T, U>(pyramid_conv[num_levels - 1], DataType::S16, ConvertPolicy::WRAP, 0);

    return pyramid_dst;
}

template std::vector<SimpleTensor<int16_t>> laplacian_pyramid(const SimpleTensor<uint8_t> &src, SimpleTensor<int16_t> &dst, size_t num_levels, BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
