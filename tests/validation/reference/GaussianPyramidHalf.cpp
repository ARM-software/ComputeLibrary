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
#include "GaussianPyramidHalf.h"

#include "arm_compute/core/Helpers.h"

#include "Gaussian5x5.h"
#include "Scale.h"
#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
std::vector<SimpleTensor<T>> gaussian_pyramid_half(const SimpleTensor<T> &src, BorderMode border_mode, uint8_t constant_border_value, size_t num_levels)
{
    std::vector<SimpleTensor<T>> dst;

    // Level0 is equal to src
    dst.push_back(src);

    for(size_t i = 1; i < num_levels; ++i)
    {
        // Gaussian Filter
        const SimpleTensor<T> out_gaus5x5 = reference::gaussian5x5(dst[i - 1], border_mode, constant_border_value);

        // Scale down by 2 with nearest interpolation
        const SimpleTensor<T> out = reference::scale(out_gaus5x5, SCALE_PYRAMID_HALF, SCALE_PYRAMID_HALF, InterpolationPolicy::NEAREST_NEIGHBOR, border_mode, constant_border_value, SamplingPolicy::CENTER,
                                                     true);

        dst.push_back(out);
    }

    return dst;
}

template std::vector<SimpleTensor<uint8_t>> gaussian_pyramid_half(const SimpleTensor<uint8_t> &src, BorderMode border_mode, uint8_t constant_border_value, size_t num_levels);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
