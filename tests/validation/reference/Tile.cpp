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
#include "Tile.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> tile(const SimpleTensor<T> &src, const Multiples &multiples)
{
    // Create reference
    const auto src_shape   = src.shape();
    const auto tiled_shape = misc::shape_calculator::compute_tiled_shape(src.shape(), multiples);

    SimpleTensor<T> dst{ tiled_shape, src.data_type() };

    for(int idx = 0; idx < dst.num_elements(); idx++)
    {
        Coordinates coord = index2coord(tiled_shape, idx);

        const size_t x = coord.x();
        const size_t y = coord.y();
        const size_t z = coord.z();
        const size_t w = coord[3];

        Coordinates src_coords{ x % src_shape[0], y % src_shape[1], z % src_shape[2], w % src_shape[3] };
        int         src_idx = coord2index(src_shape, src_coords);

        dst[idx] = src[src_idx];
    }

    return dst;
}

template SimpleTensor<uint8_t> tile(const SimpleTensor<uint8_t> &src, const Multiples &multiples);
template SimpleTensor<int8_t> tile(const SimpleTensor<int8_t> &src, const Multiples &multiples);
template SimpleTensor<uint16_t> tile(const SimpleTensor<uint16_t> &src, const Multiples &multiples);
template SimpleTensor<int16_t> tile(const SimpleTensor<int16_t> &src, const Multiples &multiples);
template SimpleTensor<uint32_t> tile(const SimpleTensor<uint32_t> &src, const Multiples &multiples);
template SimpleTensor<int32_t> tile(const SimpleTensor<int32_t> &src, const Multiples &multiples);
template SimpleTensor<half> tile(const SimpleTensor<half> &src, const Multiples &multiples);
template SimpleTensor<float> tile(const SimpleTensor<float> &src, const Multiples &multiples);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
