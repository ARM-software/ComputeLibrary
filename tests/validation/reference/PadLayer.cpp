/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "PadLayer.h"

#include "arm_compute/core/Types.h"
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
SimpleTensor<T> pad_layer(const SimpleTensor<T> &src, const PaddingList &paddings, const PixelValue const_value, const PaddingMode mode)
{
    const DataType dst_data_type = src.data_type();

    const TensorShape orig_shape = src.shape();

    std::vector<PaddingInfo> paddings_extended = paddings;

    for(size_t i = paddings.size(); i < TensorShape::num_max_dimensions; ++i)
    {
        paddings_extended.emplace_back(PaddingInfo{ 0, 0 });
    }

    const TensorShape padded_shape = misc::shape_calculator::compute_padded_shape(orig_shape, paddings);

    SimpleTensor<T> dst(padded_shape, dst_data_type);

    // Reference algorithm: loop over the different dimension of the input.
    for(int idx = 0; idx < dst.num_elements(); ++idx)
    {
        const Coordinates coord = index2coord(padded_shape, idx);

        const size_t i = coord.x();
        const size_t j = coord.y();
        const size_t k = coord.z();
        const size_t l = coord[3];
        const size_t m = coord[4];
        const size_t n = coord[5];

        const std::array<size_t, TensorShape::num_max_dimensions> dims   = { { 0, 1, 2, 3, 4, 5 } };
        const std::array<size_t, TensorShape::num_max_dimensions> coords = { { i, j, k, l, m, n } };
        auto is_padding_area = [&](size_t i)
        {
            return (coords[i] < paddings_extended[i].first || coords[i] > orig_shape[i] + paddings_extended[i].first - 1);
        };

        auto orig_coord_reflect = [&](size_t i)
        {
            if(is_padding_area(i))
            {
                if(coords[i] < paddings_extended[i].first)
                {
                    return paddings_extended[i].first - coords[i];
                }
                else
                {
                    return 2 * orig_shape[i] + paddings_extended[i].first - 2 - coords[i];
                }
            }
            return coords[i] - paddings_extended[i].first;
        };

        auto orig_coord_symm = [&](size_t i)
        {
            if(is_padding_area(i))
            {
                if(coords[i] < paddings_extended[i].first)
                {
                    return paddings_extended[i].first - coords[i] - 1;
                }
                else
                {
                    return 2 * orig_shape[i] + paddings_extended[i].first - 1 - coords[i];
                }
            }
            return coords[i] - paddings_extended[i].first;
        };

        // If the tuple [i,j,k,l,m] is in the padding area, then simply set the value
        if(std::any_of(dims.begin(), dims.end(), is_padding_area))
        {
            switch(mode)
            {
                case PaddingMode::CONSTANT:
                    const_value.get(dst[idx]);
                    break;
                case PaddingMode::REFLECT:
                {
                    const Coordinates orig_coords{ orig_coord_reflect(0),
                                             orig_coord_reflect(1),
                                             orig_coord_reflect(2),
                                             orig_coord_reflect(3),
                                             orig_coord_reflect(4),
                                             orig_coord_reflect(5) };

                    const size_t idx_src = coord2index(orig_shape, orig_coords);
                    dst[idx]             = src[idx_src];
                    break;
                }
                case PaddingMode::SYMMETRIC:
                {
                    const Coordinates orig_coords{ orig_coord_symm(0),
                                             orig_coord_symm(1),
                                             orig_coord_symm(2),
                                             orig_coord_symm(3),
                                             orig_coord_symm(4),
                                             orig_coord_symm(5) };

                    const size_t idx_src = coord2index(orig_shape, orig_coords);
                    dst[idx]             = src[idx_src];
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("Padding mode not supported.");
                    break;
            }
        }
        else
        {
            // If the tuple[i,j,k,l,m] is not in the padding area, then copy the input into the output

            const Coordinates orig_coords{ i - paddings_extended[0].first,
                                     j - paddings_extended[1].first,
                                     k - paddings_extended[2].first,
                                     l - paddings_extended[3].first,
                                     m - paddings_extended[4].first,
                                     n - paddings_extended[5].first };

            const size_t idx_src = coord2index(orig_shape, orig_coords);
            dst[idx]             = src[idx_src];
        }
    }

    return dst;
}

template SimpleTensor<float> pad_layer(const SimpleTensor<float> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<half> pad_layer(const SimpleTensor<half> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<uint32_t> pad_layer(const SimpleTensor<uint32_t> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<uint8_t> pad_layer(const SimpleTensor<uint8_t> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<int8_t> pad_layer(const SimpleTensor<int8_t> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<uint16_t> pad_layer(const SimpleTensor<uint16_t> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
template SimpleTensor<int16_t> pad_layer(const SimpleTensor<int16_t> &src, const PaddingList &paddings, const PixelValue const_value = PixelValue(), const PaddingMode mode);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
