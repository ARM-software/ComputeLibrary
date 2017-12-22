/*
 * Copyright (c) 2017 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal src the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included src all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. src NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER src AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * dst OF OR src CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "NonLinearFilter.h"
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
SimpleTensor<T> non_linear_filter(const SimpleTensor<T> &src, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode,
                                  uint8_t constant_border_value)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    ARM_COMPUTE_ERROR_ON(pattern == MatrixPattern::OTHER && mask == nullptr);
    ARM_COMPUTE_UNUSED(pattern);

    using intermediate_type = typename common_promoted_signed_type<T>::intermediate_type;

    const int                      sq_mask_size   = mask_size * mask_size;
    const int                      half_mask_size = mask_size / 2;
    std::vector<intermediate_type> vals(sq_mask_size);
    intermediate_type              current_value = 0;

    const ValidRegion valid_region = shape_to_valid_region(src.shape(), border_mode == BorderMode::UNDEFINED, BorderSize(half_mask_size));

    for(int element_idx = 0, count = 0, index = 0; element_idx < src.num_elements(); ++element_idx, count = 0, index = 0)
    {
        Coordinates id = index2coord(src.shape(), element_idx);
        if(is_in_valid_region(valid_region, id))
        {
            int idx = id.x();
            int idy = id.y();
            for(int y = idy - half_mask_size; y <= idy + half_mask_size; ++y)
            {
                for(int x = idx - half_mask_size; x <= idx + half_mask_size; ++x, ++index)
                {
                    id.set(0, x);
                    id.set(1, y);
                    current_value = tensor_elem_at(src, id, border_mode, constant_border_value);

                    if(mask[index] == 255)
                    {
                        vals[count] = static_cast<intermediate_type>(current_value);
                        ++count;
                    }
                }
            }
            std::sort(vals.begin(), vals.begin() + count);

            ARM_COMPUTE_ERROR_ON(count == 0);

            switch(function)
            {
                case NonLinearFilterFunction::MIN:
                    dst[element_idx] = saturate_cast<T>(vals[0]);
                    break;
                case NonLinearFilterFunction::MAX:
                    dst[element_idx] = saturate_cast<T>(vals[count - 1]);
                    break;
                case NonLinearFilterFunction::MEDIAN:
                    dst[element_idx] = saturate_cast<T>(vals[count / 2]);
                    break;
                default:
                    ARM_COMPUTE_ERROR("Unsupported NonLinearFilter function.");
            }
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> non_linear_filter(const SimpleTensor<uint8_t> &src, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                                                 BorderMode border_mode, uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
