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
#include "arm_compute/core/Helpers.h"

#include "Utils.h"
#include "WarpPerspective.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> warp_perspective(const SimpleTensor<T> &src, SimpleTensor<T> &valid_mask, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    // z0 = M20 * x + M21 * y + M22
    // xn = x0 / z0
    // yn = y0 / z0
    const float M00 = matrix[0];
    const float M10 = matrix[1];
    const float M20 = matrix[2];
    const float M01 = matrix[0 + 1 * 3];
    const float M11 = matrix[1 + 1 * 3];
    const float M21 = matrix[2 + 1 * 3];
    const float M02 = matrix[0 + 2 * 3];
    const float M12 = matrix[1 + 2 * 3];
    const float M22 = matrix[2 + 2 * 3];

    const int width  = src.shape().x();
    const int height = src.shape().y();

    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        valid_mask[element_idx] = 1;
        Coordinates id          = index2coord(src.shape(), element_idx);
        const int   idx         = id.x();
        const int   idy         = id.y();
        const float z0          = M20 * idx + M21 * idy + M22;

        const float x0 = (M00 * idx + M01 * idy + M02);
        const float y0 = (M10 * idx + M11 * idy + M12);

        const float xn = x0 / z0;
        const float yn = y0 / z0;
        id.set(0, static_cast<int>(std::floor(xn)));
        id.set(1, static_cast<int>(std::floor(yn)));
        if((0 <= yn) && (yn < height) && (0 <= xn) && (xn < width))
        {
            switch(policy)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    dst[element_idx] = tensor_elem_at(src, id, border_mode, constant_border_value);
                    break;
                case InterpolationPolicy::BILINEAR:
                    (valid_bilinear_policy(xn, yn, width, height, border_mode)) ? dst[element_idx] = bilinear_policy(src, id, xn, yn, border_mode, constant_border_value) : valid_mask[element_idx] = 0;
                    break;
                case InterpolationPolicy::AREA:
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
                    break;
            }
        }
        else
        {
            if(border_mode == BorderMode::UNDEFINED)
            {
                valid_mask[element_idx] = 0;
            }
            else
            {
                switch(policy)
                {
                    case InterpolationPolicy::NEAREST_NEIGHBOR:
                        if(border_mode == BorderMode::CONSTANT)
                        {
                            dst[element_idx] = constant_border_value;
                        }
                        else if(border_mode == BorderMode::REPLICATE)
                        {
                            id.set(0, std::max(0, std::min(static_cast<int>(xn), width - 1)));
                            id.set(1, std::max(0, std::min(static_cast<int>(yn), height - 1)));
                            dst[element_idx] = src[coord2index(src.shape(), id)];
                        }
                        break;
                    case InterpolationPolicy::BILINEAR:
                        dst[element_idx] = bilinear_policy(src, id, xn, yn, border_mode, constant_border_value);
                        break;
                    case InterpolationPolicy::AREA:
                    default:
                        ARM_COMPUTE_ERROR("Interpolation not supported");
                        break;
                }
            }
        }
    }
    return dst;
}

template SimpleTensor<uint8_t> warp_perspective(const SimpleTensor<uint8_t> &src, SimpleTensor<uint8_t> &valid_mask, const float *matrix, InterpolationPolicy policy, BorderMode border_mode,
                                                uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
