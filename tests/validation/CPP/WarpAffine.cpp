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
#include "WarpAffine.h"

#include "Utils.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
bool valid_bilinear_policy(float xn, float yn, int width, int height, BorderMode border_mode)
{
    if(border_mode != BorderMode::UNDEFINED)
    {
        return true;
    }
    if((0 <= yn + 1) && (yn + 1 < height) && (0 <= xn + 1) && (xn + 1 < width))
    {
        return true;
    }
    return false;
}

template <typename T>
SimpleTensor<T> warp_affine(const SimpleTensor<T> &src, SimpleTensor<T> &valid_mask, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    // x0 = M00 * x + M01 * y + M02
    // y0 = M10 * x + M11 * y + M12
    const float M00 = matrix[0];
    const float M10 = matrix[1];
    const float M01 = matrix[0 + 1 * 2];
    const float M11 = matrix[1 + 1 * 2];
    const float M02 = matrix[0 + 2 * 2];
    const float M12 = matrix[1 + 2 * 2];

    const int width  = src.shape().x();
    const int height = src.shape().y();

    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        valid_mask[element_idx] = 1;
        Coordinates id          = index2coord(src.shape(), element_idx);
        int         idx         = id.x();
        int         idy         = id.y();

        float x0 = M00 * idx + M01 * idy + M02;
        float y0 = M10 * idx + M11 * idy + M12;

        id.set(0, static_cast<int>(std::floor(x0)));
        id.set(1, static_cast<int>(std::floor(y0)));
        if((0 <= y0) && (y0 < height) && (0 <= x0) && (x0 < width))
        {
            switch(policy)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                    dst[element_idx] = tensor_elem_at(src, id, border_mode, constant_border_value);
                    break;
                case InterpolationPolicy::BILINEAR:
                    (valid_bilinear_policy(x0, y0, width, height, border_mode)) ? dst[element_idx] = bilinear_policy(src, id, x0, y0, border_mode, constant_border_value) :
                                                                                                     valid_mask[element_idx] = 0;
                    break;
                case InterpolationPolicy::AREA:
                default:
                    ARM_COMPUTE_ERROR("Interpolation not supported");
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
                            id.set(0, std::max(0, std::min(static_cast<int>(x0), width - 1)));
                            id.set(1, std::max(0, std::min(static_cast<int>(y0), height - 1)));
                            dst[element_idx] = src[coord2index(src.shape(), id)];
                        }
                        break;
                    case InterpolationPolicy::BILINEAR:
                        dst[element_idx] = bilinear_policy(src, id, x0, y0, border_mode, constant_border_value);
                        break;
                    case InterpolationPolicy::AREA:
                    default:
                        ARM_COMPUTE_ERROR("Interpolation not supported");
                }
            }
        }
    }

    return dst;
}

template SimpleTensor<uint8_t> warp_affine(const SimpleTensor<uint8_t> &src, SimpleTensor<uint8_t> &valid_mask, const float *matrix, InterpolationPolicy policy, BorderMode border_mode,
                                           uint8_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute