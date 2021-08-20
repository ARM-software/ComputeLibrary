/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "Remap.h"

#include "Utils.h"
#include "tests/validation/Helpers.h"

#include <algorithm>
#include <array>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> remap(const SimpleTensor<T> &in, SimpleTensor<float> &map_x, SimpleTensor<float> &map_y, SimpleTensor<T> &valid_mask, InterpolationPolicy policy, BorderMode border_mode,
                      T constant_border_value)
{
    ARM_COMPUTE_ERROR_ON_MSG(border_mode == BorderMode::REPLICATE, "BorderMode not supported");
    SimpleTensor<T> out(in.shape(), in.data_type());
    ARM_COMPUTE_ERROR_ON(out.num_elements() != map_x.num_elements());
    const int      width        = in.shape().x();
    const int      height       = in.shape().y();
    const uint32_t num_elements = out.num_elements();
    for(uint32_t idx = 0; idx < num_elements; idx++)
    {
        const Coordinates id_out = index2coord(out.shape(), idx);
        valid_mask[idx]          = 1;
        Coordinates src_idx      = id_out; // need to setup all coordinates and not just xy
        if((0 <= map_y[idx]) && (map_y[idx] < height) && (0 <= map_x[idx]) && (map_x[idx] < width))
        {
            switch(policy)
            {
                case InterpolationPolicy::NEAREST_NEIGHBOR:
                {
                    src_idx.set(0, static_cast<int>(std::floor(map_x[idx])));
                    src_idx.set(1, static_cast<int>(std::floor(map_y[idx])));
                    out[idx] = in[coord2index(in.shape(), src_idx)];
                    break;
                }
                case InterpolationPolicy::BILINEAR:
                {
                    (valid_bilinear_policy(map_x[idx], map_y[idx], width, height, border_mode)) ?
                    out[idx]        = bilinear_policy(in, src_idx, map_x[idx], map_y[idx], border_mode, constant_border_value) :
                                      valid_mask[idx] = 0;
                    break;
                }
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
                valid_mask[idx] = 0;
            }
            else
            {
                switch(policy)
                {
                    case InterpolationPolicy::NEAREST_NEIGHBOR:
                        out[idx] = constant_border_value;
                        break;
                    case InterpolationPolicy::BILINEAR:
                        out[idx] = bilinear_policy(in, src_idx, map_x[idx], map_y[idx], border_mode, constant_border_value);
                        break;
                    case InterpolationPolicy::AREA:
                    default:
                        break;
                }
            }
        }
    }
    return out;
}

template SimpleTensor<uint8_t> remap(const SimpleTensor<uint8_t> &src, SimpleTensor<float> &map_x, SimpleTensor<float> &map_y, SimpleTensor<uint8_t> &valid_mask, InterpolationPolicy policy,
                                     BorderMode border_mode,
                                     uint8_t    constant_border_value);

template SimpleTensor<half> remap(const SimpleTensor<half> &src, SimpleTensor<float> &map_x, SimpleTensor<float> &map_y, SimpleTensor<half> &valid_mask, InterpolationPolicy policy,
                                  BorderMode border_mode,
                                  half       constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
