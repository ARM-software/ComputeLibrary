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

#include "Scale.h"

#include "Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> scale_core(const SimpleTensor<T> &in, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, T constant_border_value,
                           SamplingPolicy sampling_policy, bool ceil_policy_scale)
{
    // Add 1 if ceil_policy_scale is true
    const size_t round_value = ceil_policy_scale ? 1U : 0U;
    TensorShape  shape_scaled(in.shape());
    shape_scaled.set(0, (in.shape()[0] + round_value) * scale_x);
    shape_scaled.set(1, (in.shape()[1] + round_value) * scale_y);
    SimpleTensor<T> out(shape_scaled, in.data_type());

    // Compute the ratio between source width/height and destination width/height
    const auto wr = static_cast<float>(in.shape()[0]) / static_cast<float>(out.shape()[0]);
    const auto hr = static_cast<float>(in.shape()[1]) / static_cast<float>(out.shape()[1]);

    const auto width  = static_cast<int>(in.shape().x());
    const auto height = static_cast<int>(in.shape().y());

    // Determine border size
    const int border_size = (border_mode == BorderMode::UNDEFINED) ? 0 : 1;

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    for(int element_idx = 0, count = 0; element_idx < out.num_elements(); ++element_idx, ++count)
    {
        Coordinates id    = index2coord(out.shape(), element_idx);
        int         idx   = id.x();
        int         idy   = id.y();
        float       x_src = 0;
        float       y_src = 0;

        switch(policy)
        {
            case InterpolationPolicy::NEAREST_NEIGHBOR:
            {
                switch(sampling_policy)
                {
                    case SamplingPolicy::TOP_LEFT:
                        x_src = std::floor(idx * wr);
                        y_src = std::floor(idy * hr);
                        break;
                    case SamplingPolicy::CENTER:
                        //Calculate the source coords without -0.5f is equivalent to round the x_scr/y_src coords
                        x_src = (idx + 0.5f) * wr;
                        y_src = (idy + 0.5f) * hr;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported sampling policy.");
                }

                id.set(0, x_src);
                id.set(1, y_src);

                // If coordinates in range of tensor's width or height
                if(is_valid_pixel_index(x_src, y_src, width, height, border_size))
                {
                    out[element_idx] = tensor_elem_at(in, id, border_mode, constant_border_value);
                }
                break;
            }
            case InterpolationPolicy::BILINEAR:
            {
                switch(sampling_policy)
                {
                    case SamplingPolicy::TOP_LEFT:
                        x_src = idx * wr;
                        y_src = idy * hr;
                        break;
                    case SamplingPolicy::CENTER:
                        x_src = (idx + 0.5f) * wr - 0.5f;
                        y_src = (idy + 0.5f) * hr - 0.5f;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Unsupported sampling policy.");
                }

                id.set(0, std::floor(x_src));
                id.set(1, std::floor(y_src));
                if(is_valid_pixel_index(x_src, y_src, width, height, border_size))
                {
                    out[element_idx] = bilinear_policy(in, id, x_src, y_src, border_mode, constant_border_value);
                }
                else
                {
                    if(border_mode == BorderMode::CONSTANT)
                    {
                        out[element_idx] = constant_border_value;
                    }
                    else if(border_mode == BorderMode::REPLICATE)
                    {
                        id.set(0, utility::clamp<int>(x_src, 0, width - 1));
                        id.set(1, utility::clamp<int>(y_src, 0, height - 1));
                        out[element_idx] = in[coord2index(in.shape(), id)];
                    }
                }
                break;
            }
            case InterpolationPolicy::AREA:
            {
                int       x_from = std::floor(idx * wr - 0.5f - x_src);
                int       y_from = std::floor(idy * hr - 0.5f - y_src);
                int       x_to   = std::ceil((idx + 1) * wr - 0.5f - x_src);
                int       y_to   = std::ceil((idy + 1) * hr - 0.5f - y_src);
                const int xi     = std::floor(x_src);
                const int yi     = std::floor(y_src);

                // Clamp position to borders
                x_src = std::max(-static_cast<float>(border_size), std::min(x_src, static_cast<float>(width - 1 + border_size)));
                y_src = std::max(-static_cast<float>(border_size), std::min(y_src, static_cast<float>(height - 1 + border_size)));

                // Clamp bounding box offsets to borders
                x_from = ((x_src + x_from) < -border_size) ? -border_size : x_from;
                y_from = ((y_src + y_from) < -border_size) ? -border_size : y_from;
                x_to   = ((x_src + x_to) >= (width + border_size)) ? (width - 1 + border_size) : x_to;
                y_to   = ((y_src + y_to) >= (height + border_size)) ? (height - 1 + border_size) : y_to;
                ARM_COMPUTE_ERROR_ON((x_to - x_from + 1) == 0 || (y_to - y_from + 1) == 0);

                float sum = 0;
                for(int j = yi + y_from, je = yi + y_to; j <= je; ++j)
                {
                    for(int i = xi + x_from, ie = xi + x_to; i <= ie; ++i)
                    {
                        id.set(0, static_cast<int>(i));
                        id.set(1, static_cast<int>(j));
                        sum += tensor_elem_at(in, id, border_mode, constant_border_value);
                    }
                }
                out[element_idx] = sum / ((x_to - x_from + 1) * (y_to - y_from + 1));

                break;
            }
            default:
                ARM_COMPUTE_ERROR("Unsupported interpolation mode");
        }
    }

    return out;
}

template <typename T>
SimpleTensor<T> scale(const SimpleTensor<T> &src, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, T constant_border_value,
                      SamplingPolicy sampling_policy, bool ceil_policy_scale)
{
    return scale_core<T>(src, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy, ceil_policy_scale);
}

template <>
SimpleTensor<uint8_t> scale(const SimpleTensor<uint8_t> &src, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value,
                            SamplingPolicy sampling_policy, bool ceil_policy_scale)
{
    SimpleTensor<uint8_t> dst;
    if(src.quantization_info().uniform().scale != 0.f)
    {
        SimpleTensor<float> src_tmp                 = convert_from_asymmetric(src);
        float               constant_border_value_f = dequantize_qasymm8(constant_border_value, src.quantization_info());
        SimpleTensor<float> dst_tmp                 = scale_core<float>(src_tmp, scale_x, scale_y, policy, border_mode, constant_border_value_f, sampling_policy, ceil_policy_scale);
        dst                                         = convert_to_asymmetric(dst_tmp, src.quantization_info());
    }
    else
    {
        dst = scale_core<uint8_t>(src, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy, ceil_policy_scale);
    }
    return dst;
}

template SimpleTensor<int16_t> scale(const SimpleTensor<int16_t> &src, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, int16_t constant_border_value,
                                     SamplingPolicy sampling_policy, bool ceil_policy_scale);
template SimpleTensor<half> scale(const SimpleTensor<half> &src, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, half constant_border_value,
                                  SamplingPolicy sampling_policy, bool ceil_policy_scale);
template SimpleTensor<float> scale(const SimpleTensor<float> &src, float scale_x, float scale_y, InterpolationPolicy policy, BorderMode border_mode, float constant_border_value,
                                   SamplingPolicy sampling_policy, bool ceil_policy_scale);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
