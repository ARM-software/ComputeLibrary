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
#include "PriorBoxLayer.h"

#include "ActivationLayer.h"

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
SimpleTensor<T> prior_box_layer(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, const PriorBoxLayerInfo &info, const TensorShape &output_shape)
{
    const auto layer_width  = static_cast<int>(src1.shape()[0]);
    const auto layer_height = static_cast<int>(src1.shape()[1]);

    int img_width  = info.img_size().x;
    int img_height = info.img_size().y;
    if(img_width == 0 || img_height == 0)
    {
        img_width  = static_cast<int>(src2.shape()[0]);
        img_height = static_cast<int>(src2.shape()[1]);
    }

    float step_x = info.steps()[0];
    float step_y = info.steps()[1];
    if(step_x == 0.f || step_y == 0.f)
    {
        step_x = static_cast<float>(img_width) / layer_width;
        step_x = static_cast<float>(img_height) / layer_height;
    }

    // Calculate number of aspect ratios
    const int num_priors     = info.aspect_ratios().size() * info.min_sizes().size() + info.max_sizes().size();
    const int total_elements = layer_width * layer_height * num_priors * 4;

    SimpleTensor<T> result(output_shape, src1.data_type());

    int idx = 0;
    for(int y = 0; y < layer_height; ++y)
    {
        for(int x = 0; x < layer_width; ++x)
        {
            const float center_x = (x + info.offset()) * step_x;
            const float center_y = (y + info.offset()) * step_y;
            float       box_width;
            float       box_height;
            for(unsigned int i = 0; i < info.min_sizes().size(); ++i)
            {
                const float min_size = info.min_sizes().at(i);
                box_width            = min_size;
                box_height           = min_size;
                // (xmin, ymin, xmax, ymax)
                result[idx++] = (center_x - box_width / 2.f) / img_width;
                result[idx++] = (center_y - box_height / 2.f) / img_height;
                result[idx++] = (center_x + box_width / 2.f) / img_width;
                result[idx++] = (center_y + box_height / 2.f) / img_height;

                if(!info.max_sizes().empty())
                {
                    const float max_size = info.max_sizes().at(i);
                    box_width            = sqrt(min_size * max_size);
                    box_height           = box_width;

                    // (xmin, ymin, xmax, ymax)
                    result[idx++] = (center_x - box_width / 2.f) / img_width;
                    result[idx++] = (center_y - box_height / 2.f) / img_height;
                    result[idx++] = (center_x + box_width / 2.f) / img_width;
                    result[idx++] = (center_y + box_height / 2.f) / img_height;
                }

                // rest of priors
                for(auto ar : info.aspect_ratios())
                {
                    if(fabs(ar - 1.) < 1e-6)
                    {
                        continue;
                    }

                    box_width  = min_size * sqrt(ar);
                    box_height = min_size / sqrt(ar);

                    // (xmin, ymin, xmax, ymax)
                    result[idx++] = (center_x - box_width / 2.f) / img_width;
                    result[idx++] = (center_y - box_height / 2.f) / img_height;
                    result[idx++] = (center_x + box_width / 2.f) / img_width;
                    result[idx++] = (center_y + box_height / 2.f) / img_height;
                }
            }
        }
    }

    // clip the coordinates
    if(info.clip())
    {
        for(int i = 0; i < total_elements; ++i)
        {
            result[i] = std::min<T>(std::max<T>(result[i], 0.f), 1.f);
        }
    }

    // set the variance.
    if(info.variances().size() == 1)
    {
        std::fill_n(result.data() + idx, total_elements, info.variances().at(0));
    }
    else
    {
        for(int h = 0; h < layer_height; ++h)
        {
            for(int w = 0; w < layer_width; ++w)
            {
                for(int i = 0; i < num_priors; ++i)
                {
                    for(int j = 0; j < 4; ++j)
                    {
                        result[idx++] = info.variances().at(j);
                    }
                }
            }
        }
    }

    return result;
}
template SimpleTensor<float> prior_box_layer(const SimpleTensor<float> &src1, const SimpleTensor<float> &src2, const PriorBoxLayerInfo &info, const TensorShape &output_shape);

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
