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
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
void fill_mask_from_pattern(uint8_t *mask, int cols, int rows, MatrixPattern pattern)
{
    unsigned int                v = 0;
    std::mt19937                gen(library->seed());
    std::bernoulli_distribution dist(0.5);

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < cols; ++c, ++v)
        {
            uint8_t val = 0;

            switch(pattern)
            {
                case MatrixPattern::BOX:
                    val = 255;
                    break;
                case MatrixPattern::CROSS:
                    val = ((r == (rows / 2)) || (c == (cols / 2))) ? 255 : 0;
                    break;
                case MatrixPattern::DISK:
                    val = (((r - rows / 2.0f + 0.5f) * (r - rows / 2.0f + 0.5f)) / ((rows / 2.0f) * (rows / 2.0f)) + ((c - cols / 2.0f + 0.5f) * (c - cols / 2.0f + 0.5f)) / ((cols / 2.0f) *
                            (cols / 2.0f))) <= 1.0f ? 255 : 0;
                    break;
                case MatrixPattern::OTHER:
                    val = (dist(gen) ? 0 : 255);
                    break;
                default:
                    return;
            }

            mask[v] = val;
        }
    }

    if(pattern == MatrixPattern::OTHER)
    {
        std::uniform_int_distribution<uint8_t> distribution_u8(0, ((cols * rows) - 1));
        mask[distribution_u8(gen)] = 255;
    }
}

TensorShape calculate_depth_concatenate_shape(const std::vector<TensorShape> &input_shapes)
{
    ARM_COMPUTE_ERROR_ON(input_shapes.empty());

    TensorShape out_shape = input_shapes[0];

    size_t max_x = 0;
    size_t max_y = 0;
    size_t depth = 0;

    for(const auto &shape : input_shapes)
    {
        max_x = std::max(shape.x(), max_x);
        max_y = std::max(shape.y(), max_y);
        depth += shape.z();
    }

    out_shape.set(0, max_x);
    out_shape.set(1, max_y);
    out_shape.set(2, depth);

    return out_shape;
}

HarrisCornersParameters harris_corners_parameters()
{
    HarrisCornersParameters params;

    std::mt19937                           gen(library->seed());
    std::uniform_real_distribution<float>  threshold_dist(0.f, 0.01f);
    std::uniform_real_distribution<float>  sensitivity(0.04f, 0.15f);
    std::uniform_real_distribution<float>  euclidean_distance(0.f, 30.f);
    std::uniform_int_distribution<uint8_t> int_dist(0, 255);

    params.threshold             = threshold_dist(gen);
    params.sensitivity           = sensitivity(gen);
    params.min_dist              = euclidean_distance(gen);
    params.constant_border_value = int_dist(gen);

    return params;
}

SimpleTensor<float> convert_from_asymmetric(const SimpleTensor<uint8_t> &src)
{
    const QuantizationInfo &quantization_info = src.quantization_info();
    SimpleTensor<float>     dst{ src.shape(), DataType::F32, 1, 0 };
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantization_info.dequantize(src[i]);
    }
    return dst;
}

SimpleTensor<uint8_t> convert_to_asymmetric(const SimpleTensor<float> &src, const QuantizationInfo &quantization_info)
{
    SimpleTensor<uint8_t> dst{ src.shape(), DataType::QASYMM8, 1, 0, quantization_info };
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = quantization_info.quantize(src[i], RoundingPolicy::TO_NEAREST_UP);
    }
    return dst;
}
} // namespace validation
} // namespace test
} // namespace arm_compute
