/*
 * Copyright (c) 2020 ARM Limited.
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

#include "LayerNormalizationLayer.h"
#include "ArithmeticOperations.h"
#include "MeanStdDevNormalizationLayer.h"
#include "PixelWiseMultiplication.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
SimpleTensor<float> layer_normalization_layer_float(SimpleTensor<float> src, SimpleTensor<float> weight, SimpleTensor<float> bias)
{
    src = mean_std_normalization_layer(src);
    src = pixel_wise_multiplication(src, weight, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO);
    return arithmetic_operation(ArithmeticOperation::ADD, src, bias, DataType::F32, ConvertPolicy::SATURATE);
}

SimpleTensor<int16_t> layer_normalization_layer(const SimpleTensor<int16_t> &src, const SimpleTensor<int16_t> &weight, const SimpleTensor<int16_t> &bias)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 2);

    SimpleTensor<float> converted_src    = convert_from_symmetric(src);
    SimpleTensor<float> converted_weight = convert_from_symmetric(weight);
    SimpleTensor<float> converted_bias   = convert_from_symmetric(bias);

    SimpleTensor<float> output = layer_normalization_layer_float(converted_src, converted_weight, converted_bias);

    return convert_to_symmetric<int16_t>(output, src.quantization_info());
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
