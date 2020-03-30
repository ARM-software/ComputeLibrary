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

#include "QLSTMLayerNormalization.h"
#include "ArithmeticOperations.h"
#include "MeanStdDevNormalizationLayer.h"
#include "PixelWiseMultiplication.h"
#include "src/core/utils/quantization/AsymmHelpers.cpp"

#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
SimpleTensor<float> qlstm_layer_normalization_float_compute(SimpleTensor<float> src, SimpleTensor<float> weight, SimpleTensor<float> bias)
{
    SimpleTensor<float> output = mean_std_normalization_layer(src);
    output                     = pixel_wise_multiplication<float, float, float>(output, weight, 1, ConvertPolicy::SATURATE, RoundingPolicy::TO_ZERO, DataType::F32);
    return arithmetic_operation(ArithmeticOperation::ADD, output, bias, DataType::F32, ConvertPolicy::SATURATE);
}

SimpleTensor<int16_t> qlstm_layer_normalization(const SimpleTensor<int16_t> &src, const SimpleTensor<int16_t> &weight, const SimpleTensor<int32_t> &bias)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 2);

    SimpleTensor<float> converted_src{ src.shape(), DataType::F32 };
    SimpleTensor<float> converted_weight{ weight.shape(), DataType::F32 };
    SimpleTensor<float> converted_bias{ bias.shape(), DataType::F32 };

    const auto iq_info = src.quantization_info().uniform();
    int        output_multiplier{};
    int        output_shift{};
    quantization::calculate_quantized_multiplier(iq_info.scale, &output_multiplier, &output_shift);

    const float layer_norm_scale = output_multiplier * std::pow(2, static_cast<double>(output_shift - 31));
    const float bias_scale       = std::pow(2., -10) * layer_norm_scale;

    for(int i = 0; i < src.num_elements(); i++)
    {
        converted_src[i] = static_cast<float>(src[i]);
    }

    for(int i = 0; i < bias.num_elements(); i++)
    {
        converted_bias[i] = static_cast<float>(bias[i]) * bias_scale;
    }

    for(int i = 0; i < weight.num_elements(); i++)
    {
        converted_weight[i] = weight[i] * layer_norm_scale;
    }

    SimpleTensor<float>   output_float = qlstm_layer_normalization_float_compute(converted_src, converted_weight, converted_bias);
    SimpleTensor<int16_t> output{ output_float.shape(), DataType::QSYMM16 };

    for(int i = 0; i < output.num_elements(); i++)
    {
        const auto output_val_s32 = static_cast<int32_t>(support::cpp11::round(output_float[i] * std::pow(2, 12)));
        output[i]                 = utility::clamp<int32_t, int16_t>(output_val_s32, std::numeric_limits<int16_t>::min());
    }

    return output;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
