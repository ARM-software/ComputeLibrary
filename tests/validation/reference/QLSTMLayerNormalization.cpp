/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/utils/misc/Utility.h"
#include "src/core/utils/quantization/AsymmHelpers.cpp"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
SimpleTensor<int16_t> qlstm_layer_normalization(const SimpleTensor<int16_t> &src, const SimpleTensor<int16_t> &weight, const SimpleTensor<int32_t> &bias)
{
    ARM_COMPUTE_ERROR_ON(src.shape().num_dimensions() > 2);
    SimpleTensor<int16_t> output{ src.shape(), DataType::QSYMM16 };

    const auto wq_info = weight.quantization_info().uniform();
    int        output_multiplier{};
    int        output_shift{};
    const auto s = quantization::calculate_quantized_multiplier(wq_info.scale, &output_multiplier, &output_shift);
    output_shift *= -1;

    if(!bool(s))
    {
        output_multiplier = 0;
        output_shift      = 0;
    }

    const uint32_t num_batch = src.shape()[1];
    const uint32_t num_input = src.shape()[0];

    for(uint32_t batch_idx = 0; batch_idx < num_batch; ++batch_idx)
    {
        int64_t sum{};
        int64_t sum_sq{};

        for(uint32_t input_idx = 0; input_idx < num_input; ++input_idx)
        {
            const auto index = batch_idx * num_input + input_idx;
            const auto val   = static_cast<int32_t>(src[index]);
            sum += val;
            sum_sq += val * val;
        }

        const auto temp     = static_cast<int64_t>(0x100000) / num_input;
        const auto mean     = sum * 1024 / static_cast<int64_t>(num_input);
        const auto variance = ((sum_sq * temp) - (mean * mean)) / 0x100000;

        int32_t stddev_invsqrt_mul{};
        int32_t stddev_invsqrt_shift{};
        quantization::get_invsqrt_quantized_multiplier_exp(variance, -1, stddev_invsqrt_mul, stddev_invsqrt_shift);

        for(uint32_t input_idx = 0; input_idx < num_input; ++input_idx)
        {
            const auto    index           = batch_idx * num_input + input_idx;
            const auto    val             = static_cast<int32_t>(src[index]);
            const auto    shifted         = (val << 10) - mean;
            const auto    rescaled        = quantization::multiply_by_quantized_multiplier(shifted, stddev_invsqrt_mul, stddev_invsqrt_shift);
            const int64_t weighted        = rescaled * weight[input_idx] + bias[input_idx];
            const auto    reverse_shifted = static_cast<int32_t>((weighted + 512) >> 10);
            auto          out_val         = quantization::multiply_by_quantized_multiplier(reverse_shifted, output_multiplier, output_shift + 12);
            out_val                       = arm_compute::utility::clamp<decltype(out_val), int16_t>(out_val, std::numeric_limits<int16_t>::min());
            output[index]                 = static_cast<int16_t>(out_val);
        }
    }
    return output;
}
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
