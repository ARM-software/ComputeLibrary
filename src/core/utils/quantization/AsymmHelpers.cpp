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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/Helpers.h"

#include <cmath>
#include <limits>
#include <numeric>

namespace arm_compute
{
namespace quantization
{
constexpr int64_t fixed_point_one_Q0 = (1LL << 31);
constexpr float   epsilon            = 0.00001f;

Status calculate_quantized_multiplier(float multiplier, int *quant_multiplier, int *shift)
{
    if(multiplier > 1.f)
    {
        Status status = calculate_quantized_multiplier_greater_than_one(multiplier, quant_multiplier, shift);
        *shift *= -1;
        return status;
    }
    else
    {
        return calculate_quantized_multiplier_less_than_one(multiplier, quant_multiplier, shift);
    }
}

Status calculate_quantized_multiplier_less_than_one(float multiplier,
                                                    int *quant_multiplier,
                                                    int *right_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quant_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(right_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < -epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f + epsilon);
    if(std::fabs(1.0f - multiplier) < epsilon)
    {
        *quant_multiplier = 1;
        *right_shift      = 0;
        return Status{};
    }

    if(std::fabs(0.0f - multiplier) < epsilon)
    {
        *quant_multiplier = 0;
        *right_shift      = 0;
        return Status{};
    }

    const double q = std::frexp(multiplier, right_shift);
    *right_shift *= -1;
    auto q_fixed = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);
    if(q_fixed == fixed_point_one_Q0)
    {
        q_fixed /= 2;
        --*right_shift;
    }
    ARM_COMPUTE_RETURN_ERROR_ON(*right_shift < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
    *quant_multiplier = static_cast<int32_t>(q_fixed);

    return Status{};
}

Status calculate_quantized_multiplier_greater_than_one(float multiplier,
                                                       int *quantized_multiplier,
                                                       int *left_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quantized_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(left_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < 1.f);
    const double q       = std::frexp(multiplier, left_shift);
    auto         q_fixed = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > fixed_point_one_Q0);
    if(q_fixed == fixed_point_one_Q0)
    {
        q_fixed /= 2;
        ++*left_shift;
    }
    ARM_COMPUTE_RETURN_ERROR_ON(*left_shift < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(q_fixed > std::numeric_limits<int32_t>::max());
    *quantized_multiplier = static_cast<int32_t>(q_fixed);

    return Status{};
}
std::pair<int, int> get_min_max_values_from_quantized_data_type(DataType data_type)
{
    int min_quant_val = 0;
    int max_quant_val = 0;
    switch(data_type)
    {
        case DataType::QASYMM8:
            min_quant_val = std::numeric_limits<uint8_t>::min();
            max_quant_val = std::numeric_limits<uint8_t>::max();
            break;
        case DataType::QSYMM8:
            min_quant_val = std::numeric_limits<int8_t>::min();
            max_quant_val = std::numeric_limits<int8_t>::max();
            break;
        case DataType::QASYMM16:
            min_quant_val = std::numeric_limits<uint16_t>::min();
            max_quant_val = std::numeric_limits<uint16_t>::max();
            break;
        case DataType::QSYMM16:
            min_quant_val = std::numeric_limits<int16_t>::min();
            max_quant_val = std::numeric_limits<int16_t>::max();
            break;
        default:
            ARM_COMPUTE_ERROR("Unsupported data type");
    }
    return std::make_pair(min_quant_val, max_quant_val);
}
void compute_quantized_multipliers_and_shifts(const ITensor *input, const ITensor *weights, const ITensor *output, int32_t *output_multipliers_ptr, int32_t *output_shifts_ptr)
{
    const unsigned int idx_c       = get_data_layout_dimension_index(weights->info()->data_layout(), DataLayoutDimension::CHANNEL);
    const unsigned int num_filters = is_data_type_quantized_per_channel(weights->info()->data_type()) ? weights->info()->dimension(idx_c) : 1;

    const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
    const QuantizationInfo        wq_info = weights->info()->quantization_info();
    const UniformQuantizationInfo oq_info = output->info()->quantization_info().uniform();

    for(unsigned int i = 0; i < num_filters; ++i)
    {
        int         output_multiplier = 0;
        int         output_shift      = 0;
        const float multiplier        = iq_info.scale * wq_info.scale()[i] / oq_info.scale;
        ARM_COMPUTE_ERROR_ON(multiplier > 1.0f);
        calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        output_multipliers_ptr[i] = output_multiplier;
        output_shifts_ptr[i]      = output_shift;
    }
}
} // quantization
} // arm_compute
