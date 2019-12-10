/*
 * Copyright (c) 2017-2020 ARM Limited.
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

Status calculate_quantized_multiplier(float multiplier, int32_t *quant_multiplier, int32_t *shift)
{
    if(multiplier >= 1.f)
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

Status calculate_quantized_multiplier_less_than_one(float    multiplier,
                                                    int32_t *quant_multiplier,
                                                    int32_t *right_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quant_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(right_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < -epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f + epsilon);
    if(std::fabs(0.0f - multiplier) < epsilon)
    {
        *quant_multiplier = 0;
        *right_shift      = 0;
        return Status{};
    }

    int          shift_exp = 0;
    const double q         = std::frexp(multiplier, &shift_exp);
    *right_shift           = -1 * shift_exp;
    auto q_fixed           = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
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

Status calculate_quantized_multiplier_greater_than_one(float    multiplier,
                                                       int32_t *quantized_multiplier,
                                                       int32_t *left_shift)
{
    ARM_COMPUTE_RETURN_ERROR_ON(quantized_multiplier == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(left_shift == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(multiplier < 1.f);

    int          shift_exp = 0;
    const double q         = std::frexp(multiplier, &shift_exp);
    *left_shift            = shift_exp;
    auto q_fixed           = static_cast<int64_t>(support::cpp11::round(q * fixed_point_one_Q0));
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

arm_compute::Status calculate_quantized_multipliers(const QuantizationInfo &iq_info,
                                                    const QuantizationInfo &wq_info,
                                                    const QuantizationInfo &oq_info,
                                                    GEMMLowpOutputStageInfo &stage_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(iq_info.scale().empty());
    ARM_COMPUTE_RETURN_ERROR_ON(wq_info.scale().empty());
    ARM_COMPUTE_RETURN_ERROR_ON(oq_info.scale().empty());

    const unsigned int size = wq_info.scale().size();

    auto &quant_multipliers = stage_info.gemmlowp_multipliers;
    auto &quant_shifts      = stage_info.gemmlowp_shifts;
    quant_multipliers.resize(size);
    quant_shifts.resize(size);

    const auto &w_scales = wq_info.scale();
    const float i_scale  = iq_info.scale().at(0);
    const float o_scale  = oq_info.scale().at(0);

    for(unsigned int i = 0; i < size; ++i)
    {
        const float multiplier       = i_scale * w_scales[i] / o_scale;
        int32_t     quant_multiplier = 0;
        int32_t     quant_shift      = 0;
        ARM_COMPUTE_RETURN_ON_ERROR(calculate_quantized_multiplier(multiplier, &quant_multiplier, &quant_shift));
        quant_multipliers[i] = quant_multiplier;
        quant_shifts[i]      = quant_shift;
    }

    // Legacy part
    stage_info.gemmlowp_shift      = quant_shifts[0];
    stage_info.gemmlowp_multiplier = quant_multipliers[0];

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
        case DataType::QASYMM8_SIGNED:
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
void compute_quantized_multipliers_and_shifts(const ITensorInfo *input,
                                              const ITensorInfo *weights,
                                              const ITensorInfo *output,
                                              unsigned int       idx_ofms,
                                              int32_t           *output_multipliers_ptr,
                                              int32_t           *output_shifts_ptr)
{
    const unsigned int num_filters = is_data_type_quantized_per_channel(weights->data_type()) ? weights->dimension(idx_ofms) : 1;

    const UniformQuantizationInfo iq_info = input->quantization_info().uniform();
    const QuantizationInfo        wq_info = weights->quantization_info();
    const UniformQuantizationInfo oq_info = output->quantization_info().uniform();

    for(unsigned int i = 0; i < num_filters; ++i)
    {
        int32_t     output_multiplier = 0;
        int32_t     output_shift      = 0;
        const float multiplier        = iq_info.scale * wq_info.scale()[i] / oq_info.scale;
        calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);

        output_multipliers_ptr[i] = output_multiplier;
        output_shifts_ptr[i]      = output_shift;
    }
}
} // quantization
} // arm_compute
