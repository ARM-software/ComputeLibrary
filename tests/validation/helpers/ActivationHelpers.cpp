/*
 * Copyright (c) 2017-2025 Arm Limited.
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

#include "tests/validation/helpers/ActivationHelpers.h"

#include "arm_compute/core/utils/DataTypeUtils.h"

#include <initializer_list>
#include <limits>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace helper
{
RelativeTolerance<float> relative_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation)
{
    switch (activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::ELU:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
        case ActivationLayerInfo::ActivationFunction::SWISH:
        case ActivationLayerInfo::ActivationFunction::GELU:
            switch (data_type)
            {
                case DataType::F16:
#if defined(ENABLE_SVE)
                    return RelativeTolerance<float>(0.25f);
#else  // !defined(ENABLE_SVE)
                    return RelativeTolerance<float>(0.1f);
#endif // defined(ENABLE_SVE)
                default:
                    return RelativeTolerance<float>(0.05f);
            }
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            switch (data_type)
            {
                case DataType::F16:
#if defined(ENABLE_SVE)
                    return RelativeTolerance<float>(0.9f);
#else  // !defined(ENABLE_SVE)
                    return RelativeTolerance<float>(0.01f);
#endif // defined(ENABLE_SVE)
                default:
                    return RelativeTolerance<float>(0.00001f);
            }
        default:
            return RelativeTolerance<float>(0.f);
    }
}

AbsoluteTolerance<float> absolute_tolerance(DataType data_type, ActivationLayerInfo::ActivationFunction activation)
{
    switch (activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
        case ActivationLayerInfo::ActivationFunction::SWISH:
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
            switch (data_type)
            {
                case DataType::F16:
#if defined(ENABLE_SVE)
                    return AbsoluteTolerance<float>(0.25f);
#else  // !defined(ENABLE_SVE)
                    return AbsoluteTolerance<float>(0.01f);
#endif // defined(ENABLE_SVE)
                default:
                    return AbsoluteTolerance<float>(0.00001f);
            }
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            switch (data_type)
            {
                case DataType::F16:
#if defined(ENABLE_SVE)
                    return AbsoluteTolerance<float>(0.9f);
#else  // !defined(ENABLE_SVE)
                    return AbsoluteTolerance<float>(0.01f);
#endif // defined(ENABLE_SVE)
                default:
                    return AbsoluteTolerance<float>(0.00001f);
            }
        default:
            return AbsoluteTolerance<float>(0.f);
    }
}

float tolerance_num(DataType data_type, ActivationLayerInfo::ActivationFunction activation)
{
    switch (data_type)
    {
        case DataType::F16:
            switch (activation)
            {
                case ActivationLayerInfo::ActivationFunction::GELU:
                case ActivationLayerInfo::ActivationFunction::ELU:
                case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
                case ActivationLayerInfo::ActivationFunction::SWISH:
                    return 0.01f;
                default:
                    return 0.f;
            }
        case DataType::F32:
        default:
            return 0.f;
    }
}

QuantizationInfo calculate_output_quantization_info(DataType                   data_type,
                                                    const ActivationLayerInfo &act_info,
                                                    const QuantizationInfo    &default_qinfo)
{
    auto qasymm8_max        = float(std::numeric_limits<uint8_t>::max()) + 1.f;
    auto qasymm8_signed_max = float(std::numeric_limits<int8_t>::max()) + 1.f;
    auto qsymm16_max        = float(std::numeric_limits<int16_t>::max()) + 1.f;

    switch (act_info.activation())
    {
        case ActivationLayerInfo::ActivationFunction::TANH:
            if (data_type == DataType::QSYMM16)
            {
                return QuantizationInfo(1.f / qsymm16_max, 0);
            }
            else if (data_type == DataType::QASYMM8)
            {
                return QuantizationInfo(1.f / (0.5 * qasymm8_max), int(0.5 * qasymm8_max));
            }
            else if (data_type == DataType::QASYMM8_SIGNED)
            {
                return QuantizationInfo(1.f / qasymm8_signed_max, 0);
            }
            else
            {
                return default_qinfo;
            }
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            if (data_type == DataType::QSYMM16)
            {
                return QuantizationInfo(1.f / qsymm16_max, 0);
            }
            else if (data_type == DataType::QASYMM8)
            {
                return QuantizationInfo(1.f / qasymm8_max, 0);
            }
            else if (data_type == DataType::QASYMM8_SIGNED)
            {
                return QuantizationInfo(1.f / (2.f * qasymm8_signed_max), -int(qasymm8_signed_max));
            }
            else
            {
                return default_qinfo;
            }
        default:
            return default_qinfo;
    }
}

template <typename T>
std::vector<T> get_boundary_values(DataType data_type, T min, T max)
{
    // This function will return a vector filled with the following values that can
    // represent two partitions derived from equivalent partitioning.
    // * Lower parition: min, min + delta, lower quarter (nominal), center - delta
    // * Upper partition: center, center + delta, upper quarter (nominal), max - delta, max
    const auto delta         = is_data_type_float(data_type) ? T(0.1f) : T(1);
    const auto center_value  = (min + max) / 2;
    const auto lower_quarter = (min + center_value) / 2;
    const auto upper_quarter = (center_value + max) / 2;

    std::vector<T> boundary_values{};

    // To ensure all the inserted values are within the given range after subtracing/adding delta
    auto insert_values = [&boundary_values, &min, &max](const std::initializer_list<T> &new_values)
    {
        for (auto &v : new_values)
        {
            if (v >= min && v <= max)
            {
                boundary_values.emplace_back(v);
            }
        }
    };

    insert_values({min, static_cast<T>(min + delta), static_cast<T>(lower_quarter),
                   static_cast<T>(center_value - delta)}); // lower partition
    insert_values({static_cast<T>(center_value), static_cast<T>(center_value + delta), static_cast<T>(upper_quarter),
                   static_cast<T>(max - delta), max}); // upper partition

    return boundary_values;
}

// Explicit instatiations to keep the implementation in the cpp file
template std::vector<float>   get_boundary_values(DataType data_type, float min, float max);
template std::vector<half>    get_boundary_values(DataType data_type, half min, half max);
template std::vector<int8_t>  get_boundary_values(DataType data_type, int8_t min, int8_t max);
template std::vector<uint8_t> get_boundary_values(DataType data_type, uint8_t min, uint8_t max);
template std::vector<int16_t> get_boundary_values(DataType data_type, int16_t min, int16_t max);

AbsoluteTolerance<uint8_t> tolerance_qasymm8(ActivationLayerInfo::ActivationFunction activation)
{
    switch (activation)
    {
        case ActivationLayerInfo::ActivationFunction::LOGISTIC:
        case ActivationLayerInfo::ActivationFunction::SQRT:
        case ActivationLayerInfo::ActivationFunction::TANH:
        case ActivationLayerInfo::ActivationFunction::HARD_SWISH:
        case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
        case ActivationLayerInfo::ActivationFunction::LEAKY_RELU:
            return AbsoluteTolerance<uint8_t>(1);
        default:
            return AbsoluteTolerance<uint8_t>(0);
    }
}

} // namespace helper
} // namespace validation
} // namespace test
} // namespace arm_compute
