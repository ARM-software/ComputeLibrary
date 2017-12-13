/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "Validation.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
/** Get the data from *ptr after casting according to @p data_type and then convert the data to double.
 *
 * @param[in] ptr       Pointer to value.
 * @param[in] data_type Data type of both values.
 *
 * @return The data from the ptr after converted to double.
 */
double get_double_data(const void *ptr, DataType data_type)
{
    if(ptr == nullptr)
    {
        ARM_COMPUTE_ERROR("Can't dereference a null pointer!");
    }

    switch(data_type)
    {
        case DataType::U8:
            return *reinterpret_cast<const uint8_t *>(ptr);
        case DataType::S8:
            return *reinterpret_cast<const int8_t *>(ptr);
        case DataType::QS8:
            return *reinterpret_cast<const qint8_t *>(ptr);
        case DataType::U16:
            return *reinterpret_cast<const uint16_t *>(ptr);
        case DataType::S16:
            return *reinterpret_cast<const int16_t *>(ptr);
        case DataType::QS16:
            return *reinterpret_cast<const qint16_t *>(ptr);
        case DataType::U32:
            return *reinterpret_cast<const uint32_t *>(ptr);
        case DataType::S32:
            return *reinterpret_cast<const int32_t *>(ptr);
        case DataType::U64:
            return *reinterpret_cast<const uint64_t *>(ptr);
        case DataType::S64:
            return *reinterpret_cast<const int64_t *>(ptr);
        case DataType::F16:
            return *reinterpret_cast<const half *>(ptr);
        case DataType::F32:
            return *reinterpret_cast<const float *>(ptr);
        case DataType::F64:
            return *reinterpret_cast<const double *>(ptr);
        case DataType::SIZET:
            return *reinterpret_cast<const size_t *>(ptr);
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
}

void check_border_element(const IAccessor &tensor, const Coordinates &id,
                          const BorderMode &border_mode, const void *border_value,
                          int64_t &num_elements, int64_t &num_mismatches)
{
    const size_t channel_size = element_size_from_data_type(tensor.data_type());
    const auto   ptr          = static_cast<const uint8_t *>(tensor(id));

    if(border_mode == BorderMode::REPLICATE)
    {
        Coordinates border_id{ id };

        if(id.x() < 0)
        {
            border_id.set(0, 0);
        }
        else if(static_cast<size_t>(id.x()) >= tensor.shape().x())
        {
            border_id.set(0, tensor.shape().x() - 1);
        }

        if(id.y() < 0)
        {
            border_id.set(1, 0);
        }
        else if(static_cast<size_t>(id.y()) >= tensor.shape().y())
        {
            border_id.set(1, tensor.shape().y() - 1);
        }

        border_value = tensor(border_id);
    }

    // Iterate over all channels within one element
    for(int channel = 0; channel < tensor.num_channels(); ++channel)
    {
        const size_t channel_offset = channel * channel_size;
        const double target         = get_double_data(ptr + channel_offset, tensor.data_type());
        const double reference      = get_double_data(static_cast<const uint8_t *>(border_value) + channel_offset, tensor.data_type());

        if(!compare<AbsoluteTolerance<double>>(target, reference))
        {
            ARM_COMPUTE_TEST_INFO("id = " << id);
            ARM_COMPUTE_TEST_INFO("channel = " << channel);
            ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << target);
            ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << reference);
            ARM_COMPUTE_EXPECT_EQUAL(target, reference, framework::LogLevel::DEBUG);

            ++num_mismatches;
        }

        ++num_elements;
    }
}
} // namespace

void validate(const arm_compute::ValidRegion &region, const arm_compute::ValidRegion &reference)
{
    ARM_COMPUTE_EXPECT_EQUAL(region.anchor.num_dimensions(), reference.anchor.num_dimensions(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(region.shape.num_dimensions(), reference.shape.num_dimensions(), framework::LogLevel::ERRORS);

    for(unsigned int d = 0; d < region.anchor.num_dimensions(); ++d)
    {
        ARM_COMPUTE_EXPECT_EQUAL(region.anchor[d], reference.anchor[d], framework::LogLevel::ERRORS);
    }

    for(unsigned int d = 0; d < region.shape.num_dimensions(); ++d)
    {
        ARM_COMPUTE_EXPECT_EQUAL(region.shape[d], reference.shape[d], framework::LogLevel::ERRORS);
    }
}

void validate(const arm_compute::PaddingSize &padding, const arm_compute::PaddingSize &reference)
{
    ARM_COMPUTE_EXPECT_EQUAL(padding.top, reference.top, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.right, reference.right, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.bottom, reference.bottom, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.left, reference.left, framework::LogLevel::ERRORS);
}

void validate(const arm_compute::PaddingSize &padding, const arm_compute::PaddingSize &width_reference, const arm_compute::PaddingSize &height_reference)
{
    ARM_COMPUTE_EXPECT_EQUAL(padding.top, height_reference.top, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.right, width_reference.right, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.bottom, height_reference.bottom, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(padding.left, width_reference.left, framework::LogLevel::ERRORS);
}

void validate(const IAccessor &tensor, const void *reference_value)
{
    ARM_COMPUTE_ASSERT(reference_value != nullptr);

    int64_t      num_mismatches = 0;
    int64_t      num_elements   = 0;
    const size_t channel_size   = element_size_from_data_type(tensor.data_type());

    // Iterate over all elements, e.g. U8, S16, RGB888, ...
    for(int element_idx = 0; element_idx < tensor.num_elements(); ++element_idx)
    {
        const Coordinates id = index2coord(tensor.shape(), element_idx);

        const auto ptr = static_cast<const uint8_t *>(tensor(id));

        // Iterate over all channels within one element
        for(int channel = 0; channel < tensor.num_channels(); ++channel)
        {
            const size_t channel_offset = channel * channel_size;
            const double target         = get_double_data(ptr + channel_offset, tensor.data_type());
            const double reference      = get_double_data(reference_value, tensor.data_type());

            if(!compare<AbsoluteTolerance<double>>(target, reference))
            {
                ARM_COMPUTE_TEST_INFO("id = " << id);
                ARM_COMPUTE_TEST_INFO("channel = " << channel);
                ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << target);
                ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << reference);
                ARM_COMPUTE_EXPECT_EQUAL(target, reference, framework::LogLevel::DEBUG);

                ++num_mismatches;
            }

            ++num_elements;
        }
    }

    if(num_elements > 0)
    {
        const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches << "%) mismatched");
        ARM_COMPUTE_EXPECT_EQUAL(num_mismatches, 0, framework::LogLevel::ERRORS);
    }
}

void validate(const IAccessor &tensor, BorderSize border_size, const BorderMode &border_mode, const void *border_value)
{
    if(border_mode == BorderMode::UNDEFINED)
    {
        return;
    }
    else if(border_mode == BorderMode::CONSTANT)
    {
        ARM_COMPUTE_ASSERT(border_value != nullptr);
    }

    int64_t   num_mismatches = 0;
    int64_t   num_elements   = 0;
    const int slice_size     = tensor.shape()[0] * tensor.shape()[1];

    for(int element_idx = 0; element_idx < tensor.num_elements(); element_idx += slice_size)
    {
        Coordinates id = index2coord(tensor.shape(), element_idx);

        // Top border
        for(int y = -border_size.top; y < 0; ++y)
        {
            id.set(1, y);

            for(int x = -border_size.left; x < static_cast<int>(tensor.shape()[0]) + static_cast<int>(border_size.right); ++x)
            {
                id.set(0, x);

                check_border_element(tensor, id, border_mode, border_value, num_elements, num_mismatches);
            }
        }

        // Bottom border
        for(int y = tensor.shape()[1]; y < static_cast<int>(tensor.shape()[1]) + static_cast<int>(border_size.bottom); ++y)
        {
            id.set(1, y);

            for(int x = -border_size.left; x < static_cast<int>(tensor.shape()[0]) + static_cast<int>(border_size.right); ++x)
            {
                id.set(0, x);

                check_border_element(tensor, id, border_mode, border_value, num_elements, num_mismatches);
            }
        }

        // Left/right border
        for(int y = 0; y < static_cast<int>(tensor.shape()[1]); ++y)
        {
            id.set(1, y);

            // Left border
            for(int x = -border_size.left; x < 0; ++x)
            {
                id.set(0, x);

                check_border_element(tensor, id, border_mode, border_value, num_elements, num_mismatches);
            }

            // Right border
            for(int x = tensor.shape()[0]; x < static_cast<int>(tensor.shape()[0]) + static_cast<int>(border_size.right); ++x)
            {
                id.set(0, x);

                check_border_element(tensor, id, border_mode, border_value, num_elements, num_mismatches);
            }
        }
    }

    if(num_elements > 0)
    {
        const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches << "%) mismatched");
        ARM_COMPUTE_EXPECT_EQUAL(num_mismatches, 0, framework::LogLevel::ERRORS);
    }
}

void validate(std::vector<unsigned int> classified_labels, std::vector<unsigned int> expected_labels)
{
    ARM_COMPUTE_EXPECT_EQUAL(classified_labels.size(), expected_labels.size(), framework::LogLevel::ERRORS);

    int64_t   num_mismatches = 0;
    const int num_elements   = std::min(classified_labels.size(), expected_labels.size());

    for(int i = 0; i < num_elements; ++i)
    {
        if(classified_labels[i] != expected_labels[i])
        {
            ++num_mismatches;
            ARM_COMPUTE_EXPECT_EQUAL(classified_labels[i], expected_labels[i], framework::LogLevel::DEBUG);
        }
    }

    if(num_elements > 0)
    {
        const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches << "%) mismatched");
        ARM_COMPUTE_EXPECT_EQUAL(num_mismatches, 0, framework::LogLevel::ERRORS);
    }
}
} // namespace validation
} // namespace test
} // namespace arm_compute
