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
#include "Validation.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/IAccessor.h"
#include "tests/RawTensor.h"
#include "tests/TypePrinter.h"
#include "tests/Utils.h"
#include "tests/validation/half.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>

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
            return *reinterpret_cast<const half_float::half *>(ptr);
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

bool is_equal(double target, double ref, double max_absolute_error = std::numeric_limits<double>::epsilon(), double max_relative_error = 0.0001f)
{
    if(!std::isfinite(target) || !std::isfinite(ref))
    {
        return false;
    }

    // No need further check if they are equal
    if(ref == target)
    {
        return true;
    }

    // Need this check for the situation when the two values close to zero but have different sign
    if(std::abs(std::abs(ref) - std::abs(target)) <= max_absolute_error)
    {
        return true;
    }

    double relative_error = 0;

    if(std::abs(target) > std::abs(ref))
    {
        relative_error = std::abs((target - ref) / target);
    }
    else
    {
        relative_error = std::abs((ref - target) / ref);
    }

    return relative_error <= max_relative_error;
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
        border_id.set(1, 0);
        border_value = tensor(border_id);
    }

    // Iterate over all channels within one element
    for(int channel = 0; channel < tensor.num_channels(); ++channel)
    {
        const size_t channel_offset = channel * channel_size;
        const double target         = get_double_data(ptr + channel_offset, tensor.data_type());
        const double ref            = get_double_data(static_cast<const uint8_t *>(border_value) + channel_offset, tensor.data_type());
        const bool   equal          = is_equal(target, ref);

        BOOST_TEST_INFO("id = " << id);
        BOOST_TEST_INFO("channel = " << channel);
        BOOST_TEST_INFO("reference = " << std::setprecision(5) << ref);
        BOOST_TEST_INFO("target = " << std::setprecision(5) << target);
        BOOST_TEST_WARN(equal);

        if(!equal)
        {
            ++num_mismatches;
        }

        ++num_elements;
    }
}

void check_single_element(const Coordinates &id, const IAccessor &tensor, const RawTensor &reference, float tolerance_value,
                          uint64_t wrap_range, int min_channels, size_t channel_size, int64_t &num_mismatches, int64_t &num_elements)
{
    const auto ptr     = static_cast<const uint8_t *>(tensor(id));
    const auto ref_ptr = static_cast<const uint8_t *>(reference(id));

    // Iterate over all channels within one element
    for(int channel = 0; channel < min_channels; ++channel)
    {
        const size_t channel_offset = channel * channel_size;
        const double target         = get_double_data(ptr + channel_offset, reference.data_type());
        const double ref            = get_double_data(ref_ptr + channel_offset, reference.data_type());
        bool         equal          = is_equal(target, ref, tolerance_value);

        if(wrap_range != 0 && !equal)
        {
            equal = is_equal(target, ref, wrap_range - tolerance_value);
        }

        if(!equal)
        {
            BOOST_TEST_INFO("id = " << id);
            BOOST_TEST_INFO("channel = " << channel);
            BOOST_TEST_INFO("reference = " << std::setprecision(5) << ref);
            BOOST_TEST_INFO("target = " << std::setprecision(5) << target);
            BOOST_TEST_WARN(equal);

            ++num_mismatches;
        }
        ++num_elements;
    }
}
} // namespace

void validate(const arm_compute::ValidRegion &region, const arm_compute::ValidRegion &reference)
{
    BOOST_TEST(region.anchor.num_dimensions() == reference.anchor.num_dimensions());
    BOOST_TEST(region.shape.num_dimensions() == reference.shape.num_dimensions());

    for(unsigned int d = 0; d < region.anchor.num_dimensions(); ++d)
    {
        BOOST_TEST(region.anchor[d] == reference.anchor[d]);
    }

    for(unsigned int d = 0; d < region.shape.num_dimensions(); ++d)
    {
        BOOST_TEST(region.shape[d] == reference.shape[d]);
    }
}

void validate(const arm_compute::PaddingSize &padding, const arm_compute::PaddingSize &reference)
{
    BOOST_TEST(padding.top == reference.top);
    BOOST_TEST(padding.right == reference.right);
    BOOST_TEST(padding.bottom == reference.bottom);
    BOOST_TEST(padding.left == reference.left);
}

void validate(const IAccessor &tensor, const RawTensor &reference, float tolerance_value, float tolerance_number, uint64_t wrap_range)
{
    // Validate with valid region covering the entire shape
    validate(tensor, reference, shape_to_valid_region(tensor.shape()), tolerance_value, tolerance_number, wrap_range);
}

void validate(const IAccessor &tensor, const RawTensor &reference, const ValidRegion &valid_region, float tolerance_value, float tolerance_number, uint64_t wrap_range)
{
    int64_t num_mismatches = 0;
    int64_t num_elements   = 0;

    BOOST_TEST(tensor.element_size() == reference.element_size());
    BOOST_TEST(tensor.format() == reference.format());
    BOOST_TEST(tensor.data_type() == reference.data_type());
    BOOST_TEST(tensor.num_channels() == reference.num_channels());
    BOOST_TEST(compare_dimensions(tensor.shape(), reference.shape()));

    const int    min_elements = std::min(tensor.num_elements(), reference.num_elements());
    const int    min_channels = std::min(tensor.num_channels(), reference.num_channels());
    const size_t channel_size = element_size_from_data_type(reference.data_type());

    // Iterate over all elements within valid region, e.g. U8, S16, RGB888, ...
    for(int element_idx = 0; element_idx < min_elements; ++element_idx)
    {
        const Coordinates id = index2coord(reference.shape(), element_idx);
        if(is_in_valid_region(valid_region, id))
        {
            check_single_element(id, tensor, reference, tolerance_value, wrap_range, min_channels, channel_size, num_mismatches, num_elements);
        }
    }

    const int64_t absolute_tolerance_number = tolerance_number * num_elements;
    const float   percent_mismatches        = static_cast<float>(num_mismatches) / num_elements * 100.f;

    BOOST_TEST(num_mismatches <= absolute_tolerance_number,
               num_mismatches << " values (" << std::setprecision(2) << percent_mismatches
               << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
}

void validate(const IAccessor &tensor, const void *reference_value)
{
    BOOST_TEST_REQUIRE((reference_value != nullptr));

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
            const double ref            = get_double_data(reference_value, tensor.data_type());
            const bool   equal          = is_equal(target, ref);

            BOOST_TEST_INFO("id = " << id);
            BOOST_TEST_INFO("channel = " << channel);
            BOOST_TEST_INFO("reference = " << std::setprecision(5) << ref);
            BOOST_TEST_INFO("target = " << std::setprecision(5) << target);
            BOOST_TEST_WARN(equal);

            if(!equal)
            {
                ++num_mismatches;
            }

            ++num_elements;
        }
    }

    const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;

    BOOST_TEST(num_mismatches == 0,
               num_mismatches << " values (" << std::setprecision(2) << percent_mismatches << "%) mismatched");
}

void validate(const IAccessor &tensor, BorderSize border_size, const BorderMode &border_mode, const void *border_value)
{
    if(border_mode == BorderMode::UNDEFINED)
    {
        return;
    }
    else if(border_mode == BorderMode::CONSTANT)
    {
        BOOST_TEST((border_value != nullptr));
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

    const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements * 100.f;

    BOOST_TEST(num_mismatches == 0,
               num_mismatches << " values (" << std::setprecision(2) << percent_mismatches << "%) mismatched");
}

void validate(std::vector<unsigned int> classified_labels, std::vector<unsigned int> expected_labels)
{
    ARM_COMPUTE_UNUSED(classified_labels);
    ARM_COMPUTE_UNUSED(expected_labels);
    BOOST_TEST(expected_labels.size() != 0);
    BOOST_TEST(classified_labels.size() == expected_labels.size());

    for(unsigned int i = 0; i < expected_labels.size(); ++i)
    {
        BOOST_TEST(classified_labels[i] == expected_labels[i]);
    }
}

void validate(float target, float ref, float tolerance_abs_error, float tolerance_relative_error)
{
    const bool equal = is_equal(target, ref, tolerance_abs_error, tolerance_relative_error);

    BOOST_TEST_INFO("reference = " << std::setprecision(5) << ref);
    BOOST_TEST_INFO("target = " << std::setprecision(5) << target);
    BOOST_TEST(equal);
}
} // namespace validation
} // namespace test
} // namespace arm_compute
