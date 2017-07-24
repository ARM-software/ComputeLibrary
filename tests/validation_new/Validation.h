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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_H__
#define __ARM_COMPUTE_TEST_VALIDATION_H__

#include "SimpleTensor.h"
#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/Types.h"
#include "framework/Asserts.h"
#include "framework/Exceptions.h"
#include "tests/IAccessor.h"
#include "tests/TypePrinter.h"
#include "tests/Utils.h"

#include <iomanip>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename T>
bool compare_dimensions(const Dimensions<T> &dimensions1, const Dimensions<T> &dimensions2)
{
    if(dimensions1.num_dimensions() != dimensions2.num_dimensions())
    {
        return false;
    }

    for(unsigned int i = 0; i < dimensions1.num_dimensions(); ++i)
    {
        if(dimensions1[i] != dimensions2[i])
        {
            return false;
        }
    }

    return true;
}

/** Validate valid regions.
 *
 * - Dimensionality has to be the same.
 * - Anchors have to match.
 * - Shapes have to match.
 */
void validate(const arm_compute::ValidRegion &region, const arm_compute::ValidRegion &reference);

/** Validate padding.
 *
 * Padding on all sides has to be the same.
 */
void validate(const arm_compute::PaddingSize &padding, const arm_compute::PaddingSize &reference);

/** Validate tensors.
 *
 * - Dimensionality has to be the same.
 * - All values have to match.
 *
 * @note: wrap_range allows cases where reference tensor rounds up to the wrapping point, causing it to wrap around to
 * zero while the test tensor stays at wrapping point to pass. This may permit true erroneous cases (difference between
 * reference tensor and test tensor is multiple of wrap_range), but such errors would be detected by
 * other test cases.
 */
template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, U tolerance_value = 0, float tolerance_number = 0.f);

/** Validate tensors with valid region.
 *
 * - Dimensionality has to be the same.
 * - All values have to match.
 *
 * @note: wrap_range allows cases where reference tensor rounds up to the wrapping point, causing it to wrap around to
 * zero while the test tensor stays at wrapping point to pass. This may permit true erroneous cases (difference between
 * reference tensor and test tensor is multiple of wrap_range), but such errors would be detected by
 * other test cases.
 */
template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const ValidRegion &valid_region, U tolerance_value = 0, float tolerance_number = 0.f);

/** Validate tensors against constant value.
 *
 * - All values have to match.
 */
void validate(const IAccessor &tensor, const void *reference_value);

/** Validate border against a constant value.
 *
 * - All border values have to match the specified value if mode is CONSTANT.
 * - All border values have to be replicated if mode is REPLICATE.
 * - Nothing is validated for mode UNDEFINED.
 */
void validate(const IAccessor &tensor, BorderSize border_size, const BorderMode &border_mode, const void *border_value);

/** Validate classified labels against expected ones.
 *
 * - All values should match
 */
void validate(std::vector<unsigned int> classified_labels, std::vector<unsigned int> expected_labels);

/** Validate float value.
 *
 * - All values should match
 */
template <typename T, typename U = T>
void validate(T target, T ref, U tolerance_abs_error = std::numeric_limits<T>::epsilon(), double tolerance_relative_error = 0.0001f);

template <typename T, typename U = T>
bool is_equal(T target, T ref, U max_absolute_error = std::numeric_limits<T>::epsilon(), double max_relative_error = 0.0001f)
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
        relative_error = std::abs(static_cast<double>(target - ref) / target);
    }
    else
    {
        relative_error = std::abs(static_cast<double>(ref - target) / ref);
    }

    return relative_error <= max_relative_error;
}

template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, U tolerance_value, float tolerance_number)
{
    // Validate with valid region covering the entire shape
    validate(tensor, reference, shape_to_valid_region(tensor.shape()), tolerance_value, tolerance_number);
}

template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const ValidRegion &valid_region, U tolerance_value, float tolerance_number)
{
    int64_t num_mismatches = 0;
    int64_t num_elements   = 0;

    ARM_COMPUTE_EXPECT_EQUAL(tensor.element_size(), reference.element_size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.format(), reference.format(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.data_type(), reference.data_type(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.num_channels(), reference.num_channels(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(compare_dimensions(tensor.shape(), reference.shape()), framework::LogLevel::ERRORS);

    const int min_elements = std::min(tensor.num_elements(), reference.num_elements());
    const int min_channels = std::min(tensor.num_channels(), reference.num_channels());

    // Iterate over all elements within valid region, e.g. U8, S16, RGB888, ...
    for(int element_idx = 0; element_idx < min_elements; ++element_idx)
    {
        const Coordinates id = index2coord(reference.shape(), element_idx);

        if(is_in_valid_region(valid_region, id))
        {
            // Iterate over all channels within one element
            for(int c = 0; c < min_channels; ++c)
            {
                const T &target_value    = reinterpret_cast<const T *>(tensor(id))[c];
                const T &reference_value = reinterpret_cast<const T *>(reference(id))[c];

                if(!is_equal(target_value, reference_value, tolerance_value))
                {
                    ARM_COMPUTE_TEST_INFO("id = " << id);
                    ARM_COMPUTE_TEST_INFO("channel = " << c);
                    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << target_value);
                    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << reference_value);
                    ARM_COMPUTE_EXPECT_EQUAL(target_value, reference_value, framework::LogLevel::DEBUG);

                    ++num_mismatches;
                }

                ++num_elements;
            }
        }
    }

    if(num_elements > 0)
    {
        const int64_t absolute_tolerance_number = tolerance_number * num_elements;
        const float   percent_mismatches        = static_cast<float>(num_mismatches) / num_elements * 100.f;

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::setprecision(2) << percent_mismatches
                              << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
        ARM_COMPUTE_EXPECT(num_mismatches <= absolute_tolerance_number, framework::LogLevel::ERRORS);
    }
}

template <typename T, typename U>
void validate(T target, T ref, U tolerance_abs_error, double tolerance_relative_error)
{
    const bool equal = is_equal(target, ref, tolerance_abs_error, tolerance_relative_error);

    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << ref);
    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << target);
    ARM_COMPUTE_EXPECT(equal, framework::LogLevel::ERRORS);
}
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__ */
