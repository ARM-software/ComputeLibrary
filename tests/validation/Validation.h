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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_H__
#define __ARM_COMPUTE_TEST_VALIDATION_H__

#include "arm_compute/core/FixedPoint.h"
#include "arm_compute/core/IArray.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"
#include "tests/IAccessor.h"
#include "tests/SimpleTensor.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Exceptions.h"
#include "utils/TypePrinter.h"

#include <iomanip>
#include <ios>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Class reprensenting an absolute tolerance value. */
template <typename T>
class AbsoluteTolerance
{
public:
    /** Underlying type. */
    using value_type = T;

    /* Default constructor.
     *
     * Initialises the tolerance to 0.
     */
    AbsoluteTolerance() = default;

    /** Constructor.
     *
     * @param[in] value Absolute tolerance value.
     */
    explicit constexpr AbsoluteTolerance(T value)
        : _value{ value }
    {
    }

    /** Implicit conversion to the underlying type. */
    constexpr operator T() const
    {
        return _value;
    }

private:
    T _value{ std::numeric_limits<T>::epsilon() };
};

/** Class reprensenting a relative tolerance value. */
template <typename T>
class RelativeTolerance
{
public:
    /** Underlying type. */
    using value_type = T;

    /* Default constructor.
     *
     * Initialises the tolerance to 0.
     */
    RelativeTolerance() = default;

    /** Constructor.
     *
     * @param[in] value Relative tolerance value.
     */
    explicit constexpr RelativeTolerance(value_type value)
        : _value{ value }
    {
    }

    /** Implicit conversion to the underlying type. */
    constexpr operator value_type() const
    {
        return _value;
    }

private:
    value_type _value{ std::numeric_limits<T>::epsilon() };
};

/** Print AbsoluteTolerance type. */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const AbsoluteTolerance<T> &tolerance)
{
    os << static_cast<typename AbsoluteTolerance<T>::value_type>(tolerance);

    return os;
}

/** Print RelativeTolerance type. */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const RelativeTolerance<T> &tolerance)
{
    os << static_cast<typename RelativeTolerance<T>::value_type>(tolerance);

    return os;
}

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

/** Validate padding.
 *
 * Padding on all sides has to be the same.
 */
void validate(const arm_compute::PaddingSize &padding, const arm_compute::PaddingSize &width_reference, const arm_compute::PaddingSize &height_reference);

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
template <typename T, typename U = AbsoluteTolerance<T>>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, U tolerance_value = U(), float tolerance_number = 0.f);

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
template <typename T, typename U = AbsoluteTolerance<T>>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const ValidRegion &valid_region, U tolerance_value = U(), float tolerance_number = 0.f);

/** Validate tensors with valid mask.
 *
 * - Dimensionality has to be the same.
 * - All values have to match.
 *
 * @note: wrap_range allows cases where reference tensor rounds up to the wrapping point, causing it to wrap around to
 * zero while the test tensor stays at wrapping point to pass. This may permit true erroneous cases (difference between
 * reference tensor and test tensor is multiple of wrap_range), but such errors would be detected by
 * other test cases.
 */
template <typename T, typename U = AbsoluteTolerance<T>>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const SimpleTensor<T> &valid_mask, U tolerance_value = U(), float tolerance_number = 0.f);

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
template <typename T, typename U = AbsoluteTolerance<T>>
bool validate(T target, T reference, U tolerance = AbsoluteTolerance<T>());

/** Validate key points. */
template <typename T, typename U, typename V = AbsoluteTolerance<float>>
void validate_keypoints(T target_first, T target_last, U reference_first, U reference_last, V tolerance = AbsoluteTolerance<float>(),
                        float allowed_missing_percentage = 5.f, float allowed_mismatch_percentage = 5.f);

template <typename T>
struct compare_base
{
    compare_base(typename T::value_type target, typename T::value_type reference, T tolerance = T(0))
        : _target{ target }, _reference{ reference }, _tolerance{ tolerance }
    {
    }

    typename T::value_type _target{};
    typename T::value_type _reference{};
    T                      _tolerance{};
};

template <typename T>
struct compare;

template <typename U>
struct compare<AbsoluteTolerance<U>> : public compare_base<AbsoluteTolerance<U>>
{
    using compare_base<AbsoluteTolerance<U>>::compare_base;

    operator bool() const
    {
        if(!support::cpp11::isfinite(this->_target) || !support::cpp11::isfinite(this->_reference))
        {
            return false;
        }
        else if(this->_target == this->_reference)
        {
            return true;
        }

        using comparison_type = typename std::conditional<std::is_integral<U>::value, int64_t, U>::type;

        const comparison_type abs_difference(std::abs(static_cast<comparison_type>(this->_target) - static_cast<comparison_type>(this->_reference)));

        return abs_difference <= static_cast<comparison_type>(this->_tolerance);
    }
};

template <typename U>
struct compare<RelativeTolerance<U>> : public compare_base<RelativeTolerance<U>>
{
    using compare_base<RelativeTolerance<U>>::compare_base;

    operator bool() const
    {
        if(!support::cpp11::isfinite(this->_target) || !support::cpp11::isfinite(this->_reference))
        {
            return false;
        }
        else if(this->_target == this->_reference)
        {
            return true;
        }

        const U epsilon = (std::is_same<half, typename std::remove_cv<U>::type>::value || (this->_reference == 0)) ? static_cast<U>(0.01) : static_cast<U>(1e-05);

        if(std::abs(static_cast<double>(this->_reference) - static_cast<double>(this->_target)) <= epsilon)
        {
            return true;
        }
        else
        {
            if(static_cast<double>(this->_reference) == 0.0f) // We have checked whether _reference and _target is closing. If _reference is 0 but not closed to _target, it should return false
            {
                return false;
            }

            const double relative_change = std::abs(static_cast<double>(this->_target) - static_cast<double>(this->_reference)) / this->_reference;

            return relative_change <= static_cast<U>(this->_tolerance);
        }
    }
};

template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, U tolerance_value, float tolerance_number)
{
    // Validate with valid region covering the entire shape
    validate(tensor, reference, shape_to_valid_region(tensor.shape()), tolerance_value, tolerance_number);
}

template <typename T, typename U, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void validate_wrap(const IAccessor &tensor, const SimpleTensor<T> &reference, U tolerance_value, float tolerance_number)
{
    // Validate with valid region covering the entire shape
    validate_wrap(tensor, reference, shape_to_valid_region(tensor.shape()), tolerance_value, tolerance_number);
}

template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const ValidRegion &valid_region, U tolerance_value, float tolerance_number)
{
    int64_t num_mismatches = 0;
    int64_t num_elements   = 0;

    ARM_COMPUTE_EXPECT_EQUAL(tensor.element_size(), reference.element_size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.data_type(), reference.data_type(), framework::LogLevel::ERRORS);

    if(reference.format() != Format::UNKNOWN)
    {
        ARM_COMPUTE_EXPECT_EQUAL(tensor.format(), reference.format(), framework::LogLevel::ERRORS);
    }

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

                if(!compare<U>(target_value, reference_value, tolerance_value))
                {
                    ARM_COMPUTE_TEST_INFO("id = " << id);
                    ARM_COMPUTE_TEST_INFO("channel = " << c);
                    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target_value));
                    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference_value));
                    ARM_COMPUTE_TEST_INFO("tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance_value)));
                    framework::ARM_COMPUTE_PRINT_INFO();

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

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches
                              << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
        ARM_COMPUTE_EXPECT(num_mismatches <= absolute_tolerance_number, framework::LogLevel::ERRORS);
    }
}

template <typename T, typename U, typename = typename std::enable_if<std::is_integral<T>::value>::type>
void validate_wrap(const IAccessor &tensor, const SimpleTensor<T> &reference, const ValidRegion &valid_region, U tolerance_value, float tolerance_number)
{
    int64_t num_mismatches = 0;
    int64_t num_elements   = 0;

    ARM_COMPUTE_EXPECT_EQUAL(tensor.element_size(), reference.element_size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.data_type(), reference.data_type(), framework::LogLevel::ERRORS);

    if(reference.format() != Format::UNKNOWN)
    {
        ARM_COMPUTE_EXPECT_EQUAL(tensor.format(), reference.format(), framework::LogLevel::ERRORS);
    }

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

                bool equal = compare<U>(target_value, reference_value, tolerance_value);

                // check for wrapping
                if(!equal)
                {
                    if(!support::cpp11::isfinite(target_value) || !support::cpp11::isfinite(reference_value))
                    {
                        equal = false;
                    }
                    else
                    {
                        using limits_type = typename std::make_unsigned<T>::type;

                        uint64_t max             = std::numeric_limits<limits_type>::max();
                        uint64_t abs_sum         = std::abs(static_cast<int64_t>(target_value)) + std::abs(static_cast<int64_t>(reference_value));
                        uint64_t wrap_difference = max - abs_sum;

                        equal = wrap_difference < static_cast<uint64_t>(tolerance_value);
                    }
                }

                if(!equal)
                {
                    ARM_COMPUTE_TEST_INFO("id = " << id);
                    ARM_COMPUTE_TEST_INFO("channel = " << c);
                    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target_value));
                    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference_value));
                    ARM_COMPUTE_TEST_INFO("wrap_tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance_value)));
                    framework::ARM_COMPUTE_PRINT_INFO();

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

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches
                              << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
        ARM_COMPUTE_EXPECT(num_mismatches <= absolute_tolerance_number, framework::LogLevel::ERRORS);
    }
}

/** Check which keypoints from [first1, last1) are missing in [first2, last2) */
template <typename T, typename U, typename V>
std::pair<int64_t, int64_t> compare_keypoints(T first1, T last1, U first2, U last2, V tolerance)
{
    int64_t num_missing    = 0;
    int64_t num_mismatches = 0;

    while(first1 != last1)
    {
        const auto point = std::find_if(first2, last2, [&](KeyPoint point)
        {
            return point.x == first1->x && point.y == first1->y;
        });

        if(point == last2)
        {
            ++num_missing;
            ARM_COMPUTE_TEST_INFO("Key point not found" << *first1)
            ARM_COMPUTE_TEST_INFO("keypoint1 = " << *first1)
        }
        else if(!validate(point->tracking_status, first1->tracking_status) || !validate(point->strength, first1->strength, tolerance) || !validate(point->scale, first1->scale)
                || !validate(point->orientation, first1->orientation) || !validate(point->error, first1->error))
        {
            ++num_mismatches;
            ARM_COMPUTE_TEST_INFO("Mismatching keypoint")
            ARM_COMPUTE_TEST_INFO("keypoint1 = " << *first1)
            ARM_COMPUTE_TEST_INFO("keypoint2 = " << *point)
        }

        ++first1;
    }

    return std::make_pair(num_missing, num_mismatches);
}

template <typename T, typename U, typename V>
void validate_keypoints(T target_first, T target_last, U reference_first, U reference_last, V tolerance,
                        float allowed_missing_percentage, float allowed_mismatch_percentage)
{
    const int64_t num_elements_target    = std::distance(target_first, target_last);
    const int64_t num_elements_reference = std::distance(reference_first, reference_last);

    int64_t num_missing    = 0;
    int64_t num_mismatches = 0;

    if(num_elements_reference > 0)
    {
        std::tie(num_missing, num_mismatches) = compare_keypoints(reference_first, reference_last, target_first, target_last, tolerance);

        const float percent_missing    = static_cast<float>(num_missing) / num_elements_reference * 100.f;
        const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements_reference * 100.f;

        ARM_COMPUTE_TEST_INFO(num_missing << " keypoints (" << std::fixed << std::setprecision(2) << percent_missing << "%) are missing in target");
        ARM_COMPUTE_EXPECT(percent_missing <= allowed_missing_percentage, framework::LogLevel::ERRORS);

        ARM_COMPUTE_TEST_INFO(num_mismatches << " keypoints (" << std::fixed << std::setprecision(2) << percent_mismatches << "%) mismatched");
        ARM_COMPUTE_EXPECT(percent_mismatches <= allowed_mismatch_percentage, framework::LogLevel::ERRORS);
    }

    if(num_elements_target > 0)
    {
        std::tie(num_missing, num_mismatches) = compare_keypoints(target_first, target_last, reference_first, reference_last, tolerance);

        const float percent_missing = static_cast<float>(num_missing) / num_elements_target * 100.f;

        ARM_COMPUTE_TEST_INFO(num_missing << " keypoints (" << std::fixed << std::setprecision(2) << percent_missing << "%) are not part of target");
        ARM_COMPUTE_EXPECT(percent_missing <= allowed_missing_percentage, framework::LogLevel::ERRORS);
    }
}

template <typename T, typename U>
void validate(const IAccessor &tensor, const SimpleTensor<T> &reference, const SimpleTensor<T> &valid_mask, U tolerance_value, float tolerance_number)
{
    int64_t num_mismatches = 0;
    int64_t num_elements   = 0;

    ARM_COMPUTE_EXPECT_EQUAL(tensor.element_size(), reference.element_size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(tensor.data_type(), reference.data_type(), framework::LogLevel::ERRORS);

    if(reference.format() != Format::UNKNOWN)
    {
        ARM_COMPUTE_EXPECT_EQUAL(tensor.format(), reference.format(), framework::LogLevel::ERRORS);
    }

    ARM_COMPUTE_EXPECT_EQUAL(tensor.num_channels(), reference.num_channels(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(compare_dimensions(tensor.shape(), reference.shape()), framework::LogLevel::ERRORS);

    const int min_elements = std::min(tensor.num_elements(), reference.num_elements());
    const int min_channels = std::min(tensor.num_channels(), reference.num_channels());

    // Iterate over all elements within valid region, e.g. U8, S16, RGB888, ...
    for(int element_idx = 0; element_idx < min_elements; ++element_idx)
    {
        const Coordinates id = index2coord(reference.shape(), element_idx);

        if(valid_mask[element_idx] == 1)
        {
            // Iterate over all channels within one element
            for(int c = 0; c < min_channels; ++c)
            {
                const T &target_value    = reinterpret_cast<const T *>(tensor(id))[c];
                const T &reference_value = reinterpret_cast<const T *>(reference(id))[c];

                if(!compare<U>(target_value, reference_value, tolerance_value))
                {
                    ARM_COMPUTE_TEST_INFO("id = " << id);
                    ARM_COMPUTE_TEST_INFO("channel = " << c);
                    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target_value));
                    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference_value));
                    ARM_COMPUTE_TEST_INFO("tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance_value)));
                    framework::ARM_COMPUTE_PRINT_INFO();

                    ++num_mismatches;
                }

                ++num_elements;
            }
        }
        else
        {
            ++num_elements;
        }
    }

    if(num_elements > 0)
    {
        const int64_t absolute_tolerance_number = tolerance_number * num_elements;
        const float   percent_mismatches        = static_cast<float>(num_mismatches) / num_elements * 100.f;

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches
                              << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
        ARM_COMPUTE_EXPECT(num_mismatches <= absolute_tolerance_number, framework::LogLevel::ERRORS);
    }
}

template <typename T, typename U>
bool validate(T target, T reference, U tolerance)
{
    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference));
    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target));
    ARM_COMPUTE_TEST_INFO("tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance)));

    const bool equal = compare<U>(target, reference, tolerance);

    ARM_COMPUTE_EXPECT(equal, framework::LogLevel::ERRORS);

    return equal;
}

template <typename T, typename U>
void validate_min_max_loc(const MinMaxLocationValues<T> &target, const MinMaxLocationValues<U> &reference)
{
    ARM_COMPUTE_EXPECT_EQUAL(target.min, reference.min, framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(target.max, reference.max, framework::LogLevel::ERRORS);

    ARM_COMPUTE_EXPECT_EQUAL(target.min_loc.size(), reference.min_loc.size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT_EQUAL(target.max_loc.size(), reference.max_loc.size(), framework::LogLevel::ERRORS);

    for(uint32_t i = 0; i < target.min_loc.size(); ++i)
    {
        const auto same_coords = std::find_if(reference.min_loc.begin(), reference.min_loc.end(), [&target, i](Coordinates2D coord)
        {
            return coord.x == target.min_loc.at(i).x && coord.y == target.min_loc.at(i).y;
        });

        ARM_COMPUTE_EXPECT(same_coords != reference.min_loc.end(), framework::LogLevel::ERRORS);
    }

    for(uint32_t i = 0; i < target.max_loc.size(); ++i)
    {
        const auto same_coords = std::find_if(reference.max_loc.begin(), reference.max_loc.end(), [&target, i](Coordinates2D coord)
        {
            return coord.x == target.max_loc.at(i).x && coord.y == target.max_loc.at(i).y;
        });

        ARM_COMPUTE_EXPECT(same_coords != reference.max_loc.end(), framework::LogLevel::ERRORS);
    }
}

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__ */
