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
class RelativeTolerance
{
public:
    /** Underlying type. */
    using value_type = double;

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
    value_type _value{ 0 };
};

/** Print AbsoluteTolerance type. */
template <typename T>
inline ::std::ostream &operator<<(::std::ostream &os, const AbsoluteTolerance<T> &tolerance)
{
    os << static_cast<typename AbsoluteTolerance<T>::value_type>(tolerance);

    return os;
}

/** Print RelativeTolerance type. */
inline ::std::ostream &operator<<(::std::ostream &os, const RelativeTolerance &tolerance)
{
    os << static_cast<typename RelativeTolerance::value_type>(tolerance);

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
template <typename T, typename U>
void validate(T target, T reference, U tolerance = AbsoluteTolerance<T>());

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

template <typename T, typename U>
struct compare;

template <typename U>
struct compare<AbsoluteTolerance<U>, U> : public compare_base<AbsoluteTolerance<U>>
{
    using compare_base<AbsoluteTolerance<U>>::compare_base;

    operator bool()
    {
        if(!std::isfinite(this->_target) || !std::isfinite(this->_reference))
        {
            return false;
        }
        else if(this->_target == this->_reference)
        {
            return true;
        }

        return static_cast<U>(std::abs(this->_target - this->_reference)) <= static_cast<U>(this->_tolerance);
    }
};

template <typename U>
struct compare<RelativeTolerance, U> : public compare_base<RelativeTolerance>
{
    using compare_base<RelativeTolerance>::compare_base;

    operator bool()
    {
        if(!std::isfinite(_target) || !std::isfinite(_reference))
        {
            return false;
        }
        else if(_target == _reference)
        {
            return true;
        }

        const double relative_change = std::abs(static_cast<double>(_target - _reference)) / _reference;

        return relative_change <= _tolerance;
    }
};

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

                if(!compare<U, typename U::value_type>(target_value, reference_value, tolerance_value))
                {
                    ARM_COMPUTE_TEST_INFO("id = " << id);
                    ARM_COMPUTE_TEST_INFO("channel = " << c);
                    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target_value));
                    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference_value));
                    ARM_COMPUTE_TEST_INFO("tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance_value)));
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

        ARM_COMPUTE_TEST_INFO(num_mismatches << " values (" << std::fixed << std::setprecision(2) << percent_mismatches
                              << "%) mismatched (maximum tolerated " << std::setprecision(2) << tolerance_number << "%)");
        ARM_COMPUTE_EXPECT(num_mismatches <= absolute_tolerance_number, framework::LogLevel::ERRORS);
    }
}

template <typename T, typename U>
void validate(T target, T reference, U tolerance)
{
    ARM_COMPUTE_TEST_INFO("reference = " << std::setprecision(5) << framework::make_printable(reference));
    ARM_COMPUTE_TEST_INFO("target = " << std::setprecision(5) << framework::make_printable(target));
    ARM_COMPUTE_TEST_INFO("tolerance = " << std::setprecision(5) << framework::make_printable(static_cast<typename U::value_type>(tolerance)));
    ARM_COMPUTE_EXPECT((compare<U, typename U::value_type>(target, reference, tolerance)), framework::LogLevel::ERRORS);
}
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__ */
