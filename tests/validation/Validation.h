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
#ifndef __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__
#define __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Array.h"

#include "boost_wrapper.h"

#include <vector>

namespace arm_compute
{
class Tensor;

namespace test
{
class RawTensor;
class IAccessor;

namespace validation
{
template <typename T>
boost::test_tools::predicate_result compare_dimensions(const Dimensions<T> &dimensions1, const Dimensions<T> &dimensions2)
{
    if(dimensions1.num_dimensions() != dimensions2.num_dimensions())
    {
        boost::test_tools::predicate_result result(false);
        result.message() << "Different dimensionality [" << dimensions1.num_dimensions() << "!=" << dimensions2.num_dimensions() << "]";
        return result;
    }

    for(unsigned int i = 0; i < dimensions1.num_dimensions(); ++i)
    {
        if(dimensions1[i] != dimensions2[i])
        {
            boost::test_tools::predicate_result result(false);
            result.message() << "Mismatch in dimension " << i << " [" << dimensions1[i] << "!=" << dimensions2[i] << "]";
            return result;
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
void validate(const IAccessor &tensor, const RawTensor &reference, float tolerance_value = 0.f, float tolerance_number = 0.f, uint64_t wrap_range = 0);

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
void validate(const IAccessor &tensor, const RawTensor &reference, const ValidRegion &valid_region, float tolerance_value = 0.f, float tolerance_number = 0.f, uint64_t wrap_range = 0);

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
void validate(float target, float ref, float tolerance_abs_error = std::numeric_limits<float>::epsilon(), float tolerance_relative_error = 0.0001f);

/** Validate min max location.
 *
 * - All values should match
 */
template <typename T>
void validate_min_max_loc(T min, T ref_min, T max, T ref_max,
                          IArray<Coordinates2D> &min_loc, IArray<Coordinates2D> &ref_min_loc, IArray<Coordinates2D> &max_loc, IArray<Coordinates2D> &ref_max_loc,
                          uint32_t min_count, uint32_t ref_min_count, uint32_t max_count, uint32_t ref_max_count)
{
    BOOST_TEST(min == ref_min);
    BOOST_TEST(max == ref_max);

    BOOST_TEST(min_count == min_loc.num_values());
    BOOST_TEST(max_count == max_loc.num_values());
    BOOST_TEST(ref_min_count == ref_min_loc.num_values());
    BOOST_TEST(ref_max_count == ref_max_loc.num_values());

    BOOST_TEST(min_count == ref_min_count);
    BOOST_TEST(max_count == ref_max_count);

    for(uint32_t i = 0; i < min_count; i++)
    {
        Coordinates2D *same_coords = std::find_if(ref_min_loc.buffer(), ref_min_loc.buffer() + min_count, [&min_loc, i](Coordinates2D coord)
        {
            return coord.x == min_loc.at(i).x && coord.y == min_loc.at(i).y;
        });

        BOOST_TEST(same_coords != ref_min_loc.buffer() + min_count);
    }

    for(uint32_t i = 0; i < max_count; i++)
    {
        Coordinates2D *same_coords = std::find_if(ref_max_loc.buffer(), ref_max_loc.buffer() + max_count, [&max_loc, i](Coordinates2D coord)
        {
            return coord.x == max_loc.at(i).x && coord.y == max_loc.at(i).y;
        });

        BOOST_TEST(same_coords != ref_max_loc.buffer() + max_count);
    }
}
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_REFERENCE_VALIDATION_H__ */
