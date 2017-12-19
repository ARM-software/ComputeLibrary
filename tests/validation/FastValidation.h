/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_FAST_VALIDATION_H__
#define __ARM_COMPUTE_TEST_FAST_VALIDATION_H__

#include "Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Check which keypoints from [first1, last1) are missing in [first2, last2) */
template <typename T, typename U, typename V>
std::pair<int64_t, int64_t> fast_compare_keypoints(T first1, T last1, U first2, U last2, V tolerance, bool check_mismatches = true)
{
    /* Keypoint (x,y) should have similar strength (within tolerance) and other properties in both reference and target */
    const auto compare_props_eq = [&](const KeyPoint & lhs, const KeyPoint & rhs)
    {
        return compare<V>(lhs.strength, rhs.strength, tolerance)
               && lhs.tracking_status == rhs.tracking_status
               && lhs.scale == rhs.scale
               && lhs.orientation == rhs.orientation
               && lhs.error == rhs.error;
    };

    /* Used to sort KeyPoints by coordinates (x, y) */
    const auto compare_coords_lt = [](const KeyPoint & lhs, const KeyPoint & rhs)
    {
        return std::tie(lhs.x, lhs.y) < std::tie(rhs.x, rhs.y);
    };

    std::sort(first1, last1, compare_coords_lt);
    std::sort(first2, last2, compare_coords_lt);

    if(check_mismatches)
    {
        std::cout << "ref count = " << std::distance(first1, last1) << " \ttarget count = " << std::distance(first2, last2) << std::endl;
    }

    int64_t num_missing    = 0;
    int64_t num_mismatches = 0;
    bool    rest_missing   = false;

    while(first1 != last1)
    {
        if(first2 == last2)
        {
            // num_missing += std::distance(first1, last1);
            rest_missing = true;
            ARM_COMPUTE_TEST_INFO("All key points from (" << first1->x << "," << first1->y << ") onwards not found");
            break;
        }

        if(compare_coords_lt(*first1, *first2))
        {
            ++num_missing;
            ARM_COMPUTE_TEST_INFO("Key point not found");
            ARM_COMPUTE_TEST_INFO("keypoint1 = " << *first1++);
        }
        else
        {
            if(!compare_coords_lt(*first2, *first1)) // Equal coordinates
            {
                if(check_mismatches && !compare_props_eq(*first1, *first2)) // Check other properties
                {
                    ++num_mismatches;
                    ARM_COMPUTE_TEST_INFO("Mismatching keypoint");
                    ARM_COMPUTE_TEST_INFO("keypoint1 [ref] = " << *first1);
                    ARM_COMPUTE_TEST_INFO("keypoint2 [tgt] = " << *first2);
                }
                ++first1;
            }
            ++first2;
        }
    }

    if(rest_missing)
    {
        while(first1 != last1)
        {
            ++num_missing;
            ARM_COMPUTE_TEST_INFO("Key point not found");
            ARM_COMPUTE_TEST_INFO("keypoint1 = " << *first1++);
        }
    }

    return std::make_pair(num_missing, num_mismatches);
}

template <typename T, typename U, typename V>
void fast_validate_keypoints(T target_first, T target_last, U reference_first, U reference_last, V tolerance,
                             float allowed_missing_percentage, float allowed_mismatch_percentage)
{
    const int64_t num_elements_target    = std::distance(target_first, target_last);
    const int64_t num_elements_reference = std::distance(reference_first, reference_last);

    int64_t num_missing    = 0;
    int64_t num_mismatches = 0;

    if(num_elements_reference > 0)
    {
        std::tie(num_missing, num_mismatches) = fast_compare_keypoints(reference_first, reference_last, target_first, target_last, tolerance);

        const float percent_missing    = static_cast<float>(num_missing) / num_elements_reference * 100.f;
        const float percent_mismatches = static_cast<float>(num_mismatches) / num_elements_reference * 100.f;

        ARM_COMPUTE_TEST_INFO(num_missing << " keypoints (" << std::fixed << std::setprecision(2) << percent_missing << "%) in ref are missing from target");
        ARM_COMPUTE_EXPECT(percent_missing <= allowed_missing_percentage, framework::LogLevel::ERRORS);

        ARM_COMPUTE_TEST_INFO(num_mismatches << " keypoints (" << std::fixed << std::setprecision(2) << percent_mismatches << "%) mismatched");
        ARM_COMPUTE_EXPECT(percent_mismatches <= allowed_mismatch_percentage, framework::LogLevel::ERRORS);

        std::cout << "Mismatched keypoints: " << num_mismatches << "/" << num_elements_reference << " = " << std::fixed << std::setprecision(2) << percent_mismatches
                  << "% \tMax allowed: " << allowed_mismatch_percentage << "%" << std::endl;
        std::cout << "Missing (not in tgt): " << num_missing << "/" << num_elements_reference << " = " << std::fixed << std::setprecision(2) << percent_missing
                  << "% \tMax allowed: " << allowed_missing_percentage << "%" << std::endl;
    }

    if(num_elements_target > 0)
    {
        // Note: no need to check for mismatches a second time (last argument is 'false')
        std::tie(num_missing, num_mismatches) = fast_compare_keypoints(target_first, target_last, reference_first, reference_last, tolerance, false);

        const float percent_missing = static_cast<float>(num_missing) / num_elements_target * 100.f;

        ARM_COMPUTE_TEST_INFO(num_missing << " keypoints (" << std::fixed << std::setprecision(2) << percent_missing << "%) in target are missing from ref");
        ARM_COMPUTE_EXPECT(percent_missing <= allowed_missing_percentage, framework::LogLevel::ERRORS);

        std::cout << "Missing (not in ref): " << num_missing << "/" << num_elements_target << " = " << std::fixed << std::setprecision(2) << percent_missing
                  << "% \tMax allowed: " << allowed_missing_percentage << "%\n"
                  << std::endl;
    }
}

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_FAST_VALIDATION_H__ */
