/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_MISC_RANDOM_H
#define ARM_COMPUTE_MISC_RANDOM_H

#include "arm_compute/core/Error.h"
#include "utils/Utils.h"

#include <random>
#include <type_traits>

namespace arm_compute
{
namespace utils
{
namespace random
{
/** Uniform distribution within a given number of sub-ranges
 *
 * @tparam T Distribution primitive type
 */
template <typename T>
class RangedUniformDistribution
{
public:
    static constexpr bool is_fp_16bit = std::is_same<T, half>::value || std::is_same<T, bfloat16>::value;
    static constexpr bool is_integral = std::is_integral<T>::value && !is_fp_16bit;

    using fp_dist     = typename std::conditional<is_fp_16bit, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;
    using DT          = typename std::conditional<is_integral, std::uniform_int_distribution<T>, fp_dist>::type;
    using result_type = T;
    using range_pair  = std::pair<result_type, result_type>;

    /** Constructor
     *
     * @param[in] low            lowest value in the range (inclusive)
     * @param[in] high           highest value in the range (inclusive for uniform_int_distribution, exclusive for uniform_real_distribution)
     * @param[in] exclude_ranges Ranges to exclude from the generator
     */
    RangedUniformDistribution(result_type low, result_type high, const std::vector<range_pair> &exclude_ranges)
        : _distributions(), _selector()
    {
        result_type clow = low;
        for(const auto &erange : exclude_ranges)
        {
            result_type epsilon = is_integral ? result_type(1) : result_type(std::numeric_limits<T>::epsilon());

            ARM_COMPUTE_ERROR_ON(clow > erange.first || clow >= erange.second);

            _distributions.emplace_back(DT(clow, erange.first - epsilon));
            clow = erange.second + epsilon;
        }
        ARM_COMPUTE_ERROR_ON(clow > high);
        _distributions.emplace_back(DT(clow, high));
        _selector = std::uniform_int_distribution<uint32_t>(0, _distributions.size() - 1);
    }
    /** Generate random number
     *
     * @tparam URNG Random number generator object type
     *
     * @param[in] g A uniform random number generator object, used as the source of randomness.
     *
     * @return A new random number.
     */
    template <class URNG>
    result_type operator()(URNG &g)
    {
        unsigned int rand_select = _selector(g);
        return _distributions[rand_select](g);
    }

private:
    std::vector<DT>                         _distributions;
    std::uniform_int_distribution<uint32_t> _selector;
};
} // namespace random
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_MISC_RANDOM_H */
