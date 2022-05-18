/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef TESTS_VALIDATION_CL_DYNAMICFUSION_UTILS
#define TESTS_VALIDATION_CL_DYNAMICFUSION_UTILS

#include "tests/AssetsLibrary.h"
#include "utils/Utils.h"

#include <chrono>
#include <limits>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace utils
{
/** A pair of macros which measures the wall clock time, and records it into a map measurement_map with name clock_name
 *
 */
#define TICK(clock_name) \
    auto clock_name##_tick = std::chrono::high_resolution_clock::now();
#define TOCK(clock_name, measurement_map)                                               \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>(clock_name##_tock - clock_name##_tick);
#define TOCK_AVG(clock_name, measurement_map, num_iterations)                           \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>((clock_name##_tock - clock_name##_tick) / (num_iterations));

template <typename T, typename U>
void fill(U &&tensor, int seed, AssetsLibrary *library)
{
    static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
    using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

    DistributionType distribution{ T(-1.0f), T(1.0f) };
    library->fill(tensor, distribution, seed);

    // Fill border with infinity in order to check the presence of NaN values (i.e. inf * 0)
    DistributionType distribution_inf{ T(std::numeric_limits<float>::infinity()), T(std::numeric_limits<float>::infinity()) };
    library->fill_borders_with_garbage(tensor, distribution_inf, seed);
}
} // namespace utils
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif //TESTS_VALIDATION_CL_DYNAMICFUSION_UTILS