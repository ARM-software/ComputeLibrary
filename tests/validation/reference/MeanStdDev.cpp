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
#include "MeanStdDev.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
std::pair<float, float> mean_and_standard_deviation(const SimpleTensor<T> &in)
{
    const int num_elements = in.num_elements();

    // Calculate mean
    float mean = std::accumulate(in.data(), in.data() + num_elements, 0.f) / num_elements;

    // Calculate standard deviation
    float std_dev = std::accumulate(in.data(), in.data() + num_elements, 0.f, [&mean](float a, float b)
    {
        return a + (mean - b) * (mean - b);
    });

    std_dev = std::sqrt(std_dev / num_elements);

    return std::make_pair(mean, std_dev);
}

template std::pair<float, float> mean_and_standard_deviation(const SimpleTensor<uint8_t> &in);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
