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
#include "EqualizeHistogram.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> equalize_histogram(const SimpleTensor<T> &src)
{
    const size_t num_bins = 256; // 0-255 inclusive

    std::vector<T>        lut(num_bins);
    std::vector<uint32_t> hist(num_bins);
    std::vector<uint32_t> cd(num_bins); // cumulative distribution

    SimpleTensor<T> dst(src.shape(), src.data_type());

    // Create the histogram
    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        hist[src[element_idx]]++;
    }

    // Calculate cumulative distribution
    std::partial_sum(hist.begin(), hist.end(), cd.begin());

    // Get the number of pixels that have the lowest non-zero value
    const uint32_t cd_min = *std::find_if(hist.begin(), hist.end(), [](const uint32_t &x)
    {
        return x > 0;
    });

    const size_t total_num_pixels = cd.back();

    // Single color - create linear distribution
    if(total_num_pixels == cd_min)
    {
        std::iota(lut.begin(), lut.end(), 0);
    }
    else
    {
        const float diff = total_num_pixels - 1;

        for(size_t i = 0; i < num_bins; ++i)
        {
            lut[i] = lround((cd[i] - cd_min) / diff * 255.f);
        }
    }

    // Fill output tensor with equalized values
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = lut[src[i]];
    }

    return dst;
}

template SimpleTensor<uint8_t> equalize_histogram(const SimpleTensor<uint8_t> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
