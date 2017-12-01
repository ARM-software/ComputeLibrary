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
#include "Histogram.h"

#include "Utils.h"
#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<uint32_t> histogram(const SimpleTensor<T> &src, size_t num_bins, int32_t offset, uint32_t range)
{
    SimpleTensor<uint32_t> dst(TensorShape(num_bins), DataType::U32);

    // Clear the distribution
    for(size_t element_idx = 0; element_idx < num_bins; ++element_idx)
    {
        dst[element_idx] = 0;
    }

    // Create the histogram
    for(int element_idx = 0; element_idx < src.num_elements(); ++element_idx)
    {
        if((offset <= src[element_idx]) && (src[element_idx] < (offset + range)))
        {
            const int index = (src[element_idx] - offset) * num_bins / range;
            dst[index]++;
        }
    }

    return dst;
}

template SimpleTensor<uint32_t> histogram(const SimpleTensor<uint8_t> &src, size_t num_bins, int32_t offset, uint32_t range);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
