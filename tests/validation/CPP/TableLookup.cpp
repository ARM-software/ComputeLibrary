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
#include "TableLookup.h"

#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> table_lookup(const SimpleTensor<T> &src, const std::map<T, T> &rawlut)
{
    SimpleTensor<T> result(src.shape(), src.data_type());

    for(int i = 0; i < src.num_elements(); ++i)
    {
        result[i] = rawlut.at(src[i]);
    }

    return result;
}

template SimpleTensor<uint8_t> table_lookup(const SimpleTensor<uint8_t> &src, const std::map<uint8_t, uint8_t> &rawlut);
template SimpleTensor<int16_t> table_lookup(const SimpleTensor<int16_t> &src, const std::map<int16_t, int16_t> &rawlut);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
