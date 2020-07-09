/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "Floor.h"

#include "tests/validation/Helpers.h"

#include <cmath>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> floor_layer(const SimpleTensor<T> &src)
{
    // Create reference
    SimpleTensor<T> dst{ src.shape(), src.data_type() };

    // Compute reference
#if defined(_OPENMP)
    #pragma omp parallel for
#endif /* _OPENMP */
    for(int i = 0; i < src.num_elements(); ++i)
    {
        dst[i] = std::floor(src[i]);
    }

    return dst;
}

template SimpleTensor<half> floor_layer(const SimpleTensor<half> &src);
template SimpleTensor<float> floor_layer(const SimpleTensor<float> &src);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
