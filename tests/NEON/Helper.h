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
#ifndef __ARM_COMPUTE_TEST_NEON_HELPER_H__
#define __ARM_COMPUTE_TEST_NEON_HELPER_H__

#include "arm_compute/runtime/Array.h"
#include "tests/Globals.h"

#include <algorithm>
#include <array>
#include <vector>

namespace arm_compute
{
namespace test
{
template <typename T>
Array<T> create_array(const std::vector<T> &v)
{
    const size_t v_size = v.size();
    Array<T>     array(v_size);

    array.resize(v_size);
    std::copy(v.begin(), v.end(), array.buffer());

    return array;
}

template <typename D, typename T, typename... Ts>
void fill_tensors(D &&dist, std::initializer_list<int> seeds, T &&tensor, Ts &&... other_tensors)
{
    const std::array < T, 1 + sizeof...(Ts) > tensors{ { std::forward<T>(tensor), std::forward<Ts>(other_tensors)... } };
    std::vector<int> vs(seeds);
    ARM_COMPUTE_ERROR_ON(vs.size() != tensors.size());
    int k = 0;
    for(auto tp : tensors)
    {
        library->fill(Accessor(*tp), std::forward<D>(dist), vs[k++]);
    }
}

} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_NEON_HELPER_H__ */
