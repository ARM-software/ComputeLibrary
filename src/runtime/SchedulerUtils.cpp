/*
 * Copyright (c) 2020 Arm Limited.
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
#include "src/runtime/SchedulerUtils.h"

#include "arm_compute/core/Error.h"

#include <cmath>

namespace arm_compute
{
namespace scheduler_utils
{
#ifndef BARE_METAL
std::pair<unsigned, unsigned> split_2d(unsigned max_threads, std::size_t m, std::size_t n)
{
    /*
     * We want the same ratio of threads in M & N to the ratio of m and n problem size
     *
     * Therefore:    mt/nt == m/n    where mt*nt == max_threads
     *
     *             max_threads/nt = mt    &    (max_threads/nt) * (m/n) = nt
     *          nt^2 = max_threads * (m/n)
     *          nt = sqrt( max_threads * (m/n) )
     */
    //ratio of m to n in problem dimensions
    double ratio = m / static_cast<double>(n);

    // nt = sqrt(max_threads * (m / n) )
    const unsigned adjusted = std::round(
                                  std::sqrt(max_threads * ratio));

    //find the nearest factor of max_threads
    for(unsigned i = 0; i != adjusted; ++i)
    {
        //try down
        const unsigned adj_down = adjusted - i;
        if(max_threads % adj_down == 0)
        {
            return { adj_down, max_threads / adj_down };
        }

        //try up
        const unsigned adj_up = adjusted + i;
        if(max_threads % adj_up == 0)
        {
            return { adj_up, max_threads / adj_up };
        }
    }

    //we didn't find anything so lets bail out with maxes biased to the largest dimension
    if(m > n)
    {
        return { std::min<unsigned>(m, max_threads), 1 };
    }
    else
    {
        return { 1, std::min<unsigned>(n, max_threads) };
    }
}
#endif /* #ifndef BARE_METAL */
} // namespace scheduler_utils
} // namespace arm_compute
