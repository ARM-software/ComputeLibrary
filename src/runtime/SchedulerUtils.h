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
#ifndef SRC_COMPUTE_SCHEDULER_UTILS_H
#define SRC_COMPUTE_SCHEDULER_UTILS_H

#include <cstddef>
#include <utility>

namespace arm_compute
{
namespace scheduler_utils
{
/** Given two dimensions and a maximum number of threads to utilise, calculate the best
 * combination of threads that fit in (multiplied together) max_threads.
 *
 * This algorithm assumes that work in either of the dimensions is equally difficult
 * to compute
 *
 * @returns [m_nthreads, n_nthreads] A pair of the threads that should be used in each dimension
 */
std::pair<unsigned, unsigned> split_2d(unsigned max_threads, std::size_t m, std::size_t n);
} // namespace scheduler_utils
} // namespace arm_compute
#endif /* SRC_COMPUTE_SCHEDULER_UTILS_H */
