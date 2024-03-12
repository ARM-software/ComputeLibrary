/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_UTILS_HELPERS_FFT_H
#define ARM_COMPUTE_UTILS_HELPERS_FFT_H

#include <set>
#include <vector>

namespace arm_compute
{
namespace helpers
{
namespace fft
{
/** Decompose a given 1D input size using the provided supported factors.
 *
 * @param[in] N                 Input size to be decomposed.
 * @param[in] supported_factors Supported factors that can be used for decomposition.
 *
 * @return A vector with the stages of the decomposition. Will be empty if decomposition failed.
 */
std::vector<unsigned int> decompose_stages(unsigned int N, const std::set<unsigned int> &supported_factors);
/** Calculate digit reverse index vector given fft size and the decomposed stages
 *
 * @param N          Input size to calculate digit reverse for
 * @param fft_stages A vector with the FFT decomposed stages
 *
 * @return A vector with the digit reverse indices. Will be empty if it failed.
 */
std::vector<unsigned int> digit_reverse_indices(unsigned int N, const std::vector<unsigned int> &fft_stages);
} // namespace fft
} // namespace helpers
} // namespace arm_compute
#endif /* ARM_COMPUTE_UTILS_HELPERS_FFT_H */
