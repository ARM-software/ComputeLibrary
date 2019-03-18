/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/utils/helpers/fft.h"

#include <numeric>

namespace arm_compute
{
namespace helpers
{
namespace fft
{
std::vector<unsigned int> decompose_stages(unsigned int N, const std::set<unsigned int> &supported_factors)
{
    std::vector<unsigned int> stages;
    unsigned int              res = N;

    // Early exit if no supported factors are provided
    if(supported_factors.empty())
    {
        return stages;
    }

    // Create reverse iterator (Start decomposing from the larger supported factors)
    auto rfactor_it = supported_factors.rbegin();

    // Decomposition step
    while(res != 0)
    {
        const unsigned int factor = *rfactor_it;
        if(0 == (res % factor) && res >= factor)
        {
            stages.push_back(factor);
            res /= factor;
        }
        else
        {
            ++rfactor_it;
            if(rfactor_it == supported_factors.rend())
            {
                if(res > 1)
                {
                    // Couldn't decompose with given factors
                    stages.clear();
                    return stages;
                }
                else
                {
                    res = 0;
                }
            }
        }
    }

    return stages;
}

std::vector<unsigned int> digit_reverse_indices(unsigned int N, const std::vector<unsigned int> &fft_stages)
{
    std::vector<unsigned int> idx_digit_reverse;

    // Early exit in case N and fft stages do not match
    const float stages_prod = std::accumulate(std::begin(fft_stages), std::end(fft_stages), 1, std::multiplies<unsigned int>());
    if(stages_prod != N)
    {
        return idx_digit_reverse;
    }

    // Resize digit reverse vector
    idx_digit_reverse.resize(N);

    // Get number of radix stages
    unsigned int n_stages = fft_stages.size();

    // Scan elements
    for(unsigned int n = 0; n < N; ++n)
    {
        unsigned int k  = n;
        unsigned int Nx = fft_stages[0];

        // Scan stages
        for(unsigned int s = 1; s < n_stages; ++s)
        {
            // radix of stage i-th
            unsigned int Ny = fft_stages[s];
            unsigned int Ni = Ny * Nx;

            // Update k index
            k = (k * Ny) % Ni + (k / Nx) % Ny + Ni * (k / Ni);

            // Update Nx
            Nx *= Ny;
        }

        // K is the index of digit-reverse
        idx_digit_reverse[n] = k;
    }

    return idx_digit_reverse;
}
} // namespace fft
} // namespace helpers
} // namespace arm_compute
