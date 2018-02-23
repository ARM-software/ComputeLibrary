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
#pragma once

#include <stdio.h>

#include "arm_gemm.hpp"

#include "mergeresults.hpp"
#include "profiler.hpp"
#include "transform.hpp"

namespace arm_gemm
{
// Implementation of the GemmCommon abstract class.
//
// This is implementation is for native GEMM with no transposition.
//
// By default the source data is used in-place, but if type conversion is
// needed we need to allocate working space (CURRENTLY NOT IMPLEMENTED).

template <typename strategy, typename To, typename Tr>
class GemmNative : public GemmCommon<To, Tr>
{
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type  Tri;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    Tr _beta;

    const CPUInfo *const _ci;

    unsigned int k_block = 0;
    unsigned int n_block = 0;

public:
    GemmNative(GemmNative &) = delete;
    GemmNative &operator=(GemmNative &) = delete;

    GemmNative(const CPUInfo *ci, const unsigned int M, const unsigned int N, const unsigned int K, const Tr beta)
        : _Msize(M), _Nsize(N), _Ksize(K), _beta(beta), _ci(ci)
    {
        /* For now don't do any blocking. TODO: figure out if we should. */
        k_block = K;
        n_block = N;
    }

    // Window is number of out_height blocks
    unsigned int get_window_size() const override
    {
        return iceildiv(_Msize, strategy::out_height);
    }

    // Actually execute the GEMM.
    void execute(unsigned int start, unsigned int end, int) override
    {
        profiler prof;
        strategy strat(_ci);

        unsigned int M_start = start * strategy::out_height;
        unsigned int M_end   = std::min(end * strategy::out_height, _Msize);

        static_assert(std::is_same<To, Toi>::value, "gemm_native: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemm_native: Result types must be the same.");

        for(unsigned int y0 = M_start; y0 < M_end; y0 += strategy::out_height)
        {
            unsigned int ymax = std::min(y0 + strategy::out_height, M_end);

            prof(PROFILE_KERNEL, (ymax - y0) * _Nsize * _Ksize, [&](void)
            {
                strat.kernel(this->_Aptr + (y0 * this->_lda), this->_lda, this->_Bptr, this->_ldb, this->_Cptr + (y0 * this->_ldc), this->_ldc, _beta, (ymax - y0), _Nsize, _Ksize);
            });
        }
    }
};

} // namespace arm_gemm
