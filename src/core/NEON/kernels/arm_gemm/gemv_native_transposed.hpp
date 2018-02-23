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
// This is implementation is for a "native" (no-transform) GEMV with a
// transposed matrix.
//
// As a native operation the source data is used in-place, so the internal
// and external operand/result types must match.
template <typename strategy, typename To, typename Tr>
class GemvNativeTransposed : public GemmCommon<To, Tr>
{
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type  Tri;

    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const Tr _alpha;

    const CPUInfo *const _ci;

    unsigned int m_block = 0;
    unsigned int n_block = 0;

public:
    GemvNativeTransposed(GemvNativeTransposed &) = delete;
    GemvNativeTransposed &operator=(GemvNativeTransposed &) = delete;

    GemvNativeTransposed(const CPUInfo *ci, const unsigned int N, const unsigned int K, const Tr alpha)
        : _Nsize(N), _Ksize(K), _alpha(alpha), _ci(ci)
    {
        /* For now don't do any blocking. TODO: figure out if we should. */
        m_block = K;
        n_block = N;
    }

    // Window is number of out_width blocks.
    unsigned int get_window_size() const override
    {
        return iceildiv(_Nsize, strategy::out_width);
    }

    // Actually execute the GEMV.
    void execute(unsigned int start, unsigned int end, int) override
    {
        profiler prof;
        strategy strat(_ci);

        unsigned int N_start = start * strategy::out_width;
        unsigned int N_end   = std::min(end * strategy::out_width, _Nsize);

        static_assert(std::is_same<To, Toi>::value, "gemv_transposed: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemv_transposed: Result types must be the same.");

        for(unsigned int m0 = 0; m0 < _Ksize; m0 += m_block)
        {
            unsigned int mmax = std::min(m0 + m_block, _Ksize);

            for(unsigned int n0 = N_start; n0 < N_end; n0 += n_block)
            {
                unsigned int nmax = std::min(n0 + n_block, N_end);

                prof(PROFILE_KERNEL, ((mmax - m0) * (nmax - n0)), [&](void)
                {
                    strat.kernel(this->_Bptr + (m0 * this->_ldb) + n0, this->_Aptr + m0, this->_Cptr + n0,
                                 _alpha, this->_ldb, (mmax - m0), (nmax - n0));
                });
            }
        }
    }
};

} // namespace arm_gemm
