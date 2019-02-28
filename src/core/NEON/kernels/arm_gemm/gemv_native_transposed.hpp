/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "transform.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

namespace arm_gemm {

// Implementation of the GemmCommon abstract class.
//
// This is implementation is for a "native" (no-transform) GEMV with a
// transposed matrix.
//
// As a native operation the source data is used in-place, so the internal
// and external operand/result types must match.
template<typename strategy, typename To, typename Tr>
class GemvNativeTransposed : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nmultis;

    const Tr _beta;

    const CPUInfo * const _ci;

    unsigned int m_block=0;
    unsigned int n_block=0;

public:
    GemvNativeTransposed(GemvNativeTransposed &) = delete;
    GemvNativeTransposed & operator= (GemvNativeTransposed &) = delete;

    GemvNativeTransposed(const GemmArgs<Tr> &args)
            : _Nsize(args._Nsize), _Ksize(args._Ksize), _nmultis(args._nmulti), _beta(args._beta), _ci(args._ci) {
        /* For now don't do any blocking. TODO: figure out if we should. */
        m_block = _Ksize;
        n_block = _Nsize;
    }

    // Window is number of out_width blocks times number of multis.
    unsigned int get_window_size() const override {
        return iceildiv(_Nsize, strategy::out_width()) * _nmultis;
    }

    // Actually execute the GEMV.
    void execute(unsigned int start, unsigned int end, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        const unsigned int window_per_multi = iceildiv(_Nsize, strategy::out_width());
        const unsigned int multi_0   = start / window_per_multi;
        const unsigned int multi_end = end   / window_per_multi;

        const unsigned int n_0   = (start - (multi_0 * window_per_multi)) * strategy::out_width();
        const unsigned int n_max = (end - (multi_end * window_per_multi)) * strategy::out_width();

        static_assert(std::is_same<To, Toi>::value, "gemv_transposed: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemv_transposed: Result types must be the same.");

        for (unsigned int multi=multi_0; multi<=multi_end; multi++) {
            const unsigned int n_start = (multi==multi_0) ? n_0 : 0;
            const unsigned int n_end = (multi==multi_end) ? n_max : _Nsize;

            if (n_end <= n_start)
                continue;

            for (unsigned int m0=0; m0<_Ksize; m0+=m_block) {
                unsigned int mmax = std::min(m0 + m_block, _Ksize);

                for (unsigned int n0=n_start; n0<n_end; n0+=n_block) {
                    unsigned int nmax = std::min(n0 + n_block, n_end);
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (mmax-m0) * (nmax-n0));
#endif
                    strat.kernel(this->_Bptr + (multi * this->_B_multi_stride) + (m0 * this->_ldb) + n0,
                                 this->_Aptr + (multi * this->_A_multi_stride) + m0,
                                 this->_Cptr + (multi * this->_C_multi_stride) + n0,
                                 _beta, this->_ldb, (mmax-m0), (nmax-n0));
                }
            }
        }
    }
};

} // namespace arm_gemm
