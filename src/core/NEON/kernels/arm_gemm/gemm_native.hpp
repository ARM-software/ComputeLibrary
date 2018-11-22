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
#include "transform.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

namespace arm_gemm {

// Implementation of the GemmCommon abstract class.
//
// This is implementation is for native GEMM with no transposition.
//
// By default the source data is used in-place, but if type conversion is
// needed we need to allocate working space (CURRENTLY NOT IMPLEMENTED).

template<typename strategy, typename To, typename Tr>
class GemmNative : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmultis;

    Tr _beta;

    const CPUInfo * const _ci;

    unsigned int k_block=0;
    unsigned int n_block=0;

    unsigned int window_per_batch() const {
        return iceildiv(_Msize, strategy::out_height());
    }

    unsigned int window_per_multi() const {
        return window_per_batch() * _nbatches;
    }

public:
    GemmNative(GemmNative &) = delete;
    GemmNative & operator= (GemmNative &) = delete;

    GemmNative(const CPUInfo *ci, const unsigned int M, const unsigned int N, const unsigned int K, const unsigned int nbatches, const unsigned int nmultis, const Tr beta) :
        _Msize(M), _Nsize(N), _Ksize(K), _nbatches(nbatches), _nmultis(nmultis), _beta(beta), _ci(ci) {
        /* For now don't do any blocking. TODO: figure out if we should. */
        k_block = K;
        n_block = N;
    }

    // Window is amount per multi multiplied by total number of multis.
    unsigned int get_window_size() const override {
        return window_per_multi() * _nmultis;
    }

    // Actually execute the GEMM.
    void execute(unsigned int start, unsigned int end, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        static_assert(std::is_same<To, Toi>::value, "gemm_native: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemm_native: Result types must be the same.");

        /* Compute starting point based on 'start' */
        unsigned int multi     = start / window_per_multi();
        unsigned int multi_pos = start % window_per_multi();

        unsigned int batch     = multi_pos / window_per_batch();
        unsigned int batch_pos = multi_pos % window_per_batch();

        unsigned int y0        = batch_pos * strategy::out_height();

        for (unsigned int pos=start; pos<end; pos++) {
            const unsigned int ymax = std::min(y0 + strategy::out_height(), _Msize);
#ifdef CYCLE_PROFILING
            auto p = prof.ScopedProfiler(PROFILE_KERNEL, (ymax-y0) * _Nsize * _Ksize);
#endif

            strat.kernel(this->_Aptr + (multi * this->_A_multi_stride) + (batch * this->_A_batch_stride) + (y0 * this->_lda), this->_lda,
                         this->_Bptr + (multi * this->_B_multi_stride), this->_ldb,
                         this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (y0 * this->_ldc), this->_ldc,
                         _beta, (ymax-y0), _Nsize, _Ksize);

            /* Advance to next item */
            y0 += strategy::out_height();

            /* Check for batch/multi overflow */
            if (y0 >= _Msize) {
                y0=0;
                batch++;
                if (batch == _nbatches) {
                    batch=0;
                    multi++;
                }
            }
        }
    }
};

} // namespace arm_gemm
