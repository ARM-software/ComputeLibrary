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
// This is implementation is for GEMV with pretransposition.
//
// batches are not supported as a batched GEMV makes no sense (can be converted to a GEMM).
template<typename strategy, typename To, typename Tr>
class GemvPretransposed : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nmultis;

    const bool _trB;

    const Tr _beta;

    const CPUInfo * const _ci;

    const unsigned int _buffer_per_multi;

    unsigned int m_block=0;
    unsigned int n_block=0;

    const Toi *_A_pretransposed = nullptr;

public:
    GemvPretransposed(GemvPretransposed &) = delete;
    GemvPretransposed & operator= (GemvPretransposed &) = delete;

    GemvPretransposed(const GemmArgs<Tr> &args)
            : _Nsize(args._Nsize), _Ksize(args._Ksize), _nmultis(args._nmulti), _trB(args._trB), _beta(args._beta), _ci(args._ci),
              _buffer_per_multi(_Ksize * iceildiv(_Nsize, strategy::A_interleave()) * strategy::A_interleave()) {
        /* For now don't do any blocking. TODO: figure out if we should. */
        if (args._cfg && args._cfg->inner_block_size) {
            m_block = args._cfg->inner_block_size;
        } else {
            m_block = _Ksize;
        }

        if (args._cfg && args._cfg->outer_block_size) {
            n_block = args._cfg->outer_block_size;
        } else {
            n_block = _Nsize;
        }
    }

    // Window is number of out_width blocks, times number of multis.
    unsigned int get_window_size() const override {
        return iceildiv(_Nsize, strategy::out_width()) * _nmultis;
    }

    // Actually execute the GEMV.
    void execute(unsigned int start, unsigned int end, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        /* Break the window values down into multis of interest... */
        const unsigned int window_per_multi = iceildiv(_Nsize, strategy::out_width());
        const unsigned int multi_0    = start / window_per_multi;
        const unsigned int multi_end  = end   / window_per_multi;

        /* ... and figure out where we start and end in the first and last multi. */
        const unsigned int n_0   = (start - (multi_0 * window_per_multi)) * strategy::out_width();
        const unsigned int n_max = (end - (multi_end * window_per_multi)) * strategy::out_width();

        static_assert(std::is_same<Tr, Tri>::value, "GemvPretransposed: Result types must be the same.");

        for (unsigned int multi=multi_0; multi<=multi_end; multi++) {
            const unsigned int n_start = (multi==multi_0) ? n_0 : 0;
            const unsigned int n_end = (multi==multi_end) ? n_max : _Nsize;

            if (n_end <= n_start)
                continue;

            for (unsigned int m0=0; m0<_Ksize; m0+=m_block) {
                unsigned int mmax = std::min(m0 + m_block, _Ksize);

                for (unsigned int n=n_start; n<n_end; n+=n_block) {
                    unsigned int nmax = std::min(n + n_block, n_end);
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (mmax-m0) * (nmax-n));
#endif
                    /* This assumes that the underlying call was a GEMM with M=1; for the N=1 case we would have to pick up this->_Bptr below instead */
                    strat.kernel(_A_pretransposed + (multi * _buffer_per_multi) + (n * _Ksize) + (m0 * strategy::A_interleave()),
                                 (_Ksize * strategy::A_interleave()),
                                 this->_Aptr + (multi * this->_A_multi_stride) + m0,
                                 this->_Cptr + (multi * this->_C_multi_stride) + n,
                                 _beta, (mmax-m0), (nmax-n));
                }
            }
        }
    }

    /* Pretransposed interface implementation */
    bool B_is_pretransposed() const override {
        return true;
    }

    bool B_pretranspose_required() const override {
        /* Transpose is required if _A_pretransposed is still nullptr */
        return (_A_pretransposed == nullptr);
    }

    size_t get_B_pretransposed_array_size() const override {
        return _buffer_per_multi * _nmultis * sizeof(To);
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride) override {
        Toi *A_buffer = reinterpret_cast<Toi *>(buffer);

        for (unsigned int multi=0; multi<_nmultis; multi++) {
            /* Reverse sense here as we are dealing with B rather than A.  So if
             * strategy::A_transpose is false and _trB is false, we still
             * transpose.  */
            if (_trB ^ strategy::A_transpose()) {
                Transform<strategy::A_interleave(), strategy::A_block(), false>(A_buffer + (multi * _buffer_per_multi), B + (multi * B_multi_stride), ldb, 0, _Nsize, 0, _Ksize);
            } else {
                Transform<strategy::A_interleave(), strategy::A_block(), true>(A_buffer + (multi * _buffer_per_multi), B + (multi * B_multi_stride), ldb, 0, _Nsize, 0, _Ksize);
            }
        }

        _A_pretransposed = A_buffer;
    }

    void set_pretransposed_B_data(void *buffer) override {
        _A_pretransposed = reinterpret_cast<Toi *>(buffer);
    }
};

} // namespace arm_gemm
