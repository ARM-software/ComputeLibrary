/*
 * Copyright (c) 2017-2019 Arm Limited.
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

#include "ndrange.hpp"

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

    const Tr _beta;

    const CPUInfo * const _ci;

    const unsigned int _k_block;
    const unsigned int _n_block;

    const NDRange<4> _window_range;

    static unsigned int compute_k_block(const GemmArgs<Tr> &args) {
        return args._Ksize;
    }

    static unsigned int compute_n_block(const GemmArgs<Tr> &args) {
        if ((args._cfg != nullptr) && args._cfg->outer_block_size > 0) {
            return args._cfg->outer_block_size;
        } else {
            return args._Nsize;
        }
    }

public:
    GemmNative(GemmNative &) = delete;
    GemmNative & operator= (GemmNative &) = delete;

    GemmNative(const GemmArgs<Tr> &args)
               : _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
                 _nbatches(args._nbatches), _nmultis(args._nmulti),
                 _beta(args._beta), _ci(args._ci),
                 _k_block(compute_k_block(args)), _n_block(compute_n_block(args)),
                 _window_range(iceildiv(_Msize, strategy::out_height()), _nbatches, iceildiv(_Nsize, _n_block), _nmultis) { }

    // Window is amount per multi multiplied by total number of multis.
    unsigned int get_window_size() const override {
        return _window_range.total_size();
    }

    // Native GEMMs can always be dynamically scheduled (whether requested or not)
    bool supports_dynamic_scheduling() const override {
        return true;
    }

    // Actually execute the GEMM.
    void execute(unsigned int start, unsigned int end, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        static_assert(std::is_same<To, Toi>::value, "gemm_native: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemm_native: Result types must be the same.");

        auto p = _window_range.iterator(start, end);

        if (p.done()) {
            return;
        }

        do {
            unsigned int y0    = p.dim(0) * strategy::out_height();
            unsigned int ymax  = std::min(p.dim0_max() * strategy::out_height(), _Msize);
            unsigned int batch = p.dim(1);
            unsigned int n0    = p.dim(2) * _n_block;
            unsigned int nmax  = std::min(n0 + _n_block, _Nsize);
            unsigned int multi = p.dim(3);

#ifdef CYCLE_PROFILING
            auto p = prof.ScopedProfiler(PROFILE_KERNEL, (ymax-y0) * (nmax - n0) * _Ksize);
#endif

            strat.kernel(this->_Aptr + (multi * this->_A_multi_stride) + (batch * this->_A_batch_stride) + (y0 * this->_lda), this->_lda,
                         this->_Bptr + (multi * this->_B_multi_stride) + n0, this->_ldb,
                         this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (y0 * this->_ldc) + n0, this->_ldc,
                         _beta, (ymax-y0), (nmax - n0), _Ksize);
        } while (p.next_dim1());
    }
};

} // namespace arm_gemm
