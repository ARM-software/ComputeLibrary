/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "bias_adder.hpp"
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

    const GemmArgs     _args;

    const unsigned int _buffer_per_multi;

    unsigned int k_block=0;
    unsigned int n_block=0;

    const Toi *_B_pretransposed = nullptr;

public:
    GemvPretransposed(GemvPretransposed &) = delete;
    GemvPretransposed & operator= (GemvPretransposed &) = delete;

    GemvPretransposed(const GemmArgs &args)
                      : _args(args),
                        _buffer_per_multi(args._Ksize * roundup(args._Nsize, strategy::out_width())) {
        /* For now don't do any blocking. TODO: figure out if we should. */
        if (strategy::supports_accumulate() && args._cfg && args._cfg->inner_block_size) {
            k_block = args._cfg->inner_block_size;
        } else {
            k_block = args._Ksize;
        }

        if (args._cfg && args._cfg->outer_block_size) {
            n_block = args._cfg->outer_block_size;
        } else {
            n_block = args._Nsize;
        }
    }

    // Window is number of out_width blocks, times number of multis.
    ndrange_t get_window_size() const override {
        return { iceildiv(_args._Nsize, strategy::out_width()) * _args._nmulti };
    }

    // Actually execute the GEMV.
    void execute(const ndcoord_t &work_range, const ndcoord_t &, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_args._ci);

        const auto start = work_range.get_position(0);
        const auto end   = work_range.get_position_end(0);

        /* Break the window values down into multis of interest... */
        const unsigned int window_per_multi = iceildiv(_args._Nsize, strategy::out_width());
        const unsigned int multi_0    = start / window_per_multi;
        const unsigned int multi_end  = end   / window_per_multi;

        /* ... and figure out where we start and end in the first and last multi. */
        const unsigned int n_0   = (start - (multi_0 * window_per_multi)) * strategy::out_width();
        const unsigned int n_max = (end - (multi_end * window_per_multi)) * strategy::out_width();

        static_assert(std::is_same<Tr, Tri>::value, "GemvPretransposed: Result types must be the same.");

        for (unsigned int multi=multi_0; multi<=multi_end; multi++) {
            const unsigned int n_start = (multi==multi_0) ? n_0 : 0;
            const unsigned int n_end = (multi==multi_end) ? n_max : _args._Nsize;

            if (n_end <= n_start)
                continue;

            for (unsigned int k0=0; k0<_args._Ksize; k0+=k_block) {
                unsigned int kmax = std::min(k0 + k_block, _args._Ksize);

                for (unsigned int n=n_start; n<n_end; n+=n_block) {
                    unsigned int nmax = std::min(n + n_block, n_end);
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (kmax-k0) * (nmax-n));
#endif
                    strat.kernel(this->_Aptr + (multi * this->_A_multi_stride) + k0,
                                 _B_pretransposed + (multi * _buffer_per_multi) + (n * roundup(_args._Ksize, strategy::k_unroll())) + (k0 * strategy::out_width()),
                                 this->_Cptr + (multi * this->_C_multi_stride) + n,
                                 (nmax - n), (kmax-k0),
                                 this->_bias ? this->_bias + (multi * this->_bias_multi_stride) + n : nullptr,
                                 _args._act, (k0 != 0));
                }
            }
        }
    }

    /* Pretransposed interface implementation */
    bool B_is_pretransposed() const override {
        return true;
    }

    bool B_pretranspose_required() const override {
        /* Transpose is required if _B_pretransposed is still nullptr */
        return (_B_pretransposed == nullptr);
    }

    size_t get_B_pretransposed_array_size() const override {
        return _buffer_per_multi * _args._nmulti * sizeof(To);
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride) override {
        Toi *B_buffer = reinterpret_cast<Toi *>(buffer);
        strategy strat(_args._ci);

        for (unsigned int multi=0; multi<_args._nmulti; multi++) {
            strat.transforms.PrepareB(B_buffer + (multi * _buffer_per_multi), B + (multi * B_multi_stride), ldb, 0, _args._Nsize, 0, _args._Ksize);
        }

        _B_pretransposed = B_buffer;
    }

    void set_pretransposed_B_data(void *buffer) override {
        _B_pretransposed = reinterpret_cast<Toi *>(buffer);
    }
};

} // namespace arm_gemm
