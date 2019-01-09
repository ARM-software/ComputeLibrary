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
#pragma once

#include <assert.h>

#include <algorithm>

#include "arm_gemm.hpp"
#include "utils.hpp"

#include "mergeresults.hpp"
#include "transform.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

namespace arm_gemm {

// Implementation of the GemmCommon abstract class.
template<typename strategy, typename To, typename Tr>
class GemmHybrid : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    /* const properties set by constructor */
    const CPUInfo * const _ci;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmulti;

    const bool _trB;

    const Tr _beta;

    /* Blocking info */
    unsigned int _k_block=0;
    unsigned int _x_block=0;
    unsigned int _Mround=0;

    /* Pretransposed buffer. */
    const Toi *_B_transposed=nullptr;

    unsigned int _B_per_multi = 0;

    /* We will need to walk through the blocks of B in a few contexts, so
     * factor that out.  */
    class blockwalker {
    private:
        /* Size loops, etc. based on our parent's configuration */
        const GemmHybrid<strategy, To, Tr> &_parent;

        /* K, X and multi parameters for current iteration. */
        unsigned int _k0=0, _x0=0;

        unsigned int _index=0;
        bool _done=false;
        bool _newkblock=true;

    public:
        blockwalker(const GemmHybrid<strategy, To, Tr> &parent) : _parent(parent) { }

        unsigned int xmax() {
            return std::min(_x0 + _parent._x_block, _parent._Nsize);
        }

        unsigned int kmax() {
            return std::min(_k0 + _parent._k_block, _parent._Ksize);
        }

        /* Advance to the next block, return false at the end. */
        bool advance(void) {
            if (_done) {
                return false;
            }

            _newkblock=false;
            _x0 += _parent._x_block;
            if (_x0 >= _parent._Nsize) {
                _x0=0;
                _k0 += _parent._k_block;
                if (_k0 >= _parent._Ksize) {
                    _done=true;
                    return false;
                }
                _newkblock=true;
            }
            _index++;

            return true;
        }

        unsigned int k0(void) { return _k0; }
        unsigned int x0(void) { return _x0; }
        unsigned int index(void) { return _index; }
        bool done(void) { return _done; }
        bool newkblock(void) { return _newkblock; }
    };


public:
    GemmHybrid(GemmHybrid &) = delete;
    GemmHybrid & operator= (GemmHybrid &) = delete;

    /* Constructor */
    GemmHybrid(const GemmArgs<Tr> &args)
              : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize), _nbatches(args._nbatches), 
                _nmulti(args._nmulti), _trB(args._trB), _beta(args._beta) {
        const unsigned int L1_size = _ci->get_L1_cache_size();
        const unsigned int L2_size = _ci->get_L2_cache_size();

        _B_per_multi = (iceildiv(_Nsize, strategy::out_width()) * strategy::out_width()) *
                       (iceildiv(_Ksize, strategy::k_unroll()) * strategy::k_unroll());

        // Work out blocking parameters, or override from config.

        if (args._cfg && args._cfg->inner_block_size) {
            _k_block = args._cfg->inner_block_size;
        } else {
            // k_block: Find out how much of the larger array can be loaded into half the cache.
            // This should account for associative caches.
            _k_block = (L1_size / 2) / (sizeof(Toi) * (std::max(strategy::out_width(), strategy::out_height())));

            // Needs to be (at least a single) multiple of the K unroll level.
            _k_block /= strategy::k_unroll();
            _k_block = std::max(_k_block, 1U) * strategy::k_unroll();

            // Now tune to presented problem size; this is how many blocks we need.
            int num_k_blocks = iceildiv(_Ksize, _k_block);

            // So divide the space equally into that many blocks.
            _k_block = iceildiv(_Ksize, num_k_blocks);

            // And round UP to the K unroll level required.
            _k_block = iceildiv(_k_block, strategy::k_unroll());
            _k_block *= strategy::k_unroll();
        }

        if (args._cfg && args._cfg->outer_block_size) {
            _x_block = args._cfg->outer_block_size;
        } else {
            // x_block: Work out how many rows (of length k_block) will fit in the L2
            // Don't allocate more than 90% of the L2 to allow for overheads, and subtract off the L1 contents.
            _x_block = (((L2_size * 9) / 10) - (_k_block * sizeof(Toi) * (strategy::out_width() + strategy::out_height()))) /
                      (sizeof(Toi) * _k_block);

            // Needs to be (at least a single) multiple of the kernel output width.
            _x_block /= strategy::out_width();
            _x_block = std::max(_x_block, 1U) * strategy::out_width();

            // And tune to the presented problem size.
            int num_x_blocks = iceildiv(_Nsize, _x_block);
            _x_block = iceildiv(_Nsize, num_x_blocks);

            _x_block = iceildiv(_x_block, strategy::out_width());
            _x_block *= strategy::out_width();
        }

        // Work out the rounded size of M - needed for some buffers.
        _Mround = iceildiv(_Msize, strategy::out_height());
        _Mround *= strategy::out_height();
    }

    // Interface implementation - Compulsory functions

    // Window size: Only the last thread should do a ragged block, so dole
    // out work in units of out_height.  Factor batches and multi into the
    // window too.
    unsigned int get_window_size() const override {
        // _Mround is a multiple of out_height by definition.
        return (_Mround / strategy::out_height()) * _nbatches * _nmulti;
    }

    // Execute
    void execute(unsigned int start, unsigned int end, int threadid) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        /* Make sure we've been set up correctly. */
        assert(_B_transposed);

        const unsigned int window_per_batch = iceildiv(_Msize, strategy::out_height());
        const unsigned int window_per_multi = window_per_batch * _nbatches;

        const unsigned int first_multi = start / window_per_multi;
        const unsigned int last_multi  = end / window_per_multi;

        const unsigned int first_batch = (start - (first_multi * window_per_multi)) / window_per_batch;
        const unsigned int last_batch  = (end - (last_multi * window_per_multi)) / window_per_batch;

        const unsigned int first_row = ((start - (first_multi * window_per_multi)) % window_per_batch) * strategy::out_height();
        const unsigned int last_row  = ((end   - (last_multi  * window_per_multi)) % window_per_batch) * strategy::out_height();

        static_assert(std::is_same<To, Toi>::value, "gemm_native: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemm_native: Result types must be the same.");

        for (unsigned int multi = first_multi; multi <= last_multi; multi++) {
            const unsigned int batch_0   = (multi == first_multi) ? first_batch : 0;
            const unsigned int batch_max = (multi == last_multi)  ? last_batch : (_nbatches - 1);

            const Toi *b_panel = _B_transposed + (multi * _B_per_multi);

            for (blockwalker current(*this); !current.done(); current.advance()) {
                int kern_k = iceildiv(current.kmax() - current.k0(), strategy::k_unroll());
                kern_k *= strat.k_unroll();

                int bblocks = iceildiv(current.xmax() - current.x0(), strategy::out_width());

                for (unsigned int batch = batch_0; batch <= batch_max; batch++) {
                    const unsigned int m_start = ((multi == first_multi) && (batch == first_batch)) ? first_row : 0;
                    const unsigned int m_end   = ((multi == last_multi)  && (batch == last_batch) ) ? last_row  : _Msize;
#ifdef CYCLE_PROFILING
                    auto p = prof.ScopedProfiler(PROFILE_KERNEL, (m_end - m_start) * kern_k * bblocks * strategy::out_width());
#endif

                    strat.kernel(this->_Aptr + (multi * this->_A_multi_stride) + (batch * this->_A_batch_stride) + (m_start * this->_lda) + current.k0(), this->_lda,
                                 b_panel,
                                 this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (m_start * this->_ldc) + current.x0(), this->_ldc,
                                 (current.k0() == 0) ? _beta : static_cast<Tr>(1),
                                 (m_end - m_start), (current.xmax() - current.x0()), kern_k);
                }

                b_panel += (bblocks * strat.out_width() * kern_k);
            }
        }
    }

    // Interface implementation - pretransposed
    bool B_is_pretransposed() const override {
        return true;
    }

    bool B_pretranspose_required() const override {
        return (_B_transposed==nullptr);
    }

    size_t get_B_pretransposed_array_size() const override {
        return _B_per_multi * _nmulti * sizeof(Toi);
    }

    void pretranspose_B_array(void *in_buffer, const To *B, const int ldb, const int B_multi_stride) override {
        Toi *buffer = reinterpret_cast<Toi *>(in_buffer);
        _B_transposed = buffer;
        strategy strat(_ci);

        for (unsigned int multi=0; multi < _nmulti; multi++) {
            blockwalker current(*this);

            do {
                /* Figure out the size of each block. */
                size_t x_size = (current.xmax() - current.x0());
                size_t k_size = (current.kmax() - current.k0());

                /* Round sizes up as needed. */
                x_size = iceildiv(x_size, strategy::out_width());
                x_size *= strategy::out_width();

                k_size = iceildiv(k_size, strategy::k_unroll());
                k_size *= strategy::k_unroll();

                strat.transforms.PrepareB(
                               buffer, B + (multi * B_multi_stride), ldb,
                               current.x0(), current.xmax(), current.k0(), current.kmax(), _trB);

                buffer += (x_size * k_size);
            } while (current.advance());
        }
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        _B_transposed = reinterpret_cast<Toi *>(in_buffer);
    }
};

} // namespace arm_gemm
