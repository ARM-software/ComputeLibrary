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
#include <assert.h>

#include <algorithm>

#include "arm_gemm.hpp"
#include "utils.hpp"

#include "buffer_manager.hpp"
#include "mergeresults.hpp"
#include "transform.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

// Some macros used to decide how much working space to allocate.
// Round allocations up to the next cache line.
#define ALLOC_ROUND	64
#define ROUND_UP(x)	((((x) + ALLOC_ROUND-1) / ALLOC_ROUND) * ALLOC_ROUND)

// Implementation of the GemmCommon abstract class.
//
// This implementation interleaves the source matrices in blocks - good for
// larger matrices.
namespace arm_gemm {

template<typename strategy, typename To, typename Tr>
class GemmInterleaved : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    /* const properties set by constructor */
    const CPUInfo * const _ci;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmulti;

    const bool _trA;
    const bool _trB;

    const Tr _alpha;
    const Tr _beta;

    const int _maxthreads;
    int _nthreads;
    const bool _pretransposed;

    /* Blocking info */
    unsigned int _k_block=0;
    unsigned int _x_block=0;
    unsigned int _Mround=0;

    /* Working space, pretransposed buffer, buffer manager */
    const Toi *_B_transposed=nullptr;
    BufferManager *_bm=nullptr;
    void *_working_space=nullptr;

    /* We will need to walk through the blocks of B in a few contexts, so
     * factor that out.  */
    class blockwalker {
    private:
        /* Size loops, etc. based on our parent's configuration */
        const GemmInterleaved<strategy, To, Tr> &_parent;

        /* K, X and multi parameters for current iteration. */
        unsigned int _k0=0, _x0=0, _multi=0;

        unsigned int _index=0;
        bool _done=false;
        bool _newkblock=true;
        bool _newmulti=true;

    public:
        blockwalker(const GemmInterleaved<strategy, To, Tr> &parent) : _parent(parent) { }

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
                    _k0=0;
                    _multi++;
                    if (_multi >= _parent._nmulti) {
                        _done=true;
                        return false;
                    }
                    _newmulti=true;
                }
                _newkblock=true;
            }
            _index++;

            return true;
        }

        unsigned int k0(void) { return _k0; }
        unsigned int x0(void) { return _x0; }
        unsigned int multi(void) { return _multi; }
        unsigned int index(void) { return _index; }
        bool done(void) { return _done; }
        bool newkblock(void) { return _newkblock; }
    };

    // A working size: One of these needed, regardless of thread count.  Divided according to window.
    size_t get_a_working_size() const {
        return ROUND_UP(sizeof(Toi) * _k_block * _Mround * _nbatches);
    }

    // B working size: 0, 1 or 3 of these needed depending on pretransposed and threading settings.
    size_t get_b_working_size() const {
        return ROUND_UP(sizeof(Toi) * _x_block * _k_block);
    }

    // C working size: One needed per thread.
    size_t get_c_working_size() const {
        return ROUND_UP(sizeof(Tri) * _x_block * strategy::out_height());
    }

    // Internal execute function.
    // This supports both the "pretransposed" and "standard" interfaces via the template parameter.
    template<bool pretransposed>
    void execute_internal(unsigned int start, unsigned int end, int threadid) {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        blockwalker current(*this);
        blockwalker next=current;

        /* Translate 'start' and 'end' into a position within the batches and rows. */
        const unsigned int window_per_batch = _Mround / strategy::out_height();
        unsigned int batch_0   = start / window_per_batch;
        unsigned int batch_end = end   / window_per_batch;

        /* Compute the M values to operate on */
        unsigned int m_0   = (start - (batch_0 * window_per_batch)) * strategy::out_height();
        unsigned int m_max = (end - (batch_end * window_per_batch)) * strategy::out_height();

        /* Make sure we've been set up correctly. */
        if (pretransposed) {
            assert(_B_transposed);
        } else {
            assert(_bm);
        }

        assert(_working_space);
        int8_t *working_space_bytes = reinterpret_cast<int8_t *>(_working_space);

        // Private buffers.  Treat working_space as an array of C buffers
        // (one per thread) first, followed by the (window-divided) A
        // buffer.
        // Set a_panel to the base of the A buffers - compute offsets into it based on M/batches later.
        Toi * const a_panel = reinterpret_cast<Toi *>(working_space_bytes + (_maxthreads * get_c_working_size()));
        Tri * const c_panel = reinterpret_cast<Tri *>(working_space_bytes + (threadid * get_c_working_size()));

        // Shared buffers - these come either from BufferManager or _B_transposed.
        const Toi *b_panel;

        if (pretransposed) {
            b_panel = _B_transposed;
        }

        //printf("Starting GEMM loop, x_block=%d, k_block=%d\n", _x_block, _k_block);

        // newkblock() is always true on the first iteration, so this will be set properly on the first loop.
        int kern_k = 0;

        for (;!current.done();current.advance()) {
            if (current.newkblock()) {
#ifdef CYCLE_PROFILING
                auto p=prof.ScopedProfiler(PROFILE_PREPA, (end - start) * strategy::out_height() * (current.kmax()-current.k0()) * sizeof(Toi));
#endif
                for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                    unsigned int first_m = (batch == batch_0)   ? m_0   : 0;
                    unsigned int last_m  = (batch == batch_end) ? m_max : _Msize;

                    if (first_m >= last_m)
                        continue;

                    strat.transforms.PrepareA(a_panel + ((batch * _Mround + first_m) * _k_block),
                                              this->_Aptr + (batch * this->_A_batch_stride) + (current.multi() * this->_A_multi_stride),
                                              this->_lda, first_m, last_m, current.k0(), current.kmax(), _trA);
                }

                // Figure out how many "K" the kernel will actually process.
                kern_k = iceildiv(current.kmax() - current.k0(), strategy::k_unroll());
                kern_k *= strat.k_unroll();
            }

            int bblocks = iceildiv(current.xmax() - current.x0(), strategy::out_width());

            if (!pretransposed) {
                /* Look ahead to the next block and populate it if necessary.
                 * This avoids the populate operation becoming a bottleneck, and
                 * helps keep the threads synchronized (the first thread to get
                 * here will populate while the rest will advance).
                 *
                 * If we are running single threaded, bm->try_populate() will do
                 * nothing.
                 */
                if (next.advance()) {
                    _bm->try_populate(next.index(), [&](void *buffer) {
#ifdef CYCLE_PROFILING
                        auto p=prof.ScopedProfiler(PROFILE_PREPB, (next.xmax()-next.x0()) * (next.kmax()-next.k0()) * sizeof(Toi));
#endif

                        Toi *b_panel = reinterpret_cast<Toi *>(buffer);

                        strat.transforms.PrepareB(b_panel, this->_Bptr + (next.multi() * this->_B_multi_stride), this->_ldb,
                                                  next.x0(), next.xmax(), next.k0(), next.kmax(), _trB);
                    });
                }

                /* Get the buffer for this iteration from the BufferManager. */
                b_panel = reinterpret_cast<Toi *>(_bm->get(current.index(), [&](void *bpv) {
#ifdef CYCLE_PROFILING
                    auto p=prof.ScopedProfiler(PROFILE_PREPB, (current.xmax()-current.x0()) * (current.kmax()-current.k0()) * sizeof(Toi));
#endif

                    Toi *b_panel = reinterpret_cast<Toi *>(bpv);

                    strat.transforms.PrepareB(b_panel, this->_Bptr + (current.multi() * this->_B_multi_stride), this->_ldb,
                                              current.x0(), current.xmax(), current.k0(), current.kmax(), _trB);
                }));
            }

            /* Do the actual work. */
            for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                unsigned int first_m = (batch == batch_0)   ? m_0   : 0;
                unsigned int last_m  = (batch == batch_end) ? m_max : _Msize;

                const Toi *a_ptr = a_panel + (batch * _Mround + first_m) * _k_block;

                if (first_m >= last_m)
                    continue;

                for (unsigned int y=first_m; y<last_m; y+=strategy::out_height()) {
                    unsigned int ymax = std::min(_Msize, y + strategy::out_height());

                    {
#ifdef CYCLE_PROFILING
                        auto p=prof.ScopedProfiler(PROFILE_KERNEL, (strategy::out_height() * bblocks * strategy::out_width() * kern_k));
#endif

                        strat.kernel(a_ptr, b_panel, c_panel, 1, bblocks, kern_k);

                        a_ptr += (strategy::out_height() * kern_k);
                    }

                    {
#ifdef CYCLE_PROFILING
                        auto p=prof.ScopedProfiler(PROFILE_MERGE, (strategy::out_height() * bblocks * strategy::out_width() * sizeof(Tr)));
#endif
                        strat.transforms.Merge(this->_Cptr + (batch * this->_C_batch_stride) + (current.multi() * this->_C_multi_stride),
                                               c_panel, this->_ldc, y, ymax, current.x0(), current.xmax(),
                                               _alpha, (current.k0()==0 ? _beta : static_cast<Tr>(1)));
                    }
                }
            }

            if (pretransposed) {
                b_panel += (bblocks * strat.out_width() * kern_k);
            } else {
                _bm->release(current.index());
            }
        }
    }

public:
    GemmInterleaved(GemmInterleaved &) = delete;
    GemmInterleaved & operator= (GemmInterleaved &) = delete;

    /* Constructor */
    GemmInterleaved(const GemmArgs<Tr> &args)
            : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
              _nbatches(args._nbatches), _nmulti(args._nmulti), _trA(args._trA), _trB(args._trB),
              _alpha(args._alpha), _beta(args._beta), _maxthreads(args._maxthreads), _nthreads(args._maxthreads),
              _pretransposed(args._pretransposed_hint) {
        const unsigned int L1_size = _ci->get_L1_cache_size();
        const unsigned int L2_size = _ci->get_L2_cache_size();

        assert(_maxthreads > 0);

        // Work out blocking parameters, or override from provided GemmConfig
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
            unsigned int num_k_blocks = iceildiv(_Ksize, _k_block);

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
            unsigned int num_x_blocks = iceildiv(_Nsize, _x_block);
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
    // out work in units of out_height.  Factor batches into the window, but
    // not multi for now (as this would cause problems with the buffer
    // manager).
    unsigned int get_window_size() const override {
        // _Mround is a multiple of out_height by definition.
        return (_Mround / strategy::out_height()) * _nbatches;
    }

    // set_nthreads: pass on to buffer manager to avoid it waiting for non-existant threads.
    void set_nthreads(int nthreads) override {
        _nthreads = std::min(nthreads, _maxthreads);
        if (_bm) {
            _bm->set_nthreads(_nthreads);
        }
    }

    // Execute
    void execute(unsigned int start, unsigned int end, int threadid) override {
        if (_pretransposed) {
            execute_internal<true>(start, end, threadid);
        } else {
            execute_internal<false>(start, end, threadid);
        }
    }

    // Interface implementation - working space
    size_t get_working_size() const override {
        // In all cases, we need one A buffer plus a C buffer per thread.
        size_t size = get_a_working_size() + (get_c_working_size() * _maxthreads);

        // For pretransposed case, there is no working space needed for B.
        // Otherwise, we need a BufferManager.
        if (!_pretransposed) {
            size += BufferManager::get_storage_requirement(_maxthreads, get_b_working_size());
        }

        size += 64; // Add on a cache line extra for alignment.

        return size;
    }

    void set_working_space(void *working_space) override {
        // Make sure everything ends up cache line aligned
        int8_t *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        intptr_t working_space_int = reinterpret_cast<intptr_t>(working_space);

        size_t diff=0;

        if (working_space_int & 0x3F) {
            diff = 0x40 - (working_space_int & 0x3F);
        }

        working_space_bytes += diff;

        if (_pretransposed) {
            // Pretransposed case: just set internal pointer to parameter value.
            _working_space = reinterpret_cast<void *>(working_space_bytes);
        } else {
            // Otherwise, use the first part of the working space for the buffer manager.
            // It's legal to call this again so don't leak a buffer manager if it already existed.
            delete _bm;

            _bm = new BufferManager(_nthreads, get_b_working_size(), reinterpret_cast<void *>(working_space_bytes));

            working_space_bytes += BufferManager::get_storage_requirement(_maxthreads, get_b_working_size());

            _working_space = reinterpret_cast<void *>(working_space_bytes);
        }
    }

    // Interface implementation - pretransposed
    bool B_is_pretransposed() const override {
        return _pretransposed;
    }

    bool B_pretranspose_required() const override {
        return _pretransposed && (_B_transposed==nullptr);
    }

    // TODO: this could almost certainly be considerably simpler.
    size_t get_B_pretransposed_array_size() const override {
        size_t total=0;
        blockwalker current(*this);

        do {
            /* Figure out the size of each block. */
            unsigned int x_size = (current.xmax() - current.x0());
            unsigned int k_size = (current.kmax() - current.k0());

            /* Round sizes up as needed. */
            x_size = iceildiv(x_size, strategy::out_width());
            x_size *= strategy::out_width();

            k_size = iceildiv(k_size, strategy::k_unroll());
            k_size *= strategy::k_unroll();

            total += x_size * k_size * sizeof(Toi);
        } while (current.advance());

        return total;
    }

    using GemmCommon<To, Tr>::pretranspose_B_array;
    void pretranspose_B_array(void *in_buffer, const To *B, const int ldb, const int B_multi_stride) override {
        blockwalker current(*this);
        Toi *buffer = reinterpret_cast<Toi *>(in_buffer);
        _B_transposed = buffer;
        strategy strat(_ci);

        do {
            /* Figure out the size of each block. */
            unsigned int x_size = (current.xmax() - current.x0());
            unsigned int k_size = (current.kmax() - current.k0());

            /* Round sizes up as needed. */
            x_size = iceildiv(x_size, strategy::out_width());
            x_size *= strategy::out_width();

            k_size = iceildiv(k_size, strategy::k_unroll());
            k_size *= strategy::k_unroll();

            strat.transforms.PrepareB(buffer, B + (current.multi() * B_multi_stride), ldb,
                                      current.x0(), current.xmax(), current.k0(), current.kmax(), _trB);

            buffer += (x_size * k_size);
        } while (current.advance());
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        _B_transposed = reinterpret_cast<Toi *>(in_buffer);
    }

    ~GemmInterleaved() override {
        delete _bm;
    }
};

} // namespace arm_gemm
