/*
 * Copyright (c) 2020 Arm Limited.
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

#include "arm_gemm.hpp"
#include "utils.hpp"

#include "mergeresults.hpp"
#include "transform.hpp"

#ifdef CYCLE_PROFILING
#include "profiler.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>

// Some macros used to decide how much working space to allocate.
// Round allocations up to the next cache line.
#define ALLOC_ROUND    64
#define ROUND_UP(x)    ((((x) + ALLOC_ROUND-1) / ALLOC_ROUND) * ALLOC_ROUND)

// Implementation of the GemmCommon abstract class.
//
// This implementation interleaves the source matrices in blocks - good for
// larger matrices.
namespace arm_gemm {

template<typename strategy, typename To, typename Tr>
class GemmInterleavedPretransposed2d : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    /* const properties set by constructor */
    const CPUInfo * const _ci;

    const unsigned int _Msize;
    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const unsigned int _nbatches;
    const unsigned int _nmulti;

    const Activation _act;

    const int _maxthreads;
    int _nthreads;

    /* Blocking info */
    unsigned int _k_block=0;
    unsigned int _x_block=0;

    unsigned int _Mround_div=0;
    unsigned int _Mround=0;
    unsigned int _Nround_div=0;
    unsigned int _Nround=0;

    /* Working space, pretransposed buffer */
    const Toi *_B_transposed=nullptr;
    void *_working_space=nullptr;

    /* We will need to walk through the blocks of B in a few contexts, so
     * factor that out.  */
    class blockwalker {
    private:
        /* Size loops, etc. based on our parent's configuration */
        const GemmInterleavedPretransposed2d<strategy, To, Tr> &_parent;

        /* K, X and multi parameters for current iteration. */
        unsigned int _k0=0, _x0=0, _xmin=0, _xmax=0, _multi=0;

        unsigned int _index=0;
        bool _done=false;
        bool _newkblock=true;
        bool _newmulti=true;

    public:
        blockwalker(const GemmInterleavedPretransposed2d<strategy, To, Tr> &parent)
        : _parent(parent)
        , _xmax { parent._Nsize }
        { }

        blockwalker(const GemmInterleavedPretransposed2d<strategy, To, Tr> &parent, unsigned int x0, unsigned int xmax)
        : _parent(parent)
        , _x0   { x0   }
        , _xmin { x0   }
        , _xmax { xmax }
        {
            assert(_x0 <= _xmax);
        }

        unsigned int xmax() {
            return std::min(_x0 + _parent._x_block, _xmax);
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
            if (_x0 >= _xmax) {
                _x0=_xmin;
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
        return ROUND_UP(sizeof(Toi) * _k_block * _Mround * _nbatches) * 2;
    }

    // As B will be pretranspose we do not need to alloc any space for it
    size_t get_b_working_size() const {
        return 0;
    }

    // C working size: One needed per thread.
    size_t get_c_working_size() const {
        return ROUND_UP(sizeof(Tri) * _x_block * strategy::out_height());
    }

    // Internal execute function.
    // This supports both the "pretransposed" and "standard" interfaces via the template parameter.
    void execute_pretranspose(unsigned int m_start, unsigned int m_end, unsigned int n_start, unsigned int n_end, int threadid, int, int) {
        /* Make sure we've been set up correctly. */
        assert(_B_transposed);
        assert(_working_space);
        assert(this->_Aptr);
        assert(this->_Cptr);

#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        /* Translate 'start' and 'end' into a position within the batches and rows. */
        const unsigned int window_per_batch = _Mround / strategy::out_height();
        unsigned int batch_0   = m_start / window_per_batch;
        unsigned int batch_end = m_end   / window_per_batch;

        /* Compute the M values to operate on */
        unsigned int m_0   = (m_start - (batch_0 * window_per_batch)) * strategy::out_height();
        unsigned int m_max = (m_end - (batch_end * window_per_batch)) * strategy::out_height();

        unsigned int n_0   = std::min(this->_Nsize, strategy::out_width() * n_start);
        unsigned int n_max = std::min(this->_Nsize, strategy::out_width() * n_end);

        blockwalker current(*this, n_0, n_max);

        int8_t *working_space_bytes = reinterpret_cast<int8_t *>(_working_space);

        auto c_panel_start = working_space_bytes;
        auto a_panel_start = c_panel_start + get_c_working_size() * _maxthreads;

        auto c_panel = reinterpret_cast<Tri *>(c_panel_start + get_c_working_size() * threadid);
        auto a_panel = reinterpret_cast<Toi *>(a_panel_start + get_a_working_size() * threadid);

        /* B^t is stored in interleaved panels separated by their K-block component
         * we want to store a pointer to the start of the current k-page
         * then when we come to the next k-block we just add the size of the previous to
         * this base pointer
         */
        const Toi *b_panel_start = _B_transposed;
        // b_panels stores a pointer to the start of our current block inside of the k-block
        const Toi *b_panel       = b_panel_start;

        // newkblock() is always true on the first iteration, so this will be set properly on the first loop.
        unsigned b_page_size = 0;
        int kern_k = 0;
        for (;!current.done();current.advance()) {
            int bblocks = iceildiv(current.xmax() - current.x0(), strategy::out_width());

            if (current.newkblock()) {
                kern_k         = iceildiv(current.kmax() - current.k0(), strategy::k_unroll());
                kern_k        *= strat.k_unroll();

                unsigned b_thread_start_offset = iceildiv(current.x0(), strategy::out_width());

                b_panel_start += b_page_size;
                b_panel        = b_panel_start + (b_thread_start_offset * strat.out_width() * kern_k);
                b_page_size    = _Nround * kern_k;

                for (unsigned int batch = batch_0; batch <= batch_end; batch++) {
                    unsigned int first_m = (batch == batch_0)   ? m_0   : 0;
                    unsigned int last_m  = (batch == batch_end) ? m_max : _Msize;

                    if (first_m >= last_m)
                        continue;

                    auto a_thread_panel_in  = this->_Aptr
                                            + (batch * this->_A_batch_stride)
                                            + (current.multi() * this->_A_multi_stride);

                    auto a_thread_panel_out = a_panel + ((batch * _Mround + first_m) * _k_block);

                    strat.transforms.PrepareA(
                        a_thread_panel_out,
                        a_thread_panel_in,
                        this->_lda,
                        first_m,
                        last_m,
                        current.k0(),
                        current.kmax(),
                        0);
                }
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

                    strat.kernel(a_ptr, b_panel, c_panel, 1, bblocks, kern_k);
                    a_ptr += (strategy::out_height() * kern_k);

                    /* Only activate on last pass, only add bias on first pass, ask for accumulation on any non-first pass */
                    const bool first_pass = current.k0()==0;
                    const bool last_pass  = current.kmax()==_Ksize;

                    auto c_panel_out = this->_Cptr
                                     + this->_C_batch_stride * batch
                                     + this->_C_multi_stride * current.multi();

                    auto bias        = (first_pass && this->_bias)
                                     ? this->_bias + (current.multi() * this->_bias_multi_stride)
                                     : nullptr;

                    auto act        = last_pass ? _act : Activation();

                    strat.transforms.Merge(
                        c_panel_out,
                        c_panel,
                        this->_ldc,
                        y,
                        ymax,
                        current.x0(),
                        current.xmax(),
                        bias,
                        act,
                        !first_pass);  //Append
                }
            }

            b_panel += (bblocks * strat.out_width() * kern_k);
        }
    }

    static unsigned int get_k_block_size(const GemmArgs &args) {
        // Work out blocking parameters, or override from provided GemmConfig
        if (args._cfg && args._cfg->inner_block_size) {
            return args._cfg->inner_block_size;
        }

        const unsigned int L1_size = args._ci->get_L1_cache_size();
        unsigned int k_block;

        // k_block: Find out how much of the larger array can be loaded into half the cache.
        // This should account for associative caches.
        k_block = (L1_size / 2) / (sizeof(Toi) * (std::max(strategy::out_width(), strategy::out_height())));

        // Needs to be (at least a single) multiple of the K unroll level.
        k_block /= strategy::k_unroll();
        k_block = std::max(k_block, 1U) * strategy::k_unroll();

        // Now tune to presented problem size; this is how many blocks we need.
        unsigned int numk_blocks = iceildiv(args._Ksize, k_block);

        // So divide the space equally into that many blocks.
        k_block = iceildiv(args._Ksize, numk_blocks);

        // And round UP to the K unroll level required.
        k_block = iceildiv(k_block, strategy::k_unroll());
        k_block *= strategy::k_unroll();

        return k_block;
    }

public:
    GemmInterleavedPretransposed2d(GemmInterleavedPretransposed2d &) = delete;
    GemmInterleavedPretransposed2d & operator= (GemmInterleavedPretransposed2d &) = delete;

    /* Constructor */
    GemmInterleavedPretransposed2d(const GemmArgs &args)
    :    _ci(args._ci)
    ,    _Msize(args._Msize)
    ,    _Nsize(args._Nsize)
    ,    _Ksize(args._Ksize)
    ,    _nbatches(args._nbatches)
    ,    _nmulti(args._nmulti)
    ,    _act(args._act)
    ,    _maxthreads(args._maxthreads)
    ,    _nthreads(args._maxthreads)
    ,    _k_block(get_k_block_size(args))
    // Work out the rounded size of M - needed for some buffers.
    ,    _Mround_div ( iceildiv(_Msize, strategy::out_height()) )
    ,    _Mround     ( _Mround_div * strategy::out_height()     )

    ,    _Nround_div ( iceildiv(_Nsize, strategy::out_width()) )
    ,    _Nround     ( _Nround_div * strategy::out_width()     )
    {
        assert(_maxthreads > 0);

        const unsigned int L2_size = _ci->get_L2_cache_size();

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
    }

    // Interface implementation - Compulsory functions
    ndrange_t get_window_size() const override {
        unsigned m = (_Mround / strategy::out_height()) * _nbatches;
        unsigned n = _Nround_div;

        return { m, n };
    }

    bool supports_dynamic_scheduling() const override {
        return true;
    }

    // set_nthreads: pass on to buffer manager to avoid it waiting for non-existant threads.
    void set_nthreads(int nthreads) override {
        _nthreads = std::min(nthreads, _maxthreads);
    }

    void execute(const ndcoord_t& work_range, const ndcoord_t& thread_locator, int threadid) override {
        /* This particular GEMM implementation can only be broken up over the M & N
         * dimensions, we inform the frame work of this limitation via the get_window_size function
         */
        const auto m_start = work_range.get_position(0);
        const auto n_start = work_range.get_position(1);
        const auto m_size  = work_range.get_size(0);
        const auto n_size  = work_range.get_size(1);
        const auto m_end   = m_start + m_size;
        const auto n_end   = n_start + n_size;

        const auto m_threadid = thread_locator.get_position(0);
        const auto n_threadid = thread_locator.get_position(1);

        execute_pretranspose(m_start, m_end, n_start, n_end, threadid, m_threadid, n_threadid);
    }

    std::size_t get_working_size() const override {
        /* Because we do not know how schedular will break up
         * the task, we need to ensure that alloc enough
         * space to be able to handle the case where every thread
         * is parallelised across B AND also every thrread is parallelised across A
         *
         * If we parallelise across A, then we only need one buffer of A and 64 buffers of B
         * If we parallelise across B, then we only need 64 buffer of B and
         */
        return get_c_working_size() * _maxthreads
             + get_a_working_size() * _maxthreads
             + 64; //to account for cacheline alignment
    }


    void set_working_space(void *working_space) override {
        // Make sure everything ends up cache line aligned
        int8_t *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        intptr_t working_space_int  = reinterpret_cast<intptr_t>(working_space);

        size_t diff=0;

        if (working_space_int & 0x3F) {
            diff = 0x40 - (working_space_int & 0x3F);
        }

        working_space_bytes += diff;

        _working_space = reinterpret_cast<void *>(working_space_bytes);
    }

    // Interface implementation - pretransposed
    bool B_is_pretransposed() const override {
        return true;
    }

    bool B_pretranspose_required() const override {
        return _B_transposed==nullptr;
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
                                      current.x0(), current.xmax(), current.k0(), current.kmax());

            buffer += (x_size * k_size);
        } while (current.advance());
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        _B_transposed = reinterpret_cast<Toi *>(in_buffer);
    }

    // Estimate cycles for given problem given provided parameters
    static uint64_t estimate_cycles(const GemmArgs &args, const PerformanceParameters &params) {
        unsigned int k_blocks = iceildiv(args._Ksize, get_k_block_size(args));
        unsigned int m_blocks = iceildiv(args._Msize, strategy::out_height()) * args._nbatches;
        unsigned int n_blocks = iceildiv(args._Nsize, strategy::out_width());

        uint64_t total_macs    = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * roundup(args._Nsize, strategy::out_width()) * roundup(args._Ksize, strategy::k_unroll());
        uint64_t prepare_bytes = static_cast<uint64_t>(args._nbatches) * args._nmulti * roundup(args._Msize, strategy::out_height()) * roundup(args._Ksize, strategy::k_unroll()) * sizeof(Toi);
        uint64_t merge_bytes   = static_cast<uint64_t>(args._nbatches) * args._nmulti * k_blocks * roundup(args._Msize, strategy::out_height()) * roundup(args._Nsize, strategy::out_width()) * sizeof(Tr);

        // Wide problems incur extra preparation cost, as it is done per thread.
        // Duplicate the logic the scheduler will later use to figure out how much that will affect us
        float ratio = m_blocks / static_cast<float>(n_blocks);

        unsigned int ideal_height = static_cast<unsigned int>(std::sqrt(args._maxthreads * ratio) + 0.5);
        unsigned int height = 1;

        if (ideal_height == 0) {
            height = 1;
        } else {
            for (unsigned int adj=0; adj<ideal_height; adj++) {
                const unsigned int round_down = ideal_height - adj;
                if (args._maxthreads % round_down == 0) {
                    height = round_down;
                    break;
                }

                const unsigned int round_up = ideal_height + adj;
                if (args._maxthreads % round_up == 0) {
                    height = round_up;
                    break;
                }
            }
        }

        // We've computed the height here - we need to multiply the amount of preparation effort by the width (which is total threads / height)
        prepare_bytes *= (args._maxthreads / height);

        float mac_cycles     = static_cast<float>(total_macs) / params.kernel_macs_cycle;
        float prepare_cycles = static_cast<float>(prepare_bytes) / params.prepare_bytes_cycle;
        float merge_cycles   = static_cast<float>(merge_bytes) / params.merge_bytes_cycle;

        float total_cycles = mac_cycles + prepare_cycles + merge_cycles;

        // We can't thread over multis, which might be a problem in some
        // threaded cases.  Penalize that here.
        float parallelism_available = static_cast<float>(iceildiv(args._Msize, strategy::out_height()) * args._nbatches * iceildiv(args._Nsize, strategy::out_width())) * 0.9;

        if (parallelism_available < args._maxthreads) {
            total_cycles *= (static_cast<float>(args._maxthreads) / parallelism_available);
        }

        return static_cast<uint64_t>(total_cycles);
    }
};

} // namespace arm_gemm
