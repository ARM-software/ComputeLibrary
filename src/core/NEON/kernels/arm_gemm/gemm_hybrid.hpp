/*
 * Copyright (c) 2017-2021 Arm Limited.
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

#include <algorithm>
#include <cassert>

#include "arm_gemm.hpp"
#include "bias_adder.hpp"
#include "ndrange.hpp"
#include "performance_parameters.hpp"
#include "transform.hpp"
#include "utils.hpp"

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

    const Activation _act;

    /* Blocking info */
    const unsigned int _k_block;
    const unsigned int _n_block;
    const unsigned int _Mround;

    /* Pretransposed buffer. */
    const Toi *_B_transposed=nullptr;

    const NDRange<4> _window_range;

    static unsigned int compute_k_block(const GemmArgs &args) {
        // Some kernels don't support accumulate mode - these can't do K blocking at all.
        if (!strategy::supports_accumulate()) {
            return args._Ksize;
        }

        if (args._cfg && args._cfg->inner_block_size) {
            return roundup(args._cfg->inner_block_size, strategy::k_unroll());
        }

        // Target block size (512 for FP32, scaling for other types).  Don't block until size reaches 1.5X this.
        unsigned int target_block_size = 2048 / sizeof(To);

        if (args._Ksize >= ((3 * target_block_size) / 2)) {
            unsigned int target_blocks = iceildiv(args._Ksize, target_block_size);

            unsigned int block_size = iceildiv(args._Ksize, target_blocks);

            block_size = roundup(block_size, strategy::k_unroll());

            return block_size;
        }

        return args._Ksize;
    }

    // New N blocking strategy: if it's narrow, or much taller than it is wide, do the full width.  Otherwise do a
    // single block.
    static unsigned int compute_n_block(const GemmArgs &args) {
        if (args._cfg && args._cfg->outer_block_size) {
            unsigned int n_block = args._cfg->outer_block_size;

            // Needs to be (at least a single) multiple of the kernel output width.
            n_block /= strategy::out_width();
            n_block = std::max(n_block, 1u) * strategy::out_width();

            return n_block;
        }

        if (args._Nsize <= 64) {
            return args._Nsize;
        }

        if ((args._Msize / args._Nsize) > 155) {
            return args._Nsize;
        }

        // Go slightly wider if thread count and depth are small.
        if ((args._Ksize <= 128) && (args._maxthreads <= 16)) {
            return strategy::out_width() * 3;
        }

        return strategy::out_width();
    }

public:
    GemmHybrid(GemmHybrid &) = delete;
    GemmHybrid & operator= (GemmHybrid &) = delete;

    /* Constructor */
    GemmHybrid(const GemmArgs &args)
              : _ci(args._ci), _Msize(args._Msize), _Nsize(args._Nsize), _Ksize(args._Ksize),
                _nbatches(args._nbatches), _nmulti(args._nmulti),
                _act(args._act),
                _k_block(compute_k_block(args)), _n_block(compute_n_block(args)),
                _Mround(roundup(args._Msize, strategy::out_height())),
                _window_range(iceildiv(args._Msize, strategy::out_height()), _nbatches, iceildiv(_Nsize, _n_block), _nmulti) { }

    // Interface implementation - Compulsory functions
    ndrange_t get_window_size() const override {
        return { _window_range.total_size() };
    }

    // This kernel can always be dynamically scheduled.
    bool supports_dynamic_scheduling() const override {
        return true;
    }

    // Execute
    void execute(const ndcoord_t &work_range, const ndcoord_t &, int) override {
#ifdef CYCLE_PROFILING
        profiler prof;
#endif
        strategy strat(_ci);

        /* Make sure we've been set up correctly. */
        assert(_B_transposed);
        static_assert(std::is_same<To, Toi>::value, "gemm_native: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemm_native: Result types must be the same.");

        /* For now, each work item implies all the K for a given output
         * pixel (so we don't need to synchronize access to the output
         * array).  So separate the loop over K blocks here.  */
        for (unsigned int k0=0; k0<_Ksize; k0+=_k_block) {
            unsigned int kmax   = std::min(k0 + _k_block, _Ksize);
            unsigned int kern_k = roundup(kmax-k0, strategy::k_unroll());

            const bool first_pass = (k0 == 0);
            const bool last_pass = (kmax == _Ksize);

            auto p = _window_range.iterator(work_range.get_position(0), work_range.get_position_end(0));

            if (p.done()) {
                return;
            }

            do {
                const unsigned int m_start = p.dim(0) * strategy::out_height();
                const unsigned int m_end   = std::min(p.dim0_max() * strategy::out_height(), _Msize);
                const unsigned int batch   = p.dim(1);
                const unsigned int n0      = p.dim(2) * _n_block;
                const unsigned int nmax    = std::min(n0 + _n_block, _Nsize);
                const unsigned int multi   = p.dim(3);

                const Toi *b_panel = _B_transposed +
                                     (multi * roundup(_Nsize, strategy::out_width()) * roundup(_Ksize, strategy::k_unroll())) +
                                     (k0 * roundup(_Nsize, strategy::out_width())) +
                                     (n0 * kern_k);

#ifdef CYCLE_PROFILING
                auto p = prof.ScopedProfiler(PROFILE_KERNEL, (unsigned long)(m_end - m_start) * kern_k * roundup(nmax-n0, strategy::out_width()));
#endif

                strat.kernel(this->_Aptr + (multi * this->_A_multi_stride) + (batch * this->_A_batch_stride) + (m_start * this->_lda) + k0, this->_lda,
                             b_panel,
                             this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (m_start * this->_ldc) + n0, this->_ldc,
                             (m_end - m_start), (nmax - n0), kmax-k0,
                             (strategy::supports_bias() && first_pass && this->_bias) ? this->_bias + (multi * this->_bias_multi_stride) + n0 : nullptr,
                             last_pass ? _act : Activation(), !first_pass);

                // Add bias externally if needed
                if (!strategy::supports_bias() && this->_bias && first_pass) {
                    bias_adder(this->_Cptr + (multi * this->_C_multi_stride) + (batch * this->_C_batch_stride) + (m_start * this->_ldc) + n0, this->_ldc,
                               this->_bias + (multi * this->_bias_multi_stride) + n0,
                               (m_end - m_start), (nmax - n0));
                }

            } while (p.next_dim1());
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
        return roundup(_Nsize, strategy::out_width()) * roundup(_Ksize, strategy::k_unroll()) * _nmulti * sizeof(Toi);
    }

    void pretranspose_B_array(void *in_buffer, const To *B, const int ldb, const int B_multi_stride) override {
        Toi *buffer = reinterpret_cast<Toi *>(in_buffer);
        _B_transposed = buffer;
        strategy strat(_ci);

        for (unsigned int multi=0; multi<_nmulti; multi++) {
            for (unsigned int k0=0; k0<_Ksize; k0+=_k_block) {
                const unsigned int kmax = std::min(k0 + _k_block, _Ksize);
                const unsigned int k_size = roundup(kmax-k0, strategy::k_unroll());

                for (unsigned int x0=0; x0<_Nsize; x0+=_n_block) {
                    const unsigned int xmax = std::min(x0+_n_block, _Nsize);

                    const unsigned int size = roundup(xmax-x0, strategy::out_width()) * k_size;

                    strat.transforms.PrepareB( buffer, B + (multi * B_multi_stride), ldb,
                                               x0, xmax, k0, kmax);

                    buffer += size;
                }
            }
        }
    }

    void set_pretransposed_B_data(void *in_buffer) override {
        _B_transposed = reinterpret_cast<Toi *>(in_buffer);
    }

    // Estimate cycles for given problem given provided parameters
    static uint64_t estimate_cycles(const GemmArgs &args, const PerformanceParameters &params) {
        // Note: Current hybrid kernels don't actually round up height (they
        // have paths for each possible height).  Might need to make this
        // configurable in future.
        uint64_t total_macs = static_cast<uint64_t>(args._nbatches) * args._nmulti * args._Msize * roundup(args._Nsize, strategy::out_width()) * roundup(args._Ksize, strategy::k_unroll());

        float mac_cycles = static_cast<float>(total_macs) / params.kernel_macs_cycle;

        // TODO: A bit of a kludge here: current hybrid kernels incur extra
        // overhead where the width is not a multiple of kernel width.  It's
        // most noticable where the overall width is quite low, so add 15%
        // penalty for such widths.
        if ((args._Nsize < strategy::out_width()) || (args._Nsize > strategy::out_width() && args._Nsize < 2*strategy::out_width())) {
            mac_cycles *= 1.15f;
        }

        uint64_t total_cycles = mac_cycles;

        return total_cycles;
    }

    GemmConfig get_config() override {
        GemmConfig c;

        c.method = GemmMethod::GEMM_HYBRID;
        c.inner_block_size = _k_block;
        c.outer_block_size = _n_block;
        c.filter = get_type_name<strategy>();

        return c;
    }
};

} // namespace arm_gemm
