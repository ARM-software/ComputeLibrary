/*
 * Copyright (c) 2017 ARM Limited.
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
#include <cassert>

#include "gemm_common.hpp"
#include "profiler.hpp"
#include "transform.hpp"
#include "mergeresults.hpp"

// Some macros used to decide how much working space to allocate.
// Round allocations up to the next cache line.
#define ALLOC_ROUND	64
#define ROUND_UP(x)	((((x) + ALLOC_ROUND-1) / ALLOC_ROUND) * ALLOC_ROUND)

// Implementation of the GemmCommon abstract class.
//
// This implementation interleaves the source matrices in blocks - good for
// larger matrices.
template<typename strategy, typename To, typename Tr>
class GemmInterleaved : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    const unsigned int M;
    const unsigned int N;
    const unsigned int K;

    const bool trA;
    const bool trB;

    const strategy strat;

    unsigned int k_block = 0;
    unsigned int x_block = 0;
    unsigned int Mround = 0;

    size_t get_a_working_size() const {
        return ROUND_UP(sizeof(Toi) * k_block * Mround);
    }

    size_t get_b_working_size() const {
        return ROUND_UP(sizeof(Toi) * x_block * k_block);
    }

    size_t get_c_working_size() const {
        return ROUND_UP(sizeof(Tri) * x_block * strat.out_height);
    }

public:
    size_t get_working_size() const override {
        return get_a_working_size() + get_b_working_size() + get_c_working_size();
    }

    GemmInterleaved(const CPUInfo *ci, const unsigned int M, const unsigned int N, const unsigned int K, const bool trA, const bool trB) : M(M), N(N), K(K), trA(trA), trB(trB), strat(ci) {
        const unsigned int L1_size = ci->L1_size;
        const unsigned int L2_size = ci->L2_size;

        // Work out blocking parameters
        // k_block: Each iteration will consume (out_width + out_height)
        // operands - so how many iterations will fill the L1?
        k_block = L1_size / (sizeof(Toi) * (strat.out_width + strat.out_height));

        // Needs to be a multiple of the K unroll level.
        k_block /= strat.k_unroll;
        k_block *= strat.k_unroll;

        // Now tune to presented problem size; this is how many blocks we need.
        int num_k_blocks = (K + (k_block - 1)) / k_block;

        // So divide the space equally into that many blocks.
        k_block = (K + num_k_blocks - 1) / num_k_blocks;

        // And round UP to the K unroll level required.
        k_block = (k_block + strat.k_unroll - 1) / strat.k_unroll;
        k_block *= strat.k_unroll;

        // x_block: Work out how many rows (of length k_block) will fit in the L2
        x_block = L2_size / (sizeof(Toi) * k_block);

        // Needs to be a multiple of the kernel output width.
        x_block /= strat.out_width;
        x_block *= strat.out_width;

        // And tune to the presented problem size.
        int num_x_blocks = (N + (x_block - 1)) / x_block;
        x_block = (N + num_x_blocks - 1) / num_x_blocks;

        x_block = (x_block + strat.out_width - 1) / strat.out_width;
        x_block *= strat.out_width;

        // Work out the rounded size of M - needed for some buffers.
        Mround = (M + (strat.out_height - 1)) / strat.out_height;
        Mround *= strat.out_height;

    }

    // Actually execute the GEMM.
    void execute(const To *A, const int lda, const To *B, const int ldb, Tr *C, const int ldc, const Tr alpha, const Tr beta, void *working_space) const override {
        assert(working_space);
        profiler prof;
        int8_t *working_space_bytes = reinterpret_cast<int8_t *>(working_space);
        intptr_t working_space_int = reinterpret_cast<intptr_t>(working_space_bytes);
        size_t diff = 0;

        if (working_space_int & 0xF) {
            diff = 0x10 - (working_space_int & 0xF);
        }

        Toi * const a_panel = reinterpret_cast<Toi *>(working_space_bytes + diff);
        Toi * const b_panel = reinterpret_cast<Toi *>(working_space_bytes + get_a_working_size() + diff);
        Tri * const c_panel = reinterpret_cast<Tri *>(working_space_bytes + get_a_working_size() + get_b_working_size() + diff);

        for (unsigned int k0=0; k0<K; k0 += k_block) {
            unsigned int kmax = k0 + k_block;
            if (kmax > K) kmax = K;

            // Figure out how many "K" the kernel will actually process.
            int kern_k = ((kmax - k0) + (strat.k_unroll - 1)) / strat.k_unroll;
            kern_k *= strat.k_unroll;

            prof(PROFILE_PREPA, (M * (kmax-k0) * sizeof(Toi)), [&](void) {
                if (trA ^ strategy::A_transpose) {
                    Transform<strategy::A_interleave, strategy::A_block, true>(a_panel, A, lda, 0, M, k0, kmax);
                } else {
                    Transform<strategy::A_interleave, strategy::A_block, false>(a_panel, A, lda, 0, M, k0, kmax);
                }
            });

            for (unsigned int x0=0; x0<N; x0 += x_block) {
                unsigned int xmax = x0 + x_block;
                if (xmax > N) xmax = N;

                int bblocks = (xmax - x0 + strat.out_width - 1) / strat.out_width;

                prof(PROFILE_PREPB, (xmax-x0) * (kmax-k0) * sizeof(Toi), [&](void) {
                    if (trB ^ strategy::B_transpose) {
                        Transform<strategy::B_interleave, strategy::B_block, true>(b_panel, B, ldb, x0, xmax, k0, kmax);
                    } else {
                        Transform<strategy::B_interleave, strategy::B_block, false>(b_panel, B, ldb, x0, xmax, k0, kmax);
                    }
                });

                for (unsigned int y=0; y<M; y+=strat.out_height) {
                    unsigned int ymax = y + strat.out_height;
                    if (ymax > M) ymax = M;

                    prof(PROFILE_KERNEL, (strat.out_height * bblocks * strat.out_width * kern_k), [&](void) { strat.kernel(a_panel + (y * kern_k), b_panel, c_panel, 1, bblocks, kern_k); });
                    prof(PROFILE_MERGE, (strat.out_height * bblocks * strat.out_width * sizeof(Tr)), [&](void) { MergeResults<strategy::out_width, strategy::out_height>(C, c_panel, ldc, y, ymax, x0, xmax, alpha, (k0==0 ? beta : static_cast<Tr>(1))); });
                }
            }
        }
    }
};
