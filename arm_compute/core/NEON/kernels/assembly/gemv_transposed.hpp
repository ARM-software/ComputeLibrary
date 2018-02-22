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
// This is implementation is for GEMV with a transposed matrix.
//
// By default the source data is used in-place, but if type conversion is
// needed we need to allocate working space (CURRENTLY NOT IMPLEMENTED).

template<typename strategy, typename To, typename Tr>
class GemvTransposed : public GemmCommon<To, Tr> {
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type Tri;

    const unsigned int N;
    const unsigned int K;

    const strategy strat;

    unsigned int m_block;
    unsigned int n_block;

    size_t get_a_working_size() const {
        return ROUND_UP(sizeof(Toi) * m_block);
    }

    size_t get_b_working_size() const {
        return ROUND_UP(sizeof(Toi) * m_block * n_block);
    }

    size_t get_c_working_size() const {
        return ROUND_UP(sizeof(Tri) * n_block);
    }

public:
    size_t get_working_size() const override {
        return get_a_working_size() + get_b_working_size() + get_c_working_size();
    }

    GemvTransposed(const CPUInfo *ci, const unsigned int N, const unsigned int K) : N(N), K(K), strat(ci) {
        m_block = K;
        n_block = N;
    }

    // Actually execute the GEMV.
    void execute(const To *A, const int lda, const To *B, const int ldb, Tr *C, const int ldc, const Tr alpha, const Tr beta, void *working_space) const override {
        profiler prof;

        static_assert(std::is_same<To, Toi>::value, "gemv_transposed: Operand types must be the same.");
        static_assert(std::is_same<Tr, Tri>::value, "gemv_transposed: Result types must be the same.");

        for (unsigned int m0=0; m0<K; m0+=m_block) {
            unsigned int mmax = m0 + m_block;
            if (mmax > K) mmax = K;

            for (unsigned int n0=0; n0<N; n0+=n_block) {
                unsigned int nmax = n0 + n_block;
                if (nmax > N) nmax = N;

                prof(PROFILE_KERNEL, ((mmax-m0) * (nmax-n0)), [&](void) { strat.kernel(B + (m0 * ldb) + n0, A + m0, C + n0, alpha, ldb, (mmax-m0), (nmax-n0)); });
            }
        }
    }
};
