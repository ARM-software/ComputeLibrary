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
#include "profiler.hpp"
#include "transform.hpp"

namespace arm_gemm
{
// Implementation of the GemmCommon abstract class.
//
// This is implementation is for GEMV with a transposed matrix.
//
// By default the source data is used in-place, but if type conversion is
// needed we need to allocate working space (CURRENTLY NOT IMPLEMENTED).

template <typename strategy, typename To, typename Tr>
class GemvPretransposed : public GemmCommon<To, Tr>
{
    typedef typename strategy::operand_type Toi;
    typedef typename strategy::result_type  Tri;

    const unsigned int _Nsize;
    const unsigned int _Ksize;

    const bool _trB;

    const Tr _beta;

    const CPUInfo *const _ci;

    unsigned int m_block = 0;
    unsigned int n_block = 0;

    const Toi *_A_pretransposed = nullptr;

public:
    GemvPretransposed(GemvPretransposed &) = delete;
    GemvPretransposed &operator=(GemvPretransposed &) = delete;

    GemvPretransposed(const CPUInfo *ci, const unsigned int N, const unsigned int K, const bool trB, const Tr beta)
        : _Nsize(N), _Ksize(K), _trB(trB), _beta(beta), _ci(ci)
    {
        /* For now don't do any blocking. TODO: figure out if we should. */
        m_block = K;
        n_block = N;
    }

    // Window is number of out_width blocks.
    unsigned int get_window_size() const override
    {
        return iceildiv(_Nsize, strategy::out_width);
    }

    // Actually execute the GEMV.
    void execute(unsigned int start, unsigned int end, int) override
    {
        profiler prof;
        strategy strat(_ci);

        unsigned int N_start = start * strategy::out_width;
        unsigned int N_end   = std::min(end * strategy::out_width, _Nsize);

        static_assert(std::is_same<Tr, Tri>::value, "GemvPretransposed: Result types must be the same.");

        for(unsigned int m0 = 0; m0 < _Ksize; m0 += m_block)
        {
            unsigned int mmax = std::min(m0 + m_block, _Ksize);

            for(unsigned int n0 = N_start; n0 < N_end; n0 += n_block)
            {
                unsigned int nmax = std::min(n0 + n_block, N_end);

                prof(PROFILE_KERNEL, ((mmax - m0) * (nmax - n0)), [&](void)
                {
                    /* This assumes that the underlying call was a GEMM with M=1; for the N=1 case we would have to pick up this->_Bptr below instead */
                    strat.kernel(_A_pretransposed + (n0 * _Ksize) + (m0 * strategy::A_interleave), (_Ksize * strategy::A_interleave), this->_Aptr + m0, this->_Cptr + n0, _beta, (mmax - m0), (nmax - n0));
                });
            }
        }
    }

    /* Pretransposed interface implementation */
    bool B_is_pretransposed() const override
    {
        return true;
    }

    bool B_pretranspose_required() const override
    {
        /* Transpose is required if _A_pretransposed is still nullptr */
        return (_A_pretransposed == nullptr);
    }

    size_t get_B_pretransposed_array_size() const override
    {
        return _Ksize * iceildiv(_Nsize, strategy::A_interleave) * strategy::A_interleave * sizeof(float);
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb) override
    {
        Toi *A_buffer = reinterpret_cast<Toi *>(buffer);

        /* Reverse sense here as we are dealing with B rather than A.  So if
         * strategy::A_transpose is false and _trB is false, we still
         * transpose.  */
        if(_trB ^ strategy::A_transpose)
        {
            Transform<strategy::A_interleave, strategy::A_block, false>(A_buffer, B, ldb, 0, _Nsize, 0, _Ksize);
        }
        else
        {
            Transform<strategy::A_interleave, strategy::A_block, true>(A_buffer, B, ldb, 0, _Nsize, 0, _Ksize);
        }

        _A_pretransposed = A_buffer;
    }
};

} // namespace arm_gemm
