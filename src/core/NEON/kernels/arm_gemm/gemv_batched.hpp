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

#include "arm_gemm.hpp"

namespace arm_gemm {

/* "Batched GEMV" (where M=1 and nbatches>1) can be executed much more
 * efficiently as a GEMM (with M'=nbatches and nbatches'=1).  This wrapper
 * implements this.  */
template<typename To, typename Tr>
class GemvBatched : public GemmCommon<To, Tr> {
private:
    UniqueGemmCommon<To, Tr> _subgemm = nullptr;

public:
    GemvBatched(const GemmArgs<Tr> &args) {
        /* Just create a subgemm with batches->M */
        GemmArgs<Tr> newargs = args;
        newargs._Msize = args._nbatches;
        newargs._nbatches = 1;
        _subgemm = gemm<To,Tr>(newargs);
    }

    void set_arrays(const To *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                    const To *B, const int ldb, const int B_multi_stride,
                    Tr *C, const int ldc, const int C_batch_stride, const int C_multi_stride) override {
        /* A and C's batch stride becomes their new row stride.  New batch stride is 0 as nbatches for subgemm is always 1. */
        _subgemm->set_arrays(A, A_batch_stride, 0, A_multi_stride,
                             B, ldb, B_multi_stride,
                             C, C_batch_stride, 0, C_multi_stride);
    }

    unsigned int get_window_size() const override {
        return _subgemm->get_window_size();
    }

    void set_nthreads(int nthreads) override {
        _subgemm->set_nthreads(nthreads);
    }

    void execute(unsigned int start, unsigned int end, int threadid) override {
        _subgemm->execute(start, end, threadid);
    }

    size_t get_working_size() const override {
        return _subgemm->get_working_size();
    }

    void set_working_space(void *space) override {
        _subgemm->set_working_space(space);
    }

    bool B_is_pretransposed() const override {
        return _subgemm->B_is_pretransposed();
    }

    bool B_pretranspose_required() const override {
        return _subgemm->B_pretranspose_required();
    }

    size_t get_B_pretransposed_array_size() const override {
        return _subgemm->get_B_pretransposed_array_size();
    }

    void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride) override {
        _subgemm->pretranspose_B_array(buffer, B, ldb, B_multi_stride);
    }

    void set_pretransposed_B_data(void *buffer) override {
        _subgemm->set_pretransposed_B_data(buffer);
    }
};

} // namespace arm_gemm
