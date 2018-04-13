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

#include <cstddef>

namespace arm_gemm {

// Abstract class for the GEMM/GEMV functions.
//
// GEMM implementations may be "native" (never require any input
// permutation), "pretransposed" (require permutation up-front) or require
// working space (permute as they go along).  This interface should support
// all of them.

template<typename To, typename Tr>
class GemmCommon {
protected:
    const To *_Aptr=nullptr;
    int _lda=0;
    int _A_batch_stride=0;
    int _A_multi_stride=0;
    const To *_Bptr=nullptr;
    int _ldb=0;
    int _B_multi_stride=0;
    Tr *_Cptr=nullptr;
    int _ldc=0;
    int _C_batch_stride=0;
    int _C_multi_stride=0;

public:
    /* Pass in the pointers to the arrays to be operated on and their
     * strides.  This has a default implementation that just captures them
     * all in protected members.  If B is pretransposed (see below) then the
     * settings for B here are ignored.  */
    virtual void set_arrays(const To *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                            const To *B, const int ldb, /* batches share B */     const int B_multi_stride,
                            Tr *C, const int ldc, const int C_batch_stride, const int C_multi_stride) {
        _Aptr = A;
        _lda = lda;
        _A_batch_stride = A_batch_stride;
        _A_multi_stride = A_multi_stride;
        _Bptr = B;
        _ldb = ldb;
        _B_multi_stride = B_multi_stride;
        _Cptr = C;
        _ldc = ldc;
        _C_batch_stride = C_batch_stride;
        _C_multi_stride = C_multi_stride;
    }

    /* For threading, we divide the work into some number of units and work
     * out internally what unit corresponds to what work.  This returns the
     * total number of units.  */
    virtual unsigned int get_window_size() const = 0;

    /* The maximum thread count is specified when the GEMM is created.  Some
     * implementations need to know how many threads will actually run in
     * order to work properly.
     *
     * In some cases, after creating the GEMM the number of threads needs to
     * be reduced (e.g. not enough work to split across threads).  This
     * method allows the number of actual threads to be run to be set (must
     * be equal or lower).
     *
     * This has an empty default implementation, as GEMMs which don't care
     * about thread count can safely ignore this.
     */
    virtual void set_nthreads(int nthreads) { };

    /* Actually do the work.  Provide a threadid to index any per-thread
     * buffers, and a start/end range to indicate which work to do.  */
    virtual void execute(unsigned int start, unsigned int end, int threadid) = 0;

    /*** Working space interface (optional) ***/
    /* Total number of bytes of temporary working space needed.  If zero, it's not necessary to call set_working_space(). */
    virtual size_t get_working_size() const { return 0; }
    /* Provide working space buffer - the void * passed in must remain allocated for the duration of any execute calls. */
    virtual void set_working_space(void *) { };

    /*** "Pretransposed" interface (optional) ***/
    /* Is this object set up for pretranspose?  If so, pretranspose_array() needs to be called before execute(); */
    virtual bool B_is_pretransposed() const { return false; }
    /* Does pretranspose still need to be done? */
    virtual bool B_pretranspose_required() const { return false; }
    /* Total number of bytes of space needed for pretransposed arrays. */
    virtual size_t get_B_pretransposed_array_size() const { return 0; }
    /* Perform pretranspose - the void * passed in must remain allocated for the duration of any execute calls. */
    virtual void pretranspose_B_array(void *buffer, const To *B, const int ldb, const int B_multi_stride) { };
    /* Set pretransposed data - the void * passed in must previously have been passed to pretranspose_B_array() for the same or a similar GEMM. */
    virtual void set_pretransposed_B_data(void *buffer) { }

    // Destructor
    virtual ~GemmCommon() { }
};

} // namespace arm_gemm
