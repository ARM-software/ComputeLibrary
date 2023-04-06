/*
 * Copyright (c) 2017-2021,2023 Arm Limited.
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

#include "convolution_parameters.hpp"
#include "ndrange.hpp"

#include <cstddef>

namespace arm_gemm
{
// Avoid circular dependency with arm_gemm.hpp
struct GemmConfig;

// Abstract class for the GEMM/GEMV functions.
//
// GEMM implementations may be "native" (never require any input
// permutation), "pretransposed" (require permutation up-front) or require
// working space (permute as they go along).  This interface should support
// all of them.

// The real GemmCommon class is templated based on the operand and return
// type.  This is an interface class which is independent of those types.
class IGemmCommon
{
public:
    /* Pass in the pointers to the arrays to be operated on and their
     * strides.  This "generic" version uses void *s, the preferred version
     * is the one provided by templated GemmCommon (below) which takes
     * appropriately typed pointers.  If B is pretransposed (see below) then
     * the settings for B here are ignored.
     */
    virtual void set_arrays_generic(const void *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                                    const void *B, const int ldb, /* batches share B */ const int B_multi_stride,
                                    void *C, const int ldc, const int C_batch_stride, const int C_multi_stride,
                                    const void *bias, /* no row or batch stride needed */ const int bias_multi_stride) = 0;

    /** @returns an ndrange containing ranges of the compute space which can be
     * broken up and parallelised over
     */
    virtual ndrange_t get_window_size() const = 0;

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
    virtual void set_nthreads(int) {};

    /* Whether this GEMM can be dynamically scheduled or not. */
    virtual bool supports_dynamic_scheduling() const
    {
        return false;
    }

    /** Main execute member fucntion
     * @param [in] work_range     specifies the range of work we want to be computed, total range defined by get_window_size()
     * @param [in] thread_locator where are we inside of the thread space
     * @param [in] threadid       a unique threadid
     */
    virtual void execute(const ndcoord_t &work_range, const ndcoord_t &thread_locator, int threadid) = 0;

    /*** Working space interface (optional) ***/
    /* Total number of bytes of temporary working space needed.  If zero, it's not necessary to call set_working_space(). */
    virtual size_t get_working_size() const
    {
        return 0;
    }
    /* Provide working space buffer - the void * passed in must remain allocated for the duration of any execute calls. */
    virtual void set_working_space(void *) {};

    /*** "Pretransposed" interface (optional) ***/
    /* Is this object set up for pretranspose?  If so, pretranspose_array() needs to be called before execute(); */
    virtual bool B_is_pretransposed() const
    {
        return false;
    }
    /* Does pretranspose still need to be done? */
    virtual bool B_pretranspose_required() const
    {
        return false;
    }
    /* Total number of bytes of space needed for pretransposed arrays. */
    virtual size_t get_B_pretransposed_array_size() const
    {
        return 0;
    }
    /* Amount of work for the threaded cases */
    virtual size_t get_B_pretranspose_window_size() const
    {
        return 1;
    }
    /* Perform pretranspose - arguments are output, input, input row stride and input multi stride. */
    /* The "real" version of this depends on the templated operand type (see below).  */
    virtual void pretranspose_B_array_generic(void *, const void *, const int, const int) = 0;
    /* Threaded version with window start/end parameters */
    virtual void pretranspose_B_array_part_generic(void *, const void *, const int, const int, const size_t, const size_t) = 0;

    /* Set pretransposed data - the void * passed in must previously have been passed to pretranspose_B_array() for the same or a similar GEMM. */
    virtual void set_pretransposed_B_data(void *)
    {
    }

    /*** "Quantized bias" interface (optional) ***/
    /* Set the bias vector for quantized GEMMs */
    virtual void set_quantized_bias(const int32_t *, size_t)
    {
    }

    /*** Indirect interface (optional) ***/
    /* Set the indirect table.  This comprises a number of values per kernel point, and a densely packed array of pointers,
     * multis * batches * kernel_points */
    virtual void set_indirect_parameters_generic(size_t, const void *const *const *)
    {
    }

    /*** Convolution interface (optional) ***/
    /* Set the convolution parameters. */
    virtual void set_convolution_parameters(ConvolutionParameters)
    {
    }

    /*** Introspection interface ***/
    /* Get the configuration of this GEMM */
    virtual GemmConfig get_config() = 0;

    // Destructor
    virtual ~IGemmCommon()
    {
    }
};

/* "Real" GemmCommon class which is templated on the operand and return types.
 *
 * In addition to correctly typed versions of the functions that operate on
 * operand and return data, this class provides a default implementation of
 * 'set_arrays' to capture the provided arguments in protected class
 * members, as essentially any implementation will need these.
 */
template <typename To, typename Tr>
class GemmCommon : public IGemmCommon
{
protected:
    const To *_Aptr              = nullptr;
    int       _lda               = 0;
    int       _A_batch_stride    = 0;
    int       _A_multi_stride    = 0;
    const To *_Bptr              = nullptr;
    int       _ldb               = 0;
    int       _B_multi_stride    = 0;
    Tr       *_Cptr              = nullptr;
    int       _ldc               = 0;
    int       _C_batch_stride    = 0;
    int       _C_multi_stride    = 0;
    const Tr *_bias              = nullptr;
    int       _bias_multi_stride = 0;

public:
    /* Pass in the pointers to the arrays to be operated on and their
     * strides (templated version with appropriate types). */
    virtual void set_arrays(const To *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                            const To *B, const int ldb, /* batches share B */ const int B_multi_stride,
                            Tr *C, const int ldc, const int C_batch_stride, const int C_multi_stride,
                            const Tr *bias, /* no row or batch stride needed */ const int bias_multi_stride)
    {
        _Aptr              = A;
        _lda               = lda;
        _A_batch_stride    = A_batch_stride;
        _A_multi_stride    = A_multi_stride;
        _Bptr              = B;
        _ldb               = ldb;
        _B_multi_stride    = B_multi_stride;
        _Cptr              = C;
        _ldc               = ldc;
        _C_batch_stride    = C_batch_stride;
        _C_multi_stride    = C_multi_stride;
        _bias              = bias;
        _bias_multi_stride = bias_multi_stride;
    }

    /* Implementation of the void * overload which casts its arguments to the appropriate type. */
    void set_arrays_generic(const void *A, const int lda, const int A_batch_stride, const int A_multi_stride,
                            const void *B, const int ldb, /* batches share B */ const int B_multi_stride,
                            void *C, const int ldc, const int C_batch_stride, const int C_multi_stride,
                            const void *bias, /* no row or batch stride needed */ const int bias_multi_stride) override
    {
        set_arrays(static_cast<const To *>(A), lda, A_batch_stride, A_multi_stride,
                   static_cast<const To *>(B), ldb, B_multi_stride,
                   static_cast<Tr *>(C), ldc, C_batch_stride, C_multi_stride,
                   static_cast<const Tr *>(bias), bias_multi_stride);
    }

    /*** "Pretransposed" interface ***/

    /* Compute col sums over all columns */
    virtual void requantize_bias(void *, const To *, const int, const int) {};

    /* Perform pretranspose - the void * passed in must remain allocated for the duration of any execute calls. */
    /* Arguments are: output buffer pointer, source pointer, source row stride, source multi stride */
    virtual void pretranspose_B_array(void *, const To *, const int, const int) {};

    /* Implementation of the void * overload which casts its arguments to the appropriate type. */
    void pretranspose_B_array_generic(void *out, const void *in, const int row_stride, const int multi_stride) override
    {
        pretranspose_B_array(out, static_cast<const To *>(in), row_stride, multi_stride);
    }

    /* Threaded versions of the above.
     * The fallback/backwards compatible version of the threaded interface exposes a window size of 1 and
     * just calls the non-threaded functions to do the work.  This is valid as with window size of 1 the only
     * legal values for start and end are 0 and 1 respectively. */
    virtual void pretranspose_B_array_part(void *out, const To *in, const int row_stride, const int multi_stride, size_t, size_t)
    {
        pretranspose_B_array(out, in, row_stride, multi_stride);
    };

    void pretranspose_B_array_part_generic(void *out, const void *in, const int row_stride, const int multi_stride, size_t start, size_t end) override
    {
        pretranspose_B_array_part(out, static_cast<const To *>(in), row_stride, multi_stride, start, end);
    }

    /*** Indirect interface ***/
    virtual void set_indirect_parameters(size_t, const To *const *const *)
    {
    }

    void set_indirect_parameters_generic(size_t sz, const void *const *const *ptr) override
    {
        set_indirect_parameters(sz, reinterpret_cast<const To *const *const *>(ptr));
    }
};

} // namespace arm_gemm
