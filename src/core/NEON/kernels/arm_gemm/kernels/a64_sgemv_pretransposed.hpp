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

#ifdef __aarch64__

namespace arm_gemm
{
// Actual kernel implementations
void a64_sgemv_pretransposed(const float *, int, const float *, float *, float, int, int);

// Pretransposed SGEMV strategy class.
class sgemv_pretransposed
{
public:
    typedef float operand_type;
    typedef float result_type;

    typedef void (*kern_type)(const float *, int, const float *, float *, float, int, int);

    /* Describes the data layout for matrix (A) input */

    /* Note that often GEMV is expressed as a GEMM with M=1, i.e.  A is the
     * (row) vector and B is the matrix, but the standard GEMV arrangement
     * is matrix A times (column) vector X.  "A_transpose" is expressed in
     * terms of this standard arrangement, so if the A matrix is in fact the
     * B matrix from a GEMM call, the sense of the transpose needs to be
     * reversed.  */
    static const int  A_interleave = 32;
    static const int  A_block      = 1;
    static const bool A_transpose  = false;

    /* Kernel blocking parameters */
    static const int out_width = 32;
    static const int k_unroll  = 1;

    kern_type kernel = a64_sgemv_pretransposed;

    sgemv_pretransposed(const CPUInfo *ci)
    {
    }
};

} // namespace arm_gemm

#endif // __aarch64__
