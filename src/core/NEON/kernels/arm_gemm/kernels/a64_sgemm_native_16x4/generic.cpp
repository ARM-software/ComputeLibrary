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
#ifdef __aarch64__

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <arm_neon.h>

namespace arm_gemm
{
void a64_sgemm_native_16x4(const float *A, int lda, const float *B, int ldb, float *C, int ldc, float beta, int M, int N, int K)
{
    const int oddk    = ((K % 8) >= 4) ? 1 : 0;
    const int beta0   = (beta == 0.0f) ? 1 : 0;
    const int oddones = (K % 4);

    float dummy_buffer[16];

    std::memset(dummy_buffer, 0, sizeof(dummy_buffer));

    /* For now, very naive with no blocking */
    for (int y=0; y<M; y+=4) {
        const int activerows = std::min(M-y, 4);

        const float * const a_ptr0_base = A + (y * lda);
        const float * const a_ptr1_base = (activerows > 1) ? (a_ptr0_base + lda) : dummy_buffer;
        const float * const a_ptr2_base = (activerows > 2) ? (a_ptr1_base + lda) : dummy_buffer;
        const float * const a_ptr3_base = (activerows > 3) ? (a_ptr2_base + lda) : dummy_buffer;

        const unsigned long a_incr1 = (activerows > 1) ? 32 : 0;
        const unsigned long a_incr2 = (activerows > 2) ? 32 : 0;
        const unsigned long a_incr3 = (activerows > 3) ? 32 : 0;

        float *c_ptr0 = C + (y * ldc);
        float *c_ptr1 = (activerows > 1) ? c_ptr0 + ldc : dummy_buffer;
        float *c_ptr2 = (activerows > 2) ? c_ptr1 + ldc : dummy_buffer;
        float *c_ptr3 = (activerows > 3) ? c_ptr2 + ldc : dummy_buffer;

        for (int x0=0; x0<N; x0+=16) {
            const float *a_ptr0 = a_ptr0_base;
            const float *a_ptr1 = a_ptr1_base;
            const float *a_ptr2 = a_ptr2_base;
            const float *a_ptr3 = a_ptr3_base;

            const float *b_ptr = B + x0;

            int loops = ((K + 4) / 8) - 1;
            int odds  = oddones;

            size_t ldbb = ldb * sizeof(float);

            __asm __volatile(
                "a0   .req v0\n"
                "a1   .req v1\n"
                "a2   .req v2\n"
                "a3   .req v3\n"
                "a0a  .req v4\n"
                "a1a  .req v5\n"
                "a2a  .req v6\n"
                "a3a  .req v7\n"
                "bb0  .req v8\n"
                "bb1  .req v9\n"
                "bb2  .req v10\n"
                "bb3  .req v11\n"
                "b0a  .req v12\n"
                "b1a  .req v13\n"
                "b2a  .req v14\n"
                "b3a  .req v15\n"

                "a0q  .req q0\n"
                "a1q  .req q1\n"
                "a2q  .req q2\n"
                "a3q  .req q3\n"
                "a0aq .req q4\n"
                "a1aq .req q5\n"
                "a2aq .req q6\n"
                "a3aq .req q7\n"
                "b0q  .req q8\n"
                "b1q  .req q9\n"
                "b2q  .req q10\n"
                "b3q  .req q11\n"
                "b0aq .req q12\n"
                "b1aq .req q13\n"
                "b2aq .req q14\n"
                "b3aq .req q15\n"

                "movi    v16.4s, #0x0\n"
                "ldr    a0q, [%[a_ptr0]]\n"
                "movi    v17.4s, #0x0\n"
                "ldr    b0q, [%[b_ptr]]\n"
                "movi    v18.4s, #0x0\n"
                "ldr    b1q, [%[b_ptr], #16]\n"
                "movi    v19.4s, #0x0\n"
                "ldr    b2q, [%[b_ptr], #32]\n"
                "movi    v20.4s, #0x0\n"
                "ldr    b3q, [%[b_ptr], #48]\n"
                "movi    v21.4s, #0x0\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "ldr    a1q, [%[a_ptr1]]\n"
                "movi    v22.4s, #0x0\n"
                "ldr    a2q, [%[a_ptr2]]\n"
                "movi    v23.4s, #0x0\n"
                "ldr    a3q, [%[a_ptr3]]\n"
                "movi    v24.4s, #0x0\n"
                "ldr    b0aq, [%[b_ptr]]\n"
                "movi    v25.4s, #0x0\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"
                "movi    v26.4s, #0x0\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"
                "cbz    %w[beta0], 5f\n"
                "movi    v27.4s, #0x0\n"
                "movi    v28.4s, #0x0\n"
                "movi    v29.4s, #0x0\n"
                "movi    v30.4s, #0x0\n"
                "movi    v31.4s, #0x0\n"

                // Skip if no complete loops.
                "cbz    %w[loops], 4f\n"
                "b    1f\n"

                // If beta is non-zero, need to load and multiply by beta
                "5:\n"
                "ld1r    {v4.4s}, [%[betaptr]]\n"
                "ldr    q16, [%[c_ptr0]]\n"
                "ldr    q17, [%[c_ptr0], #16]\n"
                "ldr    q18, [%[c_ptr0], #32]\n"
                "ldr    q19, [%[c_ptr0], #48]\n"

                "ldr    q20, [%[c_ptr1]]\n"
                "fmul    v16.4s, v16.4s, v4.4s\n"
                "ldr    q21, [%[c_ptr1], #16]\n"
                "fmul    v17.4s, v17.4s, v4.4s\n"
                "ldr    q22, [%[c_ptr1], #32]\n"
                "fmul    v18.4s, v18.4s, v4.4s\n"
                "ldr    q23, [%[c_ptr1], #48]\n"
                "fmul    v19.4s, v19.4s, v4.4s\n"

                "ldr    q24, [%[c_ptr2]]\n"
                "fmul    v20.4s, v20.4s, v4.4s\n"
                "ldr    q25, [%[c_ptr2], #16]\n"
                "fmul    v21.4s, v21.4s, v4.4s\n"
                "ldr    q26, [%[c_ptr2], #32]\n"
                "fmul    v22.4s, v22.4s, v4.4s\n"
                "ldr    q27, [%[c_ptr2], #48]\n"
                "fmul    v23.4s, v23.4s, v4.4s\n"

                "ldr    q28, [%[c_ptr3]]\n"
                "fmul    v24.4s, v24.4s, v4.4s\n"
                "ldr    q29, [%[c_ptr3], #16]\n"
                "fmul    v25.4s, v25.4s, v4.4s\n"
                "ldr    q30, [%[c_ptr3], #32]\n"
                "fmul    v26.4s, v26.4s, v4.4s\n"
                "ldr    q31, [%[c_ptr3], #48]\n"
                "fmul    v27.4s, v27.4s, v4.4s\n"

                "fmul    v28.4s, v28.4s, v4.4s\n"
                "fmul    v29.4s, v29.4s, v4.4s\n"
                "fmul    v30.4s, v30.4s, v4.4s\n"
                "fmul    v31.4s, v31.4s, v4.4s\n"

                "cbz    %w[loops], 4f\n"

                "1:\n"
                // Unroll 0
                "fmla    v16.4s, bb0.4s, a0.s[0]\n"
                "fmla    v20.4s, bb0.4s, a1.s[0]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"
                "fmla    v24.4s, bb0.4s, a2.s[0]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v28.4s, bb0.4s, a3.s[0]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0.s[0]\n"
                "fmla    v21.4s, bb1.4s, a1.s[0]\n"
                "ldr    a0aq, [%[a_ptr0], #16]\n"
                "fmla    v25.4s, bb1.4s, a2.s[0]\n"
                "fmla    v29.4s, bb1.4s, a3.s[0]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0.s[0]\n"
                "fmla    v22.4s, bb2.4s, a1.s[0]\n"
                "ldr    a1aq, [%[a_ptr1], #16]\n"
                "fmla    v26.4s, bb2.4s, a2.s[0]\n"
                "fmla    v30.4s, bb2.4s, a3.s[0]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0.s[0]\n"
                "fmla    v23.4s, bb3.4s, a1.s[0]\n"
                "ldr    a2aq, [%[a_ptr2], #16]\n"
                "fmla    v27.4s, bb3.4s, a2.s[0]\n"
                "fmla    v31.4s, bb3.4s, a3.s[0]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 1
                "fmla    v16.4s, b0a.4s, a0.s[1]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v20.4s, b0a.4s, a1.s[1]\n"
                "ldr    a3aq, [%[a_ptr3], #16]\n"
                "fmla    v24.4s, b0a.4s, a2.s[1]\n"
                "fmla    v28.4s, b0a.4s, a3.s[1]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0.s[1]\n"
                "fmla    v21.4s, b1a.4s, a1.s[1]\n"
                "subs    %w[loops], %w[loops], #1\n"
                "fmla    v25.4s, b1a.4s, a2.s[1]\n"
                "fmla    v29.4s, b1a.4s, a3.s[1]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0.s[1]\n"
                "fmla    v22.4s, b2a.4s, a1.s[1]\n"
                "fmla    v26.4s, b2a.4s, a2.s[1]\n"
                "fmla    v30.4s, b2a.4s, a3.s[1]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0.s[1]\n"
                "fmla    v23.4s, b3a.4s, a1.s[1]\n"
                "fmla    v27.4s, b3a.4s, a2.s[1]\n"
                "fmla    v31.4s, b3a.4s, a3.s[1]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 2
                "fmla	v16.4s, bb0.4s, a0.s[2]\n"
                "fmla	v20.4s, bb0.4s, a1.s[2]\n"
                "add	%[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla	v24.4s, bb0.4s, a2.s[2]\n"
                "fmla	v28.4s, bb0.4s, a3.s[2]\n"
                "ldr	b0q, [%[b_ptr]]\n"

                "fmla	v17.4s, bb1.4s, a0.s[2]\n"
                "add	%[a_ptr0], %[a_ptr0], #32\n"
                "fmla	v21.4s, bb1.4s, a1.s[2]\n"
                "add	%[a_ptr1], %[a_ptr1], %[a_incr1]\n"
                "fmla	v25.4s, bb1.4s, a2.s[2]\n"
                "add	%[a_ptr2], %[a_ptr2], %[a_incr2]\n"
                "fmla	v29.4s, bb1.4s, a3.s[2]\n"
                "ldr	b1q, [%[b_ptr], #16]\n"

                "fmla	v18.4s, bb2.4s, a0.s[2]\n"
                "add	%[a_ptr3], %[a_ptr3], %[a_incr3]\n"
                "fmla	v22.4s, bb2.4s, a1.s[2]\n"
                "fmla	v26.4s, bb2.4s, a2.s[2]\n"
                "fmla	v30.4s, bb2.4s, a3.s[2]\n"
                "ldr	b2q, [%[b_ptr], #32]\n"

                "fmla	v19.4s, bb3.4s, a0.s[2]\n"
                "fmla	v23.4s, bb3.4s, a1.s[2]\n"
                "fmla	v27.4s, bb3.4s, a2.s[2]\n"
                "fmla	v31.4s, bb3.4s, a3.s[2]\n"
                "ldr	b3q, [%[b_ptr], #48]\n"

                // Unroll 3
                "fmla    v16.4s, b0a.4s, a0.s[3]\n"
                "fmla    v20.4s, b0a.4s, a1.s[3]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, b0a.4s, a2.s[3]\n"
                "fmla    v28.4s, b0a.4s, a3.s[3]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0.s[3]\n"
                "fmla    v21.4s, b1a.4s, a1.s[3]\n"
                "fmla    v25.4s, b1a.4s, a2.s[3]\n"
                "fmla    v29.4s, b1a.4s, a3.s[3]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0.s[3]\n"
                "fmla    v22.4s, b2a.4s, a1.s[3]\n"
                "fmla    v26.4s, b2a.4s, a2.s[3]\n"
                "fmla    v30.4s, b2a.4s, a3.s[3]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0.s[3]\n"
                "fmla    v23.4s, b3a.4s, a1.s[3]\n"
                "ldr    a0q, [%[a_ptr0]]\n"
                "fmla    v27.4s, b3a.4s, a2.s[3]\n"
                "fmla    v31.4s, b3a.4s, a3.s[3]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 4
                "fmla    v16.4s, bb0.4s, a0a.s[0]\n"
                "fmla    v20.4s, bb0.4s, a1a.s[0]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, bb0.4s, a2a.s[0]\n"
                "fmla    v28.4s, bb0.4s, a3a.s[0]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0a.s[0]\n"
                "fmla    v21.4s, bb1.4s, a1a.s[0]\n"
                "ldr    a1q, [%[a_ptr1]]\n"
                "fmla    v25.4s, bb1.4s, a2a.s[0]\n"
                "fmla    v29.4s, bb1.4s, a3a.s[0]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0a.s[0]\n"
                "fmla    v22.4s, bb2.4s, a1a.s[0]\n"
                "ldr    a2q, [%[a_ptr2]]\n"
                "fmla    v26.4s, bb2.4s, a2a.s[0]\n"
                "fmla    v30.4s, bb2.4s, a3a.s[0]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0a.s[0]\n"
                "fmla    v23.4s, bb3.4s, a1a.s[0]\n"
                "ldr    a3q, [%[a_ptr3]]\n"
                "fmla    v27.4s, bb3.4s, a2a.s[0]\n"
                "fmla    v31.4s, bb3.4s, a3a.s[0]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 5
                "fmla    v16.4s, b0a.4s, a0a.s[1]\n"
                "fmla    v20.4s, b0a.4s, a1a.s[1]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, b0a.4s, a2a.s[1]\n"
                "fmla    v28.4s, b0a.4s, a3a.s[1]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0a.s[1]\n"
                "fmla    v21.4s, b1a.4s, a1a.s[1]\n"
                "fmla    v25.4s, b1a.4s, a2a.s[1]\n"
                "fmla    v29.4s, b1a.4s, a3a.s[1]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0a.s[1]\n"
                "fmla    v22.4s, b2a.4s, a1a.s[1]\n"
                "fmla    v26.4s, b2a.4s, a2a.s[1]\n"
                "fmla    v30.4s, b2a.4s, a3a.s[1]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0a.s[1]\n"
                "fmla    v23.4s, b3a.4s, a1a.s[1]\n"
                "fmla    v27.4s, b3a.4s, a2a.s[1]\n"
                "fmla    v31.4s, b3a.4s, a3a.s[1]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 6
                "fmla    v16.4s, bb0.4s, a0a.s[2]\n"
                "fmla    v20.4s, bb0.4s, a1a.s[2]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, bb0.4s, a2a.s[2]\n"
                "fmla    v28.4s, bb0.4s, a3a.s[2]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0a.s[2]\n"
                "fmla    v21.4s, bb1.4s, a1a.s[2]\n"
                "fmla    v25.4s, bb1.4s, a2a.s[2]\n"
                "fmla    v29.4s, bb1.4s, a3a.s[2]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0a.s[2]\n"
                "fmla    v22.4s, bb2.4s, a1a.s[2]\n"
                "fmla    v26.4s, bb2.4s, a2a.s[2]\n"
                "fmla    v30.4s, bb2.4s, a3a.s[2]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0a.s[2]\n"
                "fmla    v23.4s, bb3.4s, a1a.s[2]\n"
                "fmla    v27.4s, bb3.4s, a2a.s[2]\n"
                "fmla    v31.4s, bb3.4s, a3a.s[2]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 7
                "fmla    v16.4s, b0a.4s, a0a.s[3]\n"
                "fmla    v20.4s, b0a.4s, a1a.s[3]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, b0a.4s, a2a.s[3]\n"
                "fmla    v28.4s, b0a.4s, a3a.s[3]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0a.s[3]\n"
                "fmla    v21.4s, b1a.4s, a1a.s[3]\n"
                "fmla    v25.4s, b1a.4s, a2a.s[3]\n"
                "fmla    v29.4s, b1a.4s, a3a.s[3]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0a.s[3]\n"
                "fmla    v22.4s, b2a.4s, a1a.s[3]\n"
                "fmla    v26.4s, b2a.4s, a2a.s[3]\n"
                "fmla    v30.4s, b2a.4s, a3a.s[3]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0a.s[3]\n"
                "fmla    v23.4s, b3a.4s, a1a.s[3]\n"
                "fmla    v27.4s, b3a.4s, a2a.s[3]\n"
                "fmla    v31.4s, b3a.4s, a3a.s[3]\n"
                "bne    1b\n"

                // Skip to here
                "4:\n"

                // Detached final iteration
                // Unroll 0
                "fmla    v16.4s, bb0.4s, a0.s[0]\n"
                "fmla    v20.4s, bb0.4s, a1.s[0]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"
                "fmla    v24.4s, bb0.4s, a2.s[0]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v28.4s, bb0.4s, a3.s[0]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0.s[0]\n"
                "cbnz    %w[oddk], 2f\n" // Deal with odd K before we load a0a
                "fmla    v21.4s, bb1.4s, a1.s[0]\n"
                "ldr    a0aq, [%[a_ptr0], #16]\n"
                "fmla    v25.4s, bb1.4s, a2.s[0]\n"
                "fmla    v29.4s, bb1.4s, a3.s[0]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0.s[0]\n"
                "fmla    v22.4s, bb2.4s, a1.s[0]\n"
                "ldr    a1aq, [%[a_ptr1], #16]\n"
                "fmla    v26.4s, bb2.4s, a2.s[0]\n"
                "fmla    v30.4s, bb2.4s, a3.s[0]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0.s[0]\n"
                "fmla    v23.4s, bb3.4s, a1.s[0]\n"
                "ldr    a2aq, [%[a_ptr2], #16]\n"
                "fmla    v27.4s, bb3.4s, a2.s[0]\n"
                "fmla    v31.4s, bb3.4s, a3.s[0]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 1
                "fmla	v16.4s, b0a.4s, a0.s[1]\n"
                "add	%[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla	v20.4s, b0a.4s, a1.s[1]\n"
                "ldr	a3aq, [%[a_ptr3], #16]\n"
                "fmla	v24.4s, b0a.4s, a2.s[1]\n"
                "fmla	v28.4s, b0a.4s, a3.s[1]\n"
                "ldr	b0aq, [%[b_ptr]]\n"

                "fmla	v17.4s, b1a.4s, a0.s[1]\n"
                "add	%[a_ptr0], %[a_ptr0], #32\n"
                "fmla	v21.4s, b1a.4s, a1.s[1]\n"
                "add	%[a_ptr1], %[a_ptr1], %[a_incr1]\n"
                "fmla	v25.4s, b1a.4s, a2.s[1]\n"
                "add	%[a_ptr2], %[a_ptr2], %[a_incr2]\n"
                "fmla	v29.4s, b1a.4s, a3.s[1]\n"
                "ldr	b1aq, [%[b_ptr], #16]\n"

                "fmla	v18.4s, b2a.4s, a0.s[1]\n"
                "fmla	v22.4s, b2a.4s, a1.s[1]\n"
                "add	%[a_ptr3], %[a_ptr3], %[a_incr3]\n"
                "fmla	v26.4s, b2a.4s, a2.s[1]\n"
                "fmla	v30.4s, b2a.4s, a3.s[1]\n"
                "ldr	b2aq, [%[b_ptr], #32]\n"

                "fmla	v19.4s, b3a.4s, a0.s[1]\n"
                "fmla	v23.4s, b3a.4s, a1.s[1]\n"
                "fmla	v27.4s, b3a.4s, a2.s[1]\n"
                "fmla	v31.4s, b3a.4s, a3.s[1]\n"
                "ldr	b3aq, [%[b_ptr], #48]\n"

                // Unroll 2
                "fmla    v16.4s, bb0.4s, a0.s[2]\n"
                "fmla    v20.4s, bb0.4s, a1.s[2]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, bb0.4s, a2.s[2]\n"
                "fmla    v28.4s, bb0.4s, a3.s[2]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0.s[2]\n"
                "fmla    v21.4s, bb1.4s, a1.s[2]\n"
                "fmla    v25.4s, bb1.4s, a2.s[2]\n"
                "fmla    v29.4s, bb1.4s, a3.s[2]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0.s[2]\n"
                "fmla    v22.4s, bb2.4s, a1.s[2]\n"
                "fmla    v26.4s, bb2.4s, a2.s[2]\n"
                "fmla    v30.4s, bb2.4s, a3.s[2]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0.s[2]\n"
                "fmla    v23.4s, bb3.4s, a1.s[2]\n"
                "fmla    v27.4s, bb3.4s, a2.s[2]\n"
                "fmla    v31.4s, bb3.4s, a3.s[2]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 3
                "fmla    v16.4s, b0a.4s, a0.s[3]\n"
                "fmla    v20.4s, b0a.4s, a1.s[3]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, b0a.4s, a2.s[3]\n"
                "fmla    v28.4s, b0a.4s, a3.s[3]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0.s[3]\n"
                "fmla    v21.4s, b1a.4s, a1.s[3]\n"
                "fmla    v25.4s, b1a.4s, a2.s[3]\n"
                "fmla    v29.4s, b1a.4s, a3.s[3]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0.s[3]\n"
                "fmla    v22.4s, b2a.4s, a1.s[3]\n"
                "fmla    v26.4s, b2a.4s, a2.s[3]\n"
                "fmla    v30.4s, b2a.4s, a3.s[3]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0.s[3]\n"
                "fmla    v23.4s, b3a.4s, a1.s[3]\n"
                "fmla    v27.4s, b3a.4s, a2.s[3]\n"
                "fmla    v31.4s, b3a.4s, a3.s[3]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 4
                "fmla    v16.4s, bb0.4s, a0a.s[0]\n"
                "fmla    v20.4s, bb0.4s, a1a.s[0]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, bb0.4s, a2a.s[0]\n"
                "fmla    v28.4s, bb0.4s, a3a.s[0]\n"
                "ldr    b0q, [%[b_ptr]]\n"

                "fmla    v17.4s, bb1.4s, a0a.s[0]\n"
                "fmla    v21.4s, bb1.4s, a1a.s[0]\n"
                "fmla    v25.4s, bb1.4s, a2a.s[0]\n"
                "fmla    v29.4s, bb1.4s, a3a.s[0]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0a.s[0]\n"
                "fmla    v22.4s, bb2.4s, a1a.s[0]\n"
                "fmla    v26.4s, bb2.4s, a2a.s[0]\n"
                "fmla    v30.4s, bb2.4s, a3a.s[0]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0a.s[0]\n"
                "fmla    v23.4s, bb3.4s, a1a.s[0]\n"
                "fmla    v27.4s, bb3.4s, a2a.s[0]\n"
                "fmla    v31.4s, bb3.4s, a3a.s[0]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 5
                "fmla    v16.4s, b0a.4s, a0a.s[1]\n"
                "fmla    v20.4s, b0a.4s, a1a.s[1]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, b0a.4s, a2a.s[1]\n"
                "fmla    v28.4s, b0a.4s, a3a.s[1]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0a.s[1]\n"
                "fmla    v21.4s, b1a.4s, a1a.s[1]\n"
                "fmla    v25.4s, b1a.4s, a2a.s[1]\n"
                "fmla    v29.4s, b1a.4s, a3a.s[1]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0a.s[1]\n"
                "fmla    v22.4s, b2a.4s, a1a.s[1]\n"
                "fmla    v26.4s, b2a.4s, a2a.s[1]\n"
                "fmla    v30.4s, b2a.4s, a3a.s[1]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0a.s[1]\n"
                "fmla    v23.4s, b3a.4s, a1a.s[1]\n"
                "fmla    v27.4s, b3a.4s, a2a.s[1]\n"
                "fmla    v31.4s, b3a.4s, a3a.s[1]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 6
                "fmla    v16.4s, bb0.4s, a0a.s[2]\n"
                "fmla    v20.4s, bb0.4s, a1a.s[2]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v24.4s, bb0.4s, a2a.s[2]\n"
                "fmla    v28.4s, bb0.4s, a3a.s[2]\n"

                "fmla    v17.4s, bb1.4s, a0a.s[2]\n"
                "fmla    v21.4s, bb1.4s, a1a.s[2]\n"
                "fmla    v25.4s, bb1.4s, a2a.s[2]\n"
                "fmla    v29.4s, bb1.4s, a3a.s[2]\n"

                "fmla    v18.4s, bb2.4s, a0a.s[2]\n"
                "fmla    v22.4s, bb2.4s, a1a.s[2]\n"
                "fmla    v26.4s, bb2.4s, a2a.s[2]\n"
                "fmla    v30.4s, bb2.4s, a3a.s[2]\n"

                "fmla    v19.4s, bb3.4s, a0a.s[2]\n"
                "fmla    v23.4s, bb3.4s, a1a.s[2]\n"
                "fmla    v27.4s, bb3.4s, a2a.s[2]\n"
                "fmla    v31.4s, bb3.4s, a3a.s[2]\n"

                // Unroll 7
                "fmla    v16.4s, b0a.4s, a0a.s[3]\n"
                "fmla    v17.4s, b1a.4s, a0a.s[3]\n"
                "fmla    v18.4s, b2a.4s, a0a.s[3]\n"
                "fmla    v19.4s, b3a.4s, a0a.s[3]\n"
                "cbnz    %w[odds], 6f\n"

                "fmla    v20.4s, b0a.4s, a1a.s[3]\n"
                "str    q16, [%[c_ptr0]]\n"
                "fmla    v21.4s, b1a.4s, a1a.s[3]\n"
                "str    q17, [%[c_ptr0], #16]\n"
                "fmla    v22.4s, b2a.4s, a1a.s[3]\n"
                "str    q18, [%[c_ptr0], #32]\n"
                "fmla    v23.4s, b3a.4s, a1a.s[3]\n"
                "str    q19, [%[c_ptr0], #48]\n"

                "fmla    v24.4s, b0a.4s, a2a.s[3]\n"
                "str    q20, [%[c_ptr1]]\n"
                "fmla    v25.4s, b1a.4s, a2a.s[3]\n"
                "str    q21, [%[c_ptr1], #16]\n"
                "fmla    v26.4s, b2a.4s, a2a.s[3]\n"
                "str    q22, [%[c_ptr1], #32]\n"
                "fmla    v27.4s, b3a.4s, a2a.s[3]\n"
                "str    q23, [%[c_ptr1], #48]\n"

                "fmla    v28.4s, b0a.4s, a3a.s[3]\n"
                "str    q24, [%[c_ptr2]]\n"
                "fmla    v29.4s, b1a.4s, a3a.s[3]\n"
                "str    q25, [%[c_ptr2], #16]\n"
                "fmla    v30.4s, b2a.4s, a3a.s[3]\n"
                "str    q26, [%[c_ptr2], #32]\n"
                "fmla    v31.4s, b3a.4s, a3a.s[3]\n"
                "str    q27, [%[c_ptr2], #48]\n"
                "b    3f\n"

                // Odd K case: Just do 4 more.
                "2:\n"
                "fmla    v21.4s, bb1.4s, a1.s[0]\n"
                "add    %[a_ptr0], %[a_ptr0], #16\n"
                "fmla    v25.4s, bb1.4s, a2.s[0]\n"
                "add    %[a_ptr1], %[a_ptr1], #16\n"
                "fmla    v29.4s, bb1.4s, a3.s[0]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v18.4s, bb2.4s, a0.s[0]\n"
                "add    %[a_ptr2], %[a_ptr2], #16\n"
                "fmla    v22.4s, bb2.4s, a1.s[0]\n"
                "add    %[a_ptr3], %[a_ptr3], #16\n"
                "fmla    v26.4s, bb2.4s, a2.s[0]\n"
                "fmla    v30.4s, bb2.4s, a3.s[0]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"

                "fmla    v19.4s, bb3.4s, a0.s[0]\n"
                "fmla    v23.4s, bb3.4s, a1.s[0]\n"
                "fmla    v27.4s, bb3.4s, a2.s[0]\n"
                "fmla    v31.4s, bb3.4s, a3.s[0]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                // Unroll 1
                "fmla    v16.4s, b0a.4s, a0.s[1]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v20.4s, b0a.4s, a1.s[1]\n"
                "fmla    v24.4s, b0a.4s, a2.s[1]\n"
                "fmla    v28.4s, b0a.4s, a3.s[1]\n"
                "ldr    b0aq, [%[b_ptr]]\n"

                "fmla    v17.4s, b1a.4s, a0.s[1]\n"
                "fmla    v21.4s, b1a.4s, a1.s[1]\n"
                "fmla    v25.4s, b1a.4s, a2.s[1]\n"
                "fmla    v29.4s, b1a.4s, a3.s[1]\n"
                "ldr    b1aq, [%[b_ptr], #16]\n"

                "fmla    v18.4s, b2a.4s, a0.s[1]\n"
                "fmla    v22.4s, b2a.4s, a1.s[1]\n"
                "fmla    v26.4s, b2a.4s, a2.s[1]\n"
                "fmla    v30.4s, b2a.4s, a3.s[1]\n"
                "ldr    b2aq, [%[b_ptr], #32]\n"

                "fmla    v19.4s, b3a.4s, a0.s[1]\n"
                "fmla    v23.4s, b3a.4s, a1.s[1]\n"
                "fmla    v27.4s, b3a.4s, a2.s[1]\n"
                "fmla    v31.4s, b3a.4s, a3.s[1]\n"
                "ldr    b3aq, [%[b_ptr], #48]\n"

                // Unroll 2
                "fmla    v16.4s, bb0.4s, a0.s[2]\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v20.4s, bb0.4s, a1.s[2]\n"
                "fmla    v24.4s, bb0.4s, a2.s[2]\n"
                "fmla    v28.4s, bb0.4s, a3.s[2]\n"

                "fmla    v17.4s, bb1.4s, a0.s[2]\n"
                "fmla    v21.4s, bb1.4s, a1.s[2]\n"
                "fmla    v25.4s, bb1.4s, a2.s[2]\n"
                "fmla    v29.4s, bb1.4s, a3.s[2]\n"

                "fmla    v18.4s, bb2.4s, a0.s[2]\n"
                "fmla    v22.4s, bb2.4s, a1.s[2]\n"
                "fmla    v26.4s, bb2.4s, a2.s[2]\n"
                "fmla    v30.4s, bb2.4s, a3.s[2]\n"

                "fmla    v19.4s, bb3.4s, a0.s[2]\n"
                "fmla    v23.4s, bb3.4s, a1.s[2]\n"
                "fmla    v27.4s, bb3.4s, a2.s[2]\n"
                "fmla    v31.4s, bb3.4s, a3.s[2]\n"

                // Unroll 3
                "fmla    v16.4s, b0a.4s, a0.s[3]\n"
                "fmla    v17.4s, b1a.4s, a0.s[3]\n"
                "fmla    v18.4s, b2a.4s, a0.s[3]\n"
                "fmla    v19.4s, b3a.4s, a0.s[3]\n"
                "cbnz    %w[odds], 7f\n"

                "fmla    v20.4s, b0a.4s, a1.s[3]\n"
                "str    q16, [%[c_ptr0]]\n"
                "fmla    v21.4s, b1a.4s, a1.s[3]\n"
                "str    q17, [%[c_ptr0], #16]\n"
                "fmla    v22.4s, b2a.4s, a1.s[3]\n"
                "str    q18, [%[c_ptr0], #32]\n"
                "fmla    v23.4s, b3a.4s, a1.s[3]\n"
                "str    q19, [%[c_ptr0], #48]\n"

                "fmla    v24.4s, b0a.4s, a2.s[3]\n"
                "str    q20, [%[c_ptr1]]\n"
                "fmla    v25.4s, b1a.4s, a2.s[3]\n"
                "str    q21, [%[c_ptr1], #16]\n"
                "fmla    v26.4s, b2a.4s, a2.s[3]\n"
                "str    q22, [%[c_ptr1], #32]\n"
                "fmla    v27.4s, b3a.4s, a2.s[3]\n"
                "str    q23, [%[c_ptr1], #48]\n"

                "fmla    v28.4s, b0a.4s, a3.s[3]\n"
                "str    q24, [%[c_ptr2]]\n"
                "fmla    v29.4s, b1a.4s, a3.s[3]\n"
                "str    q25, [%[c_ptr2], #16]\n"
                "fmla    v30.4s, b2a.4s, a3.s[3]\n"
                "str    q26, [%[c_ptr2], #32]\n"
                "fmla    v31.4s, b3a.4s, a3.s[3]\n"
                "str    q27, [%[c_ptr2], #48]\n"
                "b    3f\n"

                // "Odd ones" - lead in from even
                "6:\n"
                "fmla    v20.4s, b0a.4s, a1a.s[3]\n"
                "fmla    v21.4s, b1a.4s, a1a.s[3]\n"
                "ldr    b0q, [%[b_ptr]]\n"
                "fmla    v22.4s, b2a.4s, a1a.s[3]\n"
                "subs    %w[odds], %w[odds], #1\n"
                "fmla    v23.4s, b3a.4s, a1a.s[3]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v24.4s, b0a.4s, a2a.s[3]\n"
                "fmla    v25.4s, b1a.4s, a2a.s[3]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"
                "fmla    v26.4s, b2a.4s, a2a.s[3]\n"
                "fmla    v27.4s, b3a.4s, a2a.s[3]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                "fmla    v28.4s, b0a.4s, a3a.s[3]\n"
                "ld1r    {a0.4s}, [%[a_ptr0]], #4\n"
                "fmla    v29.4s, b1a.4s, a3a.s[3]\n"
                "fmla    v30.4s, b2a.4s, a3a.s[3]\n"
                "ld1r    {a1.4s}, [%[a_ptr1]], #4\n"
                "fmla    v31.4s, b3a.4s, a3a.s[3]\n"

                "fmla    v16.4s, bb0.4s, a0.4s\n"
                "beq    9f\n"
                "b    8f\n"

                // "Odd ones" - lead in from odd
                "7:\n"
                "fmla    v20.4s, b0a.4s, a1.s[3]\n"
                "subs    %w[odds], %w[odds], #1\n"
                "fmla    v21.4s, b1a.4s, a1.s[3]\n"
                "ldr    b0q, [%[b_ptr]]\n"
                "fmla    v22.4s, b2a.4s, a1.s[3]\n"
                "fmla    v23.4s, b3a.4s, a1.s[3]\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v24.4s, b0a.4s, a2.s[3]\n"
                "fmla    v25.4s, b1a.4s, a2.s[3]\n"
                "ldr    b2q, [%[b_ptr], #32]\n"
                "fmla    v26.4s, b2a.4s, a2.s[3]\n"
                "fmla    v27.4s, b3a.4s, a2.s[3]\n"
                "ldr    b3q, [%[b_ptr], #48]\n"

                "fmla    v28.4s, b0a.4s, a3.s[3]\n"
                "ld1r    {a0.4s}, [%[a_ptr0]], #4\n"
                "fmla    v29.4s, b1a.4s, a3.s[3]\n"
                "fmla    v30.4s, b2a.4s, a3.s[3]\n"
                "ld1r    {a1.4s}, [%[a_ptr1]], #4\n"
                "fmla    v31.4s, b3a.4s, a3.s[3]\n"

                "fmla    v16.4s, bb0.4s, a0.4s\n"
                "beq    9f\n"

                // "Odd ones" - loop
                "8:\n"
                "fmla    v17.4s, bb1.4s, a0.4s\n"
                "ld1r    {a2.4s}, [%[a_ptr2]], #4\n"
                "fmla    v18.4s, bb2.4s, a0.4s\n"
                "add    %[b_ptr], %[b_ptr], %[ldb]\n"
                "fmla    v19.4s, bb3.4s, a0.4s\n"
                "ld1r    {a3.4s}, [%[a_ptr3]], #4\n"

                "fmla    v20.4s, bb0.4s, a1.4s\n"
                "subs    %w[odds], %w[odds], #1\n"
                "fmla    v21.4s, bb1.4s, a1.4s\n"
                "ld1r    {a0.4s}, [%[a_ptr0]], #4\n"
                "fmla    v22.4s, bb2.4s, a1.4s\n"
                "fmla    v23.4s, bb3.4s, a1.4s\n"
                "ld1r    {a1.4s}, [%[a_ptr1]], #4\n"

                "fmla    v24.4s, bb0.4s, a2.4s\n"
                "fmla    v28.4s, bb0.4s, a3.4s\n"
                "ldr    b0q, [%[b_ptr]]\n"
                "fmla    v25.4s, bb1.4s, a2.4s\n"
                "fmla    v29.4s, bb1.4s, a3.4s\n"
                "ldr    b1q, [%[b_ptr], #16]\n"

                "fmla    v26.4s, bb2.4s, a2.4s\n"
                "fmla    v30.4s, bb2.4s, a3.4s\n"
                "ldr    b2q, [%[b_ptr], #32]\n"
                "fmla    v27.4s, bb3.4s, a2.4s\n"
                "fmla    v31.4s, bb3.4s, a3.4s\n"
                "ldr    b3q, [%[b_ptr], #48]\n"
                "fmla    v16.4s, bb0.4s, a0.4s\n"
                "bne    8b\n"

                // "Odd ones" - detached final iteration
                "9:\n"
                "fmla    v17.4s, bb1.4s, a0.4s\n"
                "ld1r    {a2.4s}, [%[a_ptr2]], #4\n"
                "fmla    v18.4s, bb2.4s, a0.4s\n"
                "fmla    v19.4s, bb3.4s, a0.4s\n"
                "ld1r    {a3.4s}, [%[a_ptr3]], #4\n"

                "fmla    v20.4s, bb0.4s, a1.4s\n"
                "str    q16, [%[c_ptr0]]\n"
                "fmla    v21.4s, bb1.4s, a1.4s\n"
                "str    q17, [%[c_ptr0], #16]\n"
                "fmla    v22.4s, bb2.4s, a1.4s\n"
                "str    q18, [%[c_ptr0], #32]\n"
                "fmla    v23.4s, bb3.4s, a1.4s\n"
                "str    q19, [%[c_ptr0], #48]\n"

                "fmla    v24.4s, bb0.4s, a2.4s\n"
                "str    q20, [%[c_ptr1]]\n"
                "fmla    v25.4s, bb1.4s, a2.4s\n"
                "str    q21, [%[c_ptr1], #16]\n"
                "fmla    v26.4s, bb2.4s, a2.4s\n"
                "str    q22, [%[c_ptr1], #32]\n"
                "fmla    v27.4s, bb3.4s, a2.4s\n"
                "str    q23, [%[c_ptr1], #48]\n"

                "fmla    v28.4s, bb0.4s, a3.4s\n"
                "str    q24, [%[c_ptr2]]\n"
                "fmla    v29.4s, bb1.4s, a3.4s\n"
                "str    q25, [%[c_ptr2], #16]\n"
                "fmla    v30.4s, bb2.4s, a3.4s\n"
                "str    q26, [%[c_ptr2], #32]\n"
                "fmla    v31.4s, bb3.4s, a3.4s\n"
                "str    q27, [%[c_ptr2], #48]\n"

                "3:\n"
                "str	q28, [%[c_ptr3]]\n"
                "add	%[c_ptr0], %[c_ptr0], #64\n"
                "str	q29, [%[c_ptr3], #16]\n"
                "add	%[c_ptr1], %[c_ptr1], %[a_incr1], LSL #1\n"
                "str	q30, [%[c_ptr3], #32]\n"
                "add	%[c_ptr2], %[c_ptr2], %[a_incr2], LSL #1\n"
                "str	q31, [%[c_ptr3], #48]\n"
                "add	%[c_ptr3], %[c_ptr3], %[a_incr3], LSL #1\n"

            : [a_ptr0] "+r" (a_ptr0), [a_ptr1] "+r" (a_ptr1), [a_ptr2] "+r" (a_ptr2), [a_ptr3] "+r" (a_ptr3),
              [b_ptr] "+r" (b_ptr), [loops] "+r" (loops), [odds] "+r" (odds),
              [c_ptr0] "+r" (c_ptr0), [c_ptr1] "+r" (c_ptr1), [c_ptr2] "+r" (c_ptr2), [c_ptr3] "+r" (c_ptr3)
            : [ldb] "r" (ldbb), [oddk] "r" (oddk), [beta0] "r" (beta0), [betaptr] "r" (&beta),
              [a_incr1] "r" (a_incr1), [a_incr2] "r" (a_incr2), [a_incr3] "r" (a_incr3)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
              "cc", "memory"
            );
        }
    }
}

} // namespace arm_gemm

#endif // __aarch64__
