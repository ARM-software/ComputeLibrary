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
#ifdef __aarch64__

#include <cstddef>

#include <arm_neon.h>

#include "../../asmlib.hpp"
#include "../../utils.hpp"

// Kernel implementation - transposed GEMV
//
// The kernel will process "M" rows of A (= steps of dot product) and "N"
// columns (= dot products total)
//
// General plan is to do as many columns simultaneously as possible - a
// reasonable limit is half the NEON regfile = 64 total accumulators.
//
// It's possible that messing around with sub-blocking M and N can yield
// higher performance, but that's left to the outer loop.  In this kernel we
// process all of M at the same time.


// How far ahead to prefetch for the first and subsequent prefetches.
// These values work for A72 on JunoR2...

#define FIRST_PFD 9
#define PFD 6

namespace arm_gemm {

void a64_sgemv_trans(const float *Astart, const float *Xstart, float *Ystart, float beta, int lda, int M, int N) {
    const float *a_ptr_base = Astart;
    float *y_ptr = Ystart;
    const bool beta0 = (beta == 0.0f);

    register const float32x4_t vb asm("v1") = vdupq_n_f32(beta);

    int firstpfd=FIRST_PFD;
    if (firstpfd > M) {
        firstpfd = (M-1);
    }

    int pfd = PFD;
    if (pfd > M) {
        pfd = (M-1);
    }

    ptrdiff_t jump = lda * sizeof(int);

    for (;N>=96;N-=96) {
        int k = M-1;

        const float *a_ptr = a_ptr_base;
        const float *x_ptr = Xstart;
        const float *pf_ptr = a_ptr;
        const float *firstpf_ptr = a_ptr;
        const float *pf_limit = a_ptr + (M * lda);

        for (int i=0; i<firstpfd; i++) {
            prefetch_1x(firstpf_ptr);
            firstpf_ptr += lda;
        }

        for (int i=0; i<pfd; i++) {
            prefetch_5x(pf_ptr + 16);
            pf_ptr += lda;
        }

        a_ptr_base += 96;

        __asm __volatile (
            "movi	v8.4s,#0x0\n"
            "ldr	w0, [%[x_ptr]]\n"
            "movi	v9.4s,#0x0\n"
            "ldr	q2,  [%[a_ptr], #0]\n"
            "movi	v10.4s,#0x0\n"
            "ldr	q3,  [%[a_ptr], #0x10]\n"
            "movi	v11.4s,#0x0\n"
            "ldr	q4, [%[a_ptr], #0x20]\n"
            "movi	v12.4s,#0x0\n"
            "ldr	q5, [%[a_ptr], #0x30]\n"
            "movi	v13.4s,#0x0\n"
            "ldr	q6, [%[a_ptr], #0x40]\n"
            "movi	v14.4s,#0x0\n"
            "ldr	q7, [%[a_ptr], #0x50]\n"
            "movi	v15.4s,#0x0\n"
            ASM_PREFETCH("[%[firstpf_ptr]]")
            "movi	v16.4s, #0x0\n"
            "movi	v17.4s, #0x0\n"
            ASM_PREFETCH("[%[pf_ptr], #64]")
            "movi	v18.4s, #0x0\n"
            "movi	v19.4s, #0x0\n"
            ASM_PREFETCH("[%[pf_ptr], #128]")
            "movi	v20.4s, #0x0\n"
            "movi	v21.4s, #0x0\n"
            ASM_PREFETCH("[%[pf_ptr], #192]")
            "movi	v22.4s, #0x0\n"
            "movi	v23.4s, #0x0\n"
            ASM_PREFETCH("[%[pf_ptr], #256]")
            "movi	v24.4s, #0x0\n"
            "movi	v25.4s, #0x0\n"
            ASM_PREFETCH("[%[pf_ptr], #320]")
            "movi	v26.4s, #0x0\n"
            "movi	v27.4s, #0x0\n"
            "add	%[pf_ptr], %[pf_ptr], %[jump]\n"
            "movi	v28.4s, #0x0\n"
            "add	%[firstpf_ptr], %[firstpf_ptr], %[jump]\n"
            "movi	v29.4s, #0x0\n"
            "movi	v30.4s, #0x0\n"
            "movi	v31.4s, #0x0\n"

            // Skip everything if there are no iterations of the main loop to do.
            "cbz	%w[k], 10f\n"

            // Loop with all prefetches.  Exit this loop when firstpf_ptr
            // hits pf_limit.
            "1:\n"
            "dup	v0.4s, w0\n"
            "ldr	w0, [%[x_ptr], #4]\n"
            "add	%[x_ptr], %[x_ptr], #0x4\n"
            "fmla	v8.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x60]\n"
            "fmla	v9.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x70]\n"
            ASM_PREFETCH("[%[firstpf_ptr]]")
            "fmla	v10.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x80]\n"
            "add	%[firstpf_ptr], %[firstpf_ptr], %[jump]\n"
            "fmla	v11.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x90]\n"
            "sub	%w[k], %w[k], #1\n"
            ASM_PREFETCH("[%[x_ptr], #128]")
            "fmla	v12.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0xa0]\n"
            "fmla	v13.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0xb0]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x40]")
            "fmla	v14.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0xc0]\n"
            "fmla	v15.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0xd0]\n"
            "fmla	v16.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0xe0]\n"
            "fmla	v17.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0xf0]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x80]")
            "fmla	v18.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x100]\n"
            "fmla	v19.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x110]\n"
            "fmla	v20.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x120]\n"
            "fmla	v21.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x130]\n"
            ASM_PREFETCH("[%[pf_ptr], #0xc0]")
            "fmla	v22.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x140]\n"
            "fmla	v23.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x150]\n"
            "fmla	v24.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x160]\n"
            "fmla	v25.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x170]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x100]")
            "add	%[a_ptr], %[a_ptr], %[jump]\n"
            "fmla	v26.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x00]\n"
            "fmla	v27.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x10]\n"
            "fmla	v28.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x20]\n"
            "fmla	v29.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x30]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x140]")
            "fmla	v30.4s, v6.4s, v0.4s\n"
            "add	%[pf_ptr], %[pf_ptr], %[jump]\n"
            "ldr	q6, [%[a_ptr], #0x40]\n"
            "fmla	v31.4s, v7.4s, v0.4s\n"
            "cmp	%[firstpf_ptr], %[pf_limit]\n"
            "ldr	q7, [%[a_ptr], #0x50]\n"
            "blt	1b\n"

            // Check that there are still "main" prefetches to do.
            "cmp	%[pf_ptr], %[pf_limit]\n"
            "bge	9f\n"

            // Just the main prefetches, exit this loop when pf_ptr hits pf_limit.
            "8:\n"
            "dup	v0.4s, w0\n"
            "ldr	w0, [%[x_ptr], #4]\n"
            "add	%[x_ptr], %[x_ptr], #0x4\n"
            "fmla	v8.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x60]\n"
            "fmla	v9.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x70]\n"
            "fmla	v10.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x80]\n"
            "fmla	v11.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x90]\n"
            "sub	%w[k], %w[k], #1\n"
            ASM_PREFETCH("[%[x_ptr], #128]")
            "fmla	v12.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0xa0]\n"
            "fmla	v13.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0xb0]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x40]")
            "fmla	v14.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0xc0]\n"
            "fmla	v15.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0xd0]\n"
            "fmla	v16.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0xe0]\n"
            "fmla	v17.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0xf0]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x80]")
            "fmla	v18.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x100]\n"
            "fmla	v19.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x110]\n"
            "fmla	v20.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x120]\n"
            "fmla	v21.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x130]\n"
            ASM_PREFETCH("[%[pf_ptr], #0xc0]")
            "fmla	v22.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x140]\n"
            "fmla	v23.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x150]\n"
            "fmla	v24.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x160]\n"
            "fmla	v25.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x170]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x100]")
            "add	%[a_ptr], %[a_ptr], %[jump]\n"
            "fmla	v26.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x00]\n"
            "fmla	v27.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x10]\n"
            "fmla	v28.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x20]\n"
            "fmla	v29.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x30]\n"
            ASM_PREFETCH("[%[pf_ptr], #0x140]")
            "fmla	v30.4s, v6.4s, v0.4s\n"
            "add	%[pf_ptr], %[pf_ptr], %[jump]\n"
            "ldr	q6, [%[a_ptr], #0x40]\n"
            "fmla	v31.4s, v7.4s, v0.4s\n"
            "cmp	%[pf_ptr], %[pf_limit]\n"
            "ldr	q7, [%[a_ptr], #0x50]\n"
            "blt	8b\n"

            // Check that there is still work to do.
            "9:\n"
            "cmp	%w[k], #0\n"
            "beq	10f\n"

            // Loop without prefetches, exit when k hits 0.
            "2:\n"
            "dup	v0.4s, w0\n"
            "ldr	w0, [%[x_ptr], #4]\n"
            "add	%[x_ptr], %[x_ptr], #0x4\n"
            "fmla	v8.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x60]\n"
            "fmla	v9.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x70]\n"
            "fmla	v10.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x80]\n"
            "fmla	v11.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x90]\n"
            "subs	%w[k], %w[k], #1\n"
            "fmla	v12.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0xa0]\n"
            "fmla	v13.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0xb0]\n"
            "fmla	v14.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0xc0]\n"
            "fmla	v15.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0xd0]\n"
            "fmla	v16.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0xe0]\n"
            "fmla	v17.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0xf0]\n"
            "fmla	v18.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x100]\n"
            "fmla	v19.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x110]\n"
            "fmla	v20.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x120]\n"
            "fmla	v21.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x130]\n"
            "fmla	v22.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x140]\n"
            "fmla	v23.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x150]\n"
            "fmla	v24.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x160]\n"
            "fmla	v25.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x170]\n"
            "add	%[a_ptr], %[a_ptr], %[jump]\n"
            "fmla	v26.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x00]\n"
            "fmla	v27.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x10]\n"
            "fmla	v28.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x20]\n"
            "fmla	v29.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x30]\n"
            "fmla	v30.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x40]\n"
            "fmla	v31.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x50]\n"
            "bne	2b\n"

            "10:\n"

            // Final iteration
            "dup	v0.4s, w0\n"
            "fmla	v8.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x60]\n"
            "fmla	v9.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x70]\n"
            "fmla	v10.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x80]\n"
            "fmla	v11.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x90]\n"
            "fmla	v12.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0xa0]\n"
            "fmla	v13.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0xb0]\n"
            "fmla	v14.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0xc0]\n"
            "fmla	v15.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0xd0]\n"
            "fmla	v16.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0xe0]\n"
            "fmla	v17.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0xf0]\n"
            "fmla	v18.4s, v6.4s, v0.4s\n"

            "ldr	q6, [%[a_ptr], #0x100]\n"
            "fmla	v19.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x110]\n"
            "fmla	v20.4s, v2.4s, v0.4s\n"
            "ldr	q2, [%[a_ptr], #0x120]\n"
            "fmla	v21.4s, v3.4s, v0.4s\n"
            "ldr	q3, [%[a_ptr], #0x130]\n"
            "fmla	v22.4s, v4.4s, v0.4s\n"
            "ldr	q4, [%[a_ptr], #0x140]\n"
            "fmla	v23.4s, v5.4s, v0.4s\n"
            "ldr	q5, [%[a_ptr], #0x150]\n"
            "fmla	v24.4s, v6.4s, v0.4s\n"
            "ldr	q6, [%[a_ptr], #0x160]\n"
            "fmla	v25.4s, v7.4s, v0.4s\n"
            "ldr	q7, [%[a_ptr], #0x170]\n"
            "fmla	v26.4s, v2.4s, v0.4s\n"
            "cbnz	%w[beta0], 11f\n"
            "ldr	q2,  [%[y_ptr]]\n"
            "fmla	v27.4s, v3.4s, v0.4s\n"
            "ldr	q3,  [%[y_ptr], #0x10]\n"
            "fmla	v28.4s, v4.4s, v0.4s\n"
            "ldr	q4,  [%[y_ptr], #0x20]\n"
            "fmla	v29.4s, v5.4s, v0.4s\n"
            "ldr	q5,  [%[y_ptr], #0x30]\n"
            "fmla	v30.4s, v6.4s, v0.4s\n"
            "ldr	q6,  [%[y_ptr], #0x40]\n"
            "fmla	v31.4s, v7.4s, v0.4s\n"
            "ldr	q7,  [%[y_ptr], #0x50]\n"

            "fmla	v8.4s, v2.4s, %[vb].4s\n"
            "ldr	q2, [%[y_ptr], #0x60]\n"
            "fmla	v9.4s, v3.4s, %[vb].4s\n"
            "ldr	q3, [%[y_ptr], #0x70]\n"
            "fmla	v10.4s, v4.4s, %[vb].4s\n"
            "ldr	q4, [%[y_ptr], #0x80]\n"
            "fmla	v11.4s, v5.4s, %[vb].4s\n"
            "ldr	q5, [%[y_ptr], #0x90]\n"
            "fmla	v12.4s, v6.4s, %[vb].4s\n"
            "ldr	q6, [%[y_ptr], #0xa0]\n"
            "str	q8, [%[y_ptr], #0x00]\n"
            "fmla	v13.4s, v7.4s, %[vb].4s\n"
            "ldr	q7, [%[y_ptr], #0xb0]\n"
            "str	q9, [%[y_ptr], #0x10]\n"
            "fmla	v14.4s, v2.4s, %[vb].4s\n"
            "ldr	q2, [%[y_ptr], #0xc0]\n"
            "str	q10, [%[y_ptr], #0x20]\n"
            "fmla	v15.4s, v3.4s, %[vb].4s\n"
            "ldr	q3, [%[y_ptr], #0xd0]\n"
            "str	q11, [%[y_ptr], #0x30]\n"
            "fmla	v16.4s, v4.4s, %[vb].4s\n"
            "ldr	q4, [%[y_ptr], #0xe0]\n"
            "str	q12, [%[y_ptr], #0x40]\n"
            "fmla	v17.4s, v5.4s, %[vb].4s\n"
            "ldr	q5, [%[y_ptr], #0xf0]\n"
            "str	q13, [%[y_ptr], #0x50]\n"
            "fmla	v18.4s, v6.4s, %[vb].4s\n"
            "ldr	q6, [%[y_ptr], #0x100]\n"
            "str	q14, [%[y_ptr], #0x60]\n"
            "fmla	v19.4s, v7.4s, %[vb].4s\n"
            "ldr	q7, [%[y_ptr], #0x110]\n"
            "str	q15, [%[y_ptr], #0x70]\n"
            "fmla	v20.4s, v2.4s, %[vb].4s\n"
            "ldr	q2, [%[y_ptr], #0x120]\n"
            "str	q16, [%[y_ptr], #0x80]\n"
            "fmla	v21.4s, v3.4s, %[vb].4s\n"
            "ldr	q3, [%[y_ptr], #0x130]\n"
            "str	q17, [%[y_ptr], #0x90]\n"
            "fmla	v22.4s, v4.4s, %[vb].4s\n"
            "ldr	q4, [%[y_ptr], #0x140]\n"
            "str	q18, [%[y_ptr], #0xa0]\n"
            "fmla	v23.4s, v5.4s, %[vb].4s\n"
            "ldr	q5, [%[y_ptr], #0x150]\n"
            "str	q19, [%[y_ptr], #0xb0]\n"
            "fmla	v24.4s, v6.4s, %[vb].4s\n"
            "ldr	q6, [%[y_ptr], #0x160]\n"
            "str	q20, [%[y_ptr], #0xc0]\n"
            "fmla	v25.4s, v7.4s, %[vb].4s\n"
            "ldr	q7, [%[y_ptr], #0x170]\n"
            "str	q21, [%[y_ptr], #0xd0]\n"
            "fmla	v26.4s, v2.4s, %[vb].4s\n"
            "str	q22, [%[y_ptr], #0xe0]\n"
            "fmla	v27.4s, v3.4s, %[vb].4s\n"
            "str	q23, [%[y_ptr], #0xf0]\n"
            "fmla	v28.4s, v4.4s, %[vb].4s\n"
            "str	q24, [%[y_ptr], #0x100]\n"
            "fmla	v29.4s, v5.4s, %[vb].4s\n"
            "str	q25, [%[y_ptr], #0x110]\n"
            "fmla	v30.4s, v6.4s, %[vb].4s\n"
            "str	q26, [%[y_ptr], #0x120]\n"
            "fmla	v31.4s, v7.4s, %[vb].4s\n"
            "str	q27, [%[y_ptr], #0x130]\n"
            "b		12f\n"

            // beta 0 code - don't read.
            "11:\n"
            "str	q8, [%[y_ptr], #0x00]\n"
            "fmla	v27.4s, v3.4s, v0.4s\n"
            "str	q9, [%[y_ptr], #0x10]\n"
            "fmla	v28.4s, v4.4s, v0.4s\n"
            "str	q10, [%[y_ptr], #0x20]\n"
            "fmla	v29.4s, v5.4s, v0.4s\n"
            "str	q11, [%[y_ptr], #0x30]\n"
            "fmla	v30.4s, v6.4s, v0.4s\n"
            "str	q12, [%[y_ptr], #0x40]\n"
            "fmla	v31.4s, v7.4s, v0.4s\n"

            "str	q13, [%[y_ptr], #0x50]\n"
            "str	q14, [%[y_ptr], #0x60]\n"
            "str	q15, [%[y_ptr], #0x70]\n"
            "str	q16, [%[y_ptr], #0x80]\n"
            "str	q17, [%[y_ptr], #0x90]\n"
            "str	q18, [%[y_ptr], #0xa0]\n"
            "str	q19, [%[y_ptr], #0xb0]\n"
            "str	q20, [%[y_ptr], #0xc0]\n"
            "str	q21, [%[y_ptr], #0xd0]\n"
            "str	q22, [%[y_ptr], #0xe0]\n"
            "str	q23, [%[y_ptr], #0xf0]\n"
            "str	q24, [%[y_ptr], #0x100]\n"
            "str	q25, [%[y_ptr], #0x110]\n"
            "str	q26, [%[y_ptr], #0x120]\n"
            "str	q27, [%[y_ptr], #0x130]\n"

            "12:\n"
            "stp	q28, q29, [%[y_ptr], #0x140]\n"
            "stp	q30, q31, [%[y_ptr], #0x160]\n"
            "add	%[y_ptr], %[y_ptr], #0x180\n"



          : [a_ptr] "+r" (a_ptr), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr), [k] "+r" (k), [pf_ptr] "+r" (pf_ptr), [firstpf_ptr] "+r" (firstpf_ptr)
          : [jump] "r" (jump), [vb] "w" (vb), [pf_limit] "r" (pf_limit), [beta0] "r" (beta0)
          : "w0", "v0", "v2", "v3", "v4", "v5", "v6", "v7", "v8",  "v9", "v10", "v11", "v12", "v13",
            "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
            "v27", "v28", "v29", "v30", "v31", "cc"
        );
    }

    if (N>0) {
        // Handle N tail - up to 95 stragglers.
        // This is 0-23 vectors, plus optionally an 64-bit vector and/or a
        // single value for the remainder.

        // Independent pointers into the matrix for the odd 2 and odd 1.
        // Double up as flag to indicate whether they are needed.
        const float *odd2_aptr=NULL;
        const float *odd1_aptr=NULL;

        // Figure out how much work we need to do.
        int numvecs = N/4;
        int rem = N%4;
        int k=M;

        // Set up pointers for the odd 2/1 if needed.
        if (rem >= 2) {
            odd2_aptr = a_ptr_base + (numvecs * 4);
        }

        if (rem & 1) {
            odd1_aptr = a_ptr_base + (numvecs * 4) + (odd2_aptr==NULL ? 0 : 2);
        }

        const float *a_ptr = a_ptr_base;
        const float *firstpf_ptr = a_ptr_base;
        const float *pf_ptr = a_ptr_base;
        const float *pf_limit = a_ptr + (M * lda);

        const float *x_ptr = Xstart;
        int vecs=0; // Working variable to count how many vectors to work on.
        int dopf=1; // Track whether we are doing prefetches.

        // Figure out how many cache lines we need to prefetch each time.
        int numpfs = (N + 15) / 16;

        // Do initial prefetches
        for (int i=0; i<firstpfd+1; i++) {
            prefetch_1x(firstpf_ptr);
            firstpf_ptr += lda;
        }

        // Do "main" prefetches - adapt number to the number we actually need.
        if (numpfs > 1) {
            for (int i=0; i<pfd+1; i++) {
                switch (numpfs) {
                    case 2:
                        prefetch_1x(pf_ptr + 16);
                        break;

                    case 3:
                        prefetch_2x(pf_ptr + 16);
                        break;

                    case 4:
                        prefetch_3x(pf_ptr + 16);
                        break;

                    case 5:
                        prefetch_4x(pf_ptr + 16);
                        break;

                    case 6:
                        prefetch_5x(pf_ptr + 16);
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
                pf_ptr += lda;
            }
        } else {
            // Just disable additional prefetches
            dopf=0;
        }

        // Do the real work
        __asm __volatile (
            // Initialize all the vectors - not worth skipping this if only
            // some are needed.
            "movi	v8.4s,#0x0\n"
            "ldr	w0, [%[x_ptr]]\n"
            "movi	v9.4s,#0x0\n"
            "movi	v10.4s,#0x0\n"
            "movi	v11.4s,#0x0\n"
            "movi	v12.4s,#0x0\n"
            "movi	v13.4s,#0x0\n"
            "movi	v14.4s,#0x0\n"
            "movi	v15.4s,#0x0\n"
            "movi	v16.4s, #0x0\n"
            "movi	v17.4s, #0x0\n"
            "movi	v18.4s, #0x0\n"
            "movi	v19.4s, #0x0\n"
            "movi	v20.4s, #0x0\n"
            "movi	v21.4s, #0x0\n"
            "movi	v22.4s, #0x0\n"
            "movi	v23.4s, #0x0\n"
            "movi	v24.4s, #0x0\n"
            "movi	v25.4s, #0x0\n"
            "movi	v26.4s, #0x0\n"
            "movi	v27.4s, #0x0\n"
            "movi	v28.4s, #0x0\n"
            "movi	v29.4s, #0x0\n"
            "movi	v30.4s, #0x0\n"
            "movi	v6.2s, #0x0\n"
            "movi	v5.2s, #0x0\n"

            "1:\n"
            ASM_PREFETCH("[%[firstpf_ptr]]\n")
            "11:\n"
            "dup	v0.4s, w0\n"
            "ldr	w0, [%[x_ptr], #4]\n"
            "add	%[x_ptr], %[x_ptr], #4\n"

            "cbz	%w[numvecs], 2f\n"
            "mov	%w[vecs], %w[numvecs]\n"

            // Vector 0
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x00]\n"
            "fmla	v8.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 1
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x10]\n"
            "fmla	v9.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 2
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x20]\n"
            "fmla	v10.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 3
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x30]\n"
            "fmla	v11.4s, v7.4s, v0.4s\n"
            // Prefetch
            "cbz	%w[dopf], 3f\n"
            ASM_PREFETCH("[%[pf_ptr], #0x40]")
            "3:\n"
            "beq	2f\n"

            // Vector 4
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x40]\n"
            "fmla	v12.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 5
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x50]\n"
            "fmla	v13.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 6
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x60]\n"
            "fmla	v14.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 7
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x70]\n"
            "fmla	v15.4s, v7.4s, v0.4s\n"
            // Prefetch
            "cbz	%w[dopf], 4f\n"
            ASM_PREFETCH("[%[pf_ptr], #0x80]")
            "4:\n"
            "beq	2f\n"

            // Vector 8
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x80]\n"
            "fmla	v16.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 9
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x90]\n"
            "fmla	v17.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 10
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xa0]\n"
            "fmla	v18.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 11
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xb0]\n"
            "fmla	v19.4s, v7.4s, v0.4s\n"
            // Prefetch
            "cbz	%w[dopf], 5f\n"
            ASM_PREFETCH("[%[pf_ptr], #0xc0]")
            "5:\n"
            "beq	2f\n"

            // Vector 12
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xc0]\n"
            "fmla	v20.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 13
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xd0]\n"
            "fmla	v21.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 14
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xe0]\n"
            "fmla	v22.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 15
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0xf0]\n"
            "fmla	v23.4s, v7.4s, v0.4s\n"
            // Prefetch
            "cbz	%w[dopf], 6f\n"
            ASM_PREFETCH("[%[pf_ptr], #0x100]")
            "6:\n"
            "beq	2f\n"

            // Vector 16
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x100]\n"
            "fmla	v24.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 17
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x110]\n"
            "fmla	v25.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 18
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x120]\n"
            "fmla	v26.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 19
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x130]\n"
            "fmla	v27.4s, v7.4s, v0.4s\n"
            // Prefetch
            "cbz	%w[dopf], 7f\n"
            ASM_PREFETCH("[%[pf_ptr], #0x140]")
            "7:\n"
            "beq	2f\n"

            // Vector 20
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x140]\n"
            "fmla	v28.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 21
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x150]\n"
            "fmla	v29.4s, v7.4s, v0.4s\n"
            "beq	2f\n"
            // Vector 22
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7,[%[a_ptr], #0x160]\n"
            "fmla	v30.4s, v7.4s, v0.4s\n"

            "2:\n"
            "add	%[a_ptr], %[a_ptr], %[jump]\n"

            // Do the odd 2-vector, if needed
            "cbz	%[odd2_aptr], 8f\n"
            "ldr	d7, [%[odd2_aptr]]\n"
            "fmla	v6.2s, v7.2s, v0.2s\n"
            "add	%[odd2_aptr], %[odd2_aptr], %[jump]\n"

            "8:\n"
            // Do the odd 1-vector, if needed
            "cbz	%[odd1_aptr], 9f\n"
            "ldr	s7, [%[odd1_aptr]]\n"
            "fmla	v5.2s, v7.2s, v0.2s\n"
            "add	%[odd1_aptr], %[odd1_aptr], %[jump]\n"

            // Get out if needed.
            "9:\n"
            "subs	%w[k], %w[k], #1\n"
            "beq	10f\n"

            // Update the "main" prefetch pointer, if it strays beyond the limit turn off "dopf"
            "add	%[pf_ptr], %[pf_ptr], %[jump]\n"
            "cmp	%[pf_ptr], %[pf_limit]\n"
            "csel	%w[dopf], %w[dopf], WZR, LT\n"

            // Update the "leading" prefetch pointer, don't do the first
            // instruction of the loop if it's over the limit.
            "add	%[firstpf_ptr], %[firstpf_ptr], %[jump]\n"
            "cmp	%[firstpf_ptr], %[pf_limit]\n"
            "blt	1b\n"
            "b		11b\n"

            // Now write out the outputs
            "10:\n"
            "cbnz	%w[beta0], 15f\n"

            "cbz	%w[numvecs], 12f\n"
            "mov	%w[vecs], %w[numvecs]\n"

            // Vector 0
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v8.4s, v7.4s, %[vb].4s\n"
            "str	q8, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 1
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v9.4s, v7.4s, %[vb].4s\n"
            "str	q9, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 2
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v10.4s, v7.4s, %[vb].4s\n"
            "str	q10, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 3
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v11.4s, v7.4s, %[vb].4s\n"
            "str	q11, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 4
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v12.4s, v7.4s, %[vb].4s\n"
            "str	q12, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 5
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v13.4s, v7.4s, %[vb].4s\n"
            "str	q13, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 6
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v14.4s, v7.4s, %[vb].4s\n"
            "str	q14, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 7
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v15.4s, v7.4s, %[vb].4s\n"
            "str	q15, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 8
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v16.4s, v7.4s, %[vb].4s\n"
            "str	q16, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 9
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v17.4s, v7.4s, %[vb].4s\n"
            "str	q17, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 10
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v18.4s, v7.4s, %[vb].4s\n"
            "str	q18, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 11
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v19.4s, v7.4s, %[vb].4s\n"
            "str	q19, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 12
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v20.4s, v7.4s, %[vb].4s\n"
            "str	q20, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 13
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v21.4s, v7.4s, %[vb].4s\n"
            "str	q21, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 14
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v22.4s, v7.4s, %[vb].4s\n"
            "str	q22, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 15
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v23.4s, v7.4s, %[vb].4s\n"
            "str	q23, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 16
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v24.4s, v7.4s, %[vb].4s\n"
            "str	q24, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 17
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v25.4s, v7.4s, %[vb].4s\n"
            "str	q25, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 18
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v26.4s, v7.4s, %[vb].4s\n"
            "str	q26, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 19
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v27.4s, v7.4s, %[vb].4s\n"
            "str	q27, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 20
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v28.4s, v7.4s, %[vb].4s\n"
            "str	q28, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 21
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v29.4s, v7.4s, %[vb].4s\n"
            "str	q29, [%[y_ptr]], #0x10\n"
            "beq	12f\n"
            // Vector 22
            "subs	%w[vecs], %w[vecs], #1\n"
            "ldr	q7, [%[y_ptr]]\n"
            "fmla	v30.4s, v7.4s, %[vb].4s\n"
            "str	q30, [%[y_ptr]], #0x10\n"

            // Odd 2
            "12:\n"
            "cbz	%[odd2_aptr], 13f\n"
            "ldr	d7, [%[y_ptr]]\n"
            "fmla	v6.2s, v7.2s, %[vb].2s\n"
            "str	d6, [%[y_ptr]], #0x8\n"

            // Odd 1
            "13:\n"
            "cbz	%[odd1_aptr], 14f\n"
            "ldr	s7, [%[y_ptr]]\n"
            "fmla	v5.2s, v7.2s, %[vb].2s\n"
            "str	s5, [%[y_ptr]]\n"
            "b		14f\n"

            "15:\n"
            // beta0 code
            "cbz	%w[numvecs], 16f\n"
            "mov	%w[vecs], %w[numvecs]\n"

            // Vector 0
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q8, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 1
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q9, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 2
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q10, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 3
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q11, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 4
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q12, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 5
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q13, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 6
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q14, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 7
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q15, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 8
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q16, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 9
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q17, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 10
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q18, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 11
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q19, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 12
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q20, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 13
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q21, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 14
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q22, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 15
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q23, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 16
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q24, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 17
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q25, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 18
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q26, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 19
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q27, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 20
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q28, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 21
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q29, [%[y_ptr]], #0x10\n"
            "beq	16f\n"
            // Vector 22
            "subs	%w[vecs], %w[vecs], #1\n"
            "str	q30, [%[y_ptr]], #0x10\n"

            // Odd 2
            "16:\n"
            "cbz	%[odd2_aptr], 17f\n"
            "str	d6, [%[y_ptr]], #0x8\n"

            // Odd 1
            "17:\n"
            "cbz	%[odd1_aptr], 14f\n"
            "str	s5, [%[y_ptr]]\n"

            "14:\n"
          : [a_ptr] "+r" (a_ptr), [x_ptr] "+r" (x_ptr), [y_ptr] "+r" (y_ptr), [k] "+r" (k),
            [pf_ptr] "+r" (pf_ptr), [firstpf_ptr] "+r" (firstpf_ptr),
            [odd1_aptr] "+r" (odd1_aptr), [odd2_aptr] "+r" (odd2_aptr),
            [dopf] "+r" (dopf), [vecs] "+r" (vecs)
          : [jump] "r" (jump), [vb] "w" (vb), [pf_limit] "r" (pf_limit), [numvecs] "r" (numvecs), [beta0] "r" (beta0)
          : "w0", "v0", "v2", "v3", "v4", "v5", "v6", "v7", "v8",  "v9", "v10", "v11", "v12", "v13",
            "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
            "v27", "v28", "v29", "v30", "v31", "cc"
        );
    }
}

} // namespace arm_gemm

#endif // __aarch64__
