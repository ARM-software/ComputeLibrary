/*
 * Copyright (c) 2017-2018 Arm Limited.
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

#include <arm_neon.h>

#include "../../asmlib.hpp"

namespace arm_gemm {

void a64_sgemm_asimd_8x12_a55r1(const float *Apanel, const float *Bpanel, float *Cpanel, const int ablocks, const int bblocks, const int K) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k_iters = ((K+1)/2) - 1;

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            int k = k_iters;

            register float32x4_t a0  asm("v0");
            register float32x4_t a1  asm("v1");
            register float32x4_t b0  asm("v2");
            register float32x4_t b1  asm("v3");
            register float32x4_t b2  asm("v4");
            register float32x4_t a0a asm("v5");
            register float32x4_t a1a asm("v6");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi   v8.4s, #0x0\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "movi   v9.4s, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi   v10.4s, #0x0\n"
                "ldr    %q[a1], [%[a_ptr], #16]\n"
                "movi   v11.4s, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi   v12.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi   v13.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi   v14.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi   v15.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi   v16.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi   v17.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi   v18.4s, #0x0\n"
                "movi   v19.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi   v20.4s, #0x0\n"
                "movi   v21.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi   v22.4s, #0x0\n"
                "movi   v23.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "movi   v24.4s, #0x0\n"
                "movi   v25.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #384]")
                "movi   v26.4s, #0x0\n"
                "movi   v27.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #448]")
                "movi   v28.4s, #0x0\n"
                "movi   v29.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #384]")
                "movi   v30.4s, #0x0\n"
                "movi   v31.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #512]")

                // The loop is offset by these two instructions which must
                // always be executed.
                "fmla	v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr	%d[b2], [%[b_ptr], #32]\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz	%w[k], 4f\n"

                "1:\n"
                // Unroll 0
                "fmla	v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr	x20, [%[b_ptr], #40]\n"
                "fmla	v10.4s, %[b0].4s, %[a0].s[2]\n"
                "subs	%w[k], %w[k], #1\n"
                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr	%d[a0a], [%[a_ptr], #32]\n"

                "fmla	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "ins	%[b2].d[1], x20\n"
                "fmla	v13.4s, %[b0].4s, %[a1].s[1]\n"
                "ldr	x20, [%[a_ptr], #40]\n"
                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr	%d[a1a], [%[a_ptr], #48]\n"

                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "ins	%[a0a].d[1], x20\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                "ldr	x20, [%[a_ptr], #56]\n"
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                "ldr	%d[b0], [%[b_ptr], #48]\n"

                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "ins	%[a1a].d[1], x20\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                "ldr	x20, [%[b_ptr], #56]\n"
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr	%d[b1], [%[b_ptr], #64]\n"

                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "ins	%[b0].d[1], x20\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                "ldr	x20, [%[b_ptr], #72]\n"
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                ASM_PREFETCH("[%[a_ptr], #448]")

                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                ASM_PREFETCH("[%[b_ptr], #576]")
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"

                // Unroll 1
                "ldr	%d[b2], [%[b_ptr], #80]\n"

                "fmla	v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "ins	%[b1].d[1], x20\n"
                "fmla	v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "ldr	x20, [%[b_ptr], #88]\n"
                "fmla	v10.4s, %[b0].4s, %[a0a].s[2]\n"
                "fmla	v11.4s, %[b0].4s, %[a0a].s[3]\n"
                "ldr	%d[a0], [%[a_ptr], #64]\n"

                "fmla	v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "ins	%[b2].d[1], x20\n"
                "fmla	v13.4s, %[b0].4s, %[a1a].s[1]\n"
                "ldr	x20, [%[a_ptr], #72]\n"
                "fmla	v14.4s, %[b0].4s, %[a1a].s[2]\n"
                "fmla	v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "ldr	%d[a1], [%[a_ptr], #80]\n"

                "fmla	v16.4s, %[b1].4s, %[a0a].s[0]\n"
                "ins	%[a0].d[1], x20\n"
                "fmla	v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "ldr	x20, [%[a_ptr], #88]\n"
                "fmla	v18.4s, %[b1].4s, %[a0a].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0a].s[3]\n"
                "ldr	%d[b0], [%[b_ptr], #96]\n"

                "fmla	v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "ins	%[a1].d[1], x20\n"
                "fmla	v21.4s, %[b1].4s, %[a1a].s[1]\n"
                "ldr	x20, [%[b_ptr], #104]\n"
                "fmla	v22.4s, %[b1].4s, %[a1a].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1a].s[3]\n"
                "ldr	%d[b1], [%[b_ptr], #112]\n"

                "fmla	v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "ins	%[b0].d[1], x20\n"
                "fmla	v25.4s, %[b2].4s, %[a0a].s[1]\n"
                "ldr	x20, [%[b_ptr], #120]\n"
                "fmla	v26.4s, %[b2].4s, %[a0a].s[2]\n"

                "fmla	v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"

                "fmla	v28.4s, %[b2].4s, %[a1a].s[0]\n"
                ASM_PREFETCH("[%[b_ptr], #640]")
                "fmla	v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "fmla	v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "ins	%[b1].d[1], x20\n"
                "fmla	v31.4s, %[b2].4s, %[a1a].s[3]\n"
                "ldr	%d[b2], [%[b_ptr], #32]\n"

                "fmla	v8.4s , %[b0].4s, %[a0].s[0]\n"
                "b.ne	1b\n"

                // Branch here if K=1 or 2.  Do the right thing for odd/even at the end.
                "4:\n"

		// Start final iteration - branch off to "odd" code before we load a0a.
                "fmla	v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr	x20, [%[b_ptr], #40]\n"
                "fmla	v10.4s, %[b0].4s, %[a0].s[2]\n"
                "cbnz	%w[oddk], 2f\n"

                // Even K continuation
                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr	%d[a0a], [%[a_ptr], #32]\n"

                "fmla	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "ins	%[b2].d[1], x20\n"
                "fmla	v13.4s, %[b0].4s, %[a1].s[1]\n"
                "ldr	x20, [%[a_ptr], #40]\n"
                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                ASM_PREFETCHW("[%[c_ptr]]")
                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr	%d[a1a], [%[a_ptr], #48]\n"

                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "ins	%[a0a].d[1], x20\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                "ldr	x20, [%[a_ptr], #56]\n"
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                "ldr	%d[b0], [%[b_ptr], #48]\n"

                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "ins	%[a1a].d[1], x20\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                "ldr	x20, [%[b_ptr], #56]\n"
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                ASM_PREFETCHW("[%[c_ptr], #64]")
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"

                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                ASM_PREFETCHW("[%[c_ptr], #128]")
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                "ldr	%d[b1], [%[b_ptr], #64]\n"

                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                "ins	%[b0].d[1], x20\n"
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                "ldr	x20, [%[b_ptr], #72]\n"
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                ASM_PREFETCHW("[%[c_ptr], #192]")
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr	%d[b2], [%[b_ptr], #80]\n"

                "fmla	v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "ins	%[b1].d[1], x20\n"
                "fmla	v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "ldr	x20, [%[b_ptr], #88]\n"
                "fmla	v10.4s, %[b0].4s, %[a0a].s[2]\n"
                "ins	%[b2].d[1], x20\n"

                "fmla	v11.4s, %[b0].4s, %[a0a].s[3]\n"
                ASM_PREFETCHW("[%[c_ptr], #256]")
                "fmla	v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "fmla	v13.4s, %[b0].4s, %[a1a].s[1]\n"
                "fmla	v14.4s, %[b0].4s, %[a1a].s[2]\n"
                ASM_PREFETCHW("[%[c_ptr], #320]")
                "fmla	v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "fmla	v16.4s, %[b1].4s, %[a0a].s[0]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #384]")
                "fmla	v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "fmla	v18.4s, %[b1].4s, %[a0a].s[2]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #448]")
                "fmla	v19.4s, %[b1].4s, %[a0a].s[3]\n"
                "fmla	v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "fmla	v21.4s, %[b1].4s, %[a1a].s[1]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #512]")
                "fmla	v22.4s, %[b1].4s, %[a1a].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1a].s[3]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #576]")
                "fmla	v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0a].s[1]\n"
                "fmla	v26.4s, %[b2].4s, %[a0a].s[2]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #640]")
                "fmla	v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "fmla	v28.4s, %[b2].4s, %[a1a].s[0]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #704]")
                "fmla	v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "fmla	v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "fmla	v31.4s, %[b2].4s, %[a1a].s[3]\n"
                "b	3f\n"

                // Odd K continuation
                "2:\n"
                "fmla	v11.4s, %[b0].4s, %[a0].s[3]\n"
                ASM_PREFETCHW("[%[c_ptr]]")
                "fmla	v12.4s, %[b0].4s, %[a1].s[0]\n"
                "ins	%[b2].d[1], x20\n"
                "fmla	v13.4s, %[b0].4s, %[a1].s[1]\n"
                ASM_PREFETCHW("[%[c_ptr], #64]")
                "fmla	v14.4s, %[b0].4s, %[a1].s[2]\n"
                "add	%[a_ptr], %[a_ptr], #32\n"
                "fmla	v15.4s, %[b0].4s, %[a1].s[3]\n"
                ASM_PREFETCHW("[%[c_ptr], #128]")
                "fmla	v16.4s, %[b1].4s, %[a0].s[0]\n"
                "add	%[b_ptr], %[b_ptr], #48\n"
                "fmla	v17.4s, %[b1].4s, %[a0].s[1]\n"
                ASM_PREFETCHW("[%[c_ptr], #192]")
                "fmla	v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla	v19.4s, %[b1].4s, %[a0].s[3]\n"
                ASM_PREFETCHW("[%[c_ptr], #256]")
                "fmla	v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla	v21.4s, %[b1].4s, %[a1].s[1]\n"
                ASM_PREFETCHW("[%[c_ptr], #320]")
                "fmla	v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla	v23.4s, %[b1].4s, %[a1].s[3]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #384]")
                "fmla	v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla	v25.4s, %[b2].4s, %[a0].s[1]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #448]")
                "fmla	v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla	v27.4s, %[b2].4s, %[a0].s[3]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #512]")
                "fmla	v28.4s, %[b2].4s, %[a1].s[0]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #576]")
                "fmla	v29.4s, %[b2].4s, %[a1].s[1]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #640]")
                "fmla	v30.4s, %[b2].4s, %[a1].s[2]\n"
                ASM_PREFETCHWL2("[%[c_ptr], #704]")
                "fmla	v31.4s, %[b2].4s, %[a1].s[3]\n"

                // Common tail
                "3:\n"
                "str	q8,   [%[c_ptr]]\n"
                "str	q16,  [%[c_ptr], #16]\n"
                "str	q24,  [%[c_ptr], #32]\n"
                "str	q9,   [%[c_ptr], #48]\n"
                "str	q17,  [%[c_ptr], #64]\n"
                "str	q25,  [%[c_ptr], #80]\n"
                "str	q10,  [%[c_ptr], #96]\n"
                "str	q18,  [%[c_ptr], #112]\n"
                "str	q26,  [%[c_ptr], #128]\n"
                "str	q11,  [%[c_ptr], #144]\n"
                "str	q19,  [%[c_ptr], #160]\n"
                "str	q27,  [%[c_ptr], #176]\n"
                "str	q12,  [%[c_ptr], #192]\n"
                "str	q20,  [%[c_ptr], #208]\n"
                "str	q28,  [%[c_ptr], #224]\n"
                "str	q13,  [%[c_ptr], #240]\n"
                "str	q21,  [%[c_ptr], #256]\n"
                "str	q29,  [%[c_ptr], #272]\n"
                "str	q14,  [%[c_ptr], #288]\n"
                "str	q22,  [%[c_ptr], #304]\n"
                "str	q30,  [%[c_ptr], #320]\n"
                "str	q15,  [%[c_ptr], #336]\n"
                "str	q23,  [%[c_ptr], #352]\n"
                "str	q31,  [%[c_ptr], #368]\n"
                "add	%[c_ptr], %[c_ptr], #384\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
            );
        }
    }
}

} // namespace arm_gemm

#endif
