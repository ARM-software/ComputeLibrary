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

#ifdef __aarch64__

#include <arm_neon.h>
#include "dot_toolchain_support.h"
#include <cassert>


inline void a64_gemm_s8_12x8(const int8_t *Apanel, const int8_t *Bpanel, int32_t *Cpanel, int ablocks, int bblocks, int K) {
    assert(Apanel);
    assert(Bpanel);
    assert(Cpanel);
    K/=4;
    const long int row_jump=0;
    const long int block_jump=0;
    const int32_t *a_ptr = reinterpret_cast<const int32_t*>(Apanel);
    int32_t *c_ptr = reinterpret_cast<int32_t*>(Cpanel);
    for (int yb=0; yb<ablocks; yb++) {
        const int32_t *a_ptr0 = a_ptr;
        const int32_t *b_ptr = reinterpret_cast<const int32_t*>(Bpanel);
        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;
            register int32x4_t a0  asm("v0");
            register int32x4_t a1  asm("v1");
            register int32x4_t b0  asm("v2");
            register int32x4_t b1  asm("v3");
            register int32x4_t b2  asm("v4");
            register int32x4_t a0a asm("v5");
            register int32x4_t a1a asm("v6");
            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi	v8.4s, #0x0\n"
                "ldr	%q[a0], [%[a_ptr]]\n"
                "movi	v9.4s, #0x0\n"
                "ldr	%q[b0], [%[b_ptr]]\n"
                "movi	v10.4s, #0x0\n"
                "ldr	%q[a1], [%[a_ptr], #16]\n"
                "movi	v11.4s, #0x0\n"
                "ldr	%q[b1], [%[b_ptr], #16]\n"
                "movi	v12.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi	v13.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi	v14.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi	v15.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi	v16.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi	v17.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi	v18.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi	v19.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi	v20.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "movi	v21.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #384]")
                "movi	v22.4s, #0x0\n"
                "movi	v23.4s, #0x0\n"
                "movi	v24.4s, #0x0\n"
                "movi	v25.4s, #0x0\n"
                "movi	v26.4s, #0x0\n"
                "movi	v27.4s, #0x0\n"
                "movi	v28.4s, #0x0\n"
                "movi	v29.4s, #0x0\n"
                "movi	v30.4s, #0x0\n"
                "movi	v31.4s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz	%w[k], 4f\n"

                _DECLARE_SDOT

                // Loop proper
                "1:\n"
                "sdot	v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "sdot  	v9.4s , %[b0].16b, %[a0].4b[1]\n"

                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "sdot	v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "sdot	v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "ldr	%q[a0a], [%[a_ptr], #32]\n"
                "sdot 	v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "sdot	v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "ldr	%q[a1a], [%[a_ptr], #48]\n"
                "sdot	v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "sdot	v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                "sdot	v16.4s, %[b1].16b, %[a0].4b[0]\n"
                "sdot	v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[a_ptr], #320]")
                "sdot	v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "sdot	v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "sdot	v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "sdot	v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "sdot	v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "sdot	v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                "sdot	v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "sdot	v25.4s, %[b2].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #448]")
                "sdot	v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "sdot	v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "sdot	v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "sdot	v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "sdot	v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "sdot	v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"

                "sdot	v8.4s , %[b0].16b, %[a0a].4b[0]\n"
                "sdot	v9.4s , %[b0].16b, %[a0a].4b[1]\n"
                "ldr	%q[a0], [%[a_ptr], #64]\n"
                "sdot	v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "sdot	v11.4s, %[b0].16b, %[a0a].4b[3]\n"
                "sdot 	v12.4s, %[b0].16b, %[a1a].4b[0]\n"
                "ldr	%q[a1], [%[a_ptr], #80]\n"
                "sdot   v13.4s, %[b0].16b, %[a1a].4b[1]\n"
                "sdot	v14.4s, %[b0].16b, %[a1a].4b[2]\n"
                "sdot	v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #96]\n"

                "sdot	v16.4s, %[b1].16b, %[a0a].4b[0]\n"
                "sdot	v17.4s, %[b1].16b, %[a0a].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #512]")
                "sdot	v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                "sdot	v19.4s, %[b1].16b, %[a0a].4b[3]\n"
                "sdot	v20.4s, %[b1].16b, %[a1a].4b[0]\n"
                "sdot	v21.4s, %[b1].16b, %[a1a].4b[1]\n"
                "sdot	v22.4s, %[b1].16b, %[a1a].4b[2]\n"
                "sdot	v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #112]\n"

                "sdot	v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                "sdot	v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "sdot	v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                "sdot	v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "sdot	v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                "sdot	v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                "subs	%w[k], %w[k], #1\n"
                "sdot	v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                "sdot	v31.4s, %[b2].16b, %[a1a].4b[3]\n"
                "bne	1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz	%w[oddk], 2f\n"

                // Detached final iteration (even K)
                "sdot	v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "sdot   v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "sdot	v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "sdot	v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "ldr	%q[a0a], [%[a_ptr], #32]\n"
                "sdot 	v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "sdot   v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "ldr	%q[a1a], [%[a_ptr], #48]\n"
                "sdot	v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "sdot	v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                "sdot	v16.4s, %[b1].16b, %[a0].4b[0]\n"
                "sdot	v17.4s, %[b1].16b, %[a0].4b[1]\n"
                "sdot	v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "sdot	v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "sdot	v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "sdot	v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "sdot	v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "sdot	v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                "sdot	v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "sdot	v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "sdot	v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "sdot	v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "sdot	v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "sdot	v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "sdot	v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "sdot	v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"

                "sdot	v8.4s , %[b0].16b, %[a0a].4b[0]\n"

                "add	%[b_ptr], %[b_ptr], %[block_jump]\n"
                "sdot	v16.4s, %[b1].16b, %[a0a].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                "sdot   v9.4s , %[b0].16b, %[a0a].4b[1]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "str	q8, [%[c_ptr], #0]\n"
                "sdot	v17.4s, %[b1].16b, %[a0a].4b[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                "sdot	v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                "str	q24, [%[c_ptr], #32]\n"

                "sdot	v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                "str	q9, [%[c_ptr], #48]\n"
                "sdot	v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                "sdot	v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                "sdot	v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                "sdot	v11.4s, %[b0].16b, %[a0a].4b[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                "sdot	v19.4s, %[b1].16b, %[a0a].4b[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                "sdot	v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                "sdot 	v12.4s, %[b0].16b, %[a1a].4b[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                "sdot	v20.4s, %[b1].16b, %[a1a].4b[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                "sdot	v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                "sdot   v13.4s, %[b0].16b, %[a1a].4b[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                "sdot	v21.4s, %[b1].16b, %[a1a].4b[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                "sdot	v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                "sdot	v14.4s, %[b0].16b, %[a1a].4b[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                "sdot	v22.4s, %[b1].16b, %[a1a].4b[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                "sdot	v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                "sdot	v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                "sdot	v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                "sdot	v31.4s, %[b2].16b, %[a1a].4b[3]\n"
                "str	q15, [%[c_ptr], #336]\n"

                "b	3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "sdot	v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                "sdot	v16.4s, %[b1].16b, %[a0].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], %[row_jump]\n"
                "sdot   v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "str	q8, [%[c_ptr], #0]\n"
                "sdot	v17.4s, %[b1].16b, %[a0].4b[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                "sdot	v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], #48\n"
                "add	%[a_ptr], %[a_ptr], #32\n"
                "str	q24, [%[c_ptr], #32]\n"
                "sdot	v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "str	q9, [%[c_ptr], #48]\n"

                "sdot	v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                "sdot	v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                "sdot	v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                "sdot	v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                "sdot	v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                "sdot	v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                "sdot 	v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                "sdot	v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                "sdot	v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                "sdot   v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                "sdot	v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                "sdot	v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                "sdot	v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                "sdot	v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                "sdot	v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                "sdot	v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                "sdot	v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                "sdot	v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "str	q15, [%[c_ptr], #336]\n"


                // Common tail
                "3:\n"
                "str	q23, [%[c_ptr], #352]\n"
                "str	q31, [%[c_ptr], #368]\n"
                "add	%[c_ptr], %[c_ptr], #384\n"

                ".purgem sdot\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk), [row_jump] "r" (row_jump), [block_jump] "r" (block_jump)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
            );
        }
    }


}


#endif 
