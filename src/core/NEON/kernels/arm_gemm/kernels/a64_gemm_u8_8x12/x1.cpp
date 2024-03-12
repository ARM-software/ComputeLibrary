/*
 * Copyright (c) 2017-2020 Arm Limited.
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

void a64_gemm_u8_8x12_x1(const uint8_t *Apanel, const uint8_t *Bpanel, uint32_t *Cpanel, int ablocks, int bblocks, int K) {
    const uint8_t *a_ptr = Apanel;
    uint32_t *c_ptr = Cpanel;
    // We divide K by 4 because the udot instruction processes 4 elements at a time.
    const int W = K/4;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    const int oddk = (W & 1);
    const int init_value_k = ((W+1)/2) - 1;
    for (int yb=0; yb<ablocks; yb++) {
        const uint8_t *a_ptr0 = a_ptr;
        const uint8_t *b_ptr = Bpanel;
        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            int k = init_value_k;
            register uint8x16_t a0  asm("v0");
            register uint8x16_t a1  asm("v1");
            register uint8x16_t b0  asm("v2");
            register uint8x16_t b1  asm("v3");
            register uint8x16_t b2  asm("v4");

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

                // Loop proper
                "1:\n"
                ".word 0x6f80e048 // udot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e049 // udot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                ".word 0x6f80e84a // udot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e84b // udot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                ".word 0x6f81e04c // udot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e04d // udot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                ".word 0x6f81e84e // udot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e84f // udot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                ".word 0x6f80e070 // udot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e071 // udot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[a_ptr], #320]")
                ".word 0x6f80e872 // udot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e873 // udot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                ".word 0x6f81e074 // udot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e075 // udot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                ".word 0x6f81e876 // udot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e877 // udot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                ".word 0x6f80e098 // udot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e099 // udot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #448]")
                ".word 0x6f80e89a // udot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e89b // udot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "ldr	%q[a0], [%[a_ptr], #32]\n"
                ".word 0x6f81e09c // udot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e09d // udot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                ".word 0x6f81e89e // udot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e89f // udot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr	%q[a1], [%[a_ptr], #48]\n"

                ".word 0x6f80e048 // udot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e049 // udot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"
                ".word 0x6f80e84a // udot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e84b // udot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                ".word 0x6f81e04c // udot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e04d // udot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                ".word 0x6f81e84e // udot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e84f // udot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #96]\n"

                ".word 0x6f80e070 // udot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e071 // udot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #512]")
                ".word 0x6f80e872 // udot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e873 // udot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "subs	%w[k], %w[k], #1\n"
                ".word 0x6f81e074 // udot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e075 // udot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                ".word 0x6f81e876 // udot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e877 // udot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #112]\n"

                ".word 0x6f80e098 // udot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e099 // udot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                ".word 0x6f80e89a // udot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e89b // udot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "ldr	%q[a0], [%[a_ptr]]\n"
                ".word 0x6f81e09c // udot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                ".word 0x6fa1e09d // udot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                ".word 0x6f81e89e // udot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e89f // udot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr	%q[a1], [%[a_ptr], #16]\n"
                "bne	1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz	%w[oddk], 2f\n"

                // Detached final iteration (even K)
                ".word 0x6f80e048 // udot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e049 // udot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                ".word 0x6f80e84a // udot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e84b // udot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                ".word 0x6f81e04c // udot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e04d // udot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                ".word 0x6f81e84e // udot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e84f // udot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr	%q[b0], [%[b_ptr], #48]\n"

                ".word 0x6f80e070 // udot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e071 // udot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ".word 0x6f80e872 // udot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e873 // udot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                ".word 0x6f81e074 // udot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e075 // udot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                ".word 0x6f81e876 // udot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e877 // udot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr	%q[b1], [%[b_ptr], #64]\n"

                ".word 0x6f80e098 // udot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e099 // udot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                ".word 0x6f80e89a // udot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                ".word 0x6fa0e89b // udot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "ldr	%q[a0], [%[a_ptr], #-32]\n"
                ".word 0x6f81e09c // udot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                ".word 0x6fa1e09d // udot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                ".word 0x6f81e89e // udot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                ".word 0x6fa1e89f // udot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr	%q[a1], [%[a_ptr], #-16]\n"

                ".word 0x6f80e048 // udot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "ldr	%q[b2], [%[b_ptr], #80]\n"

                ".word 0x6f80e070 // udot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], #96\n"
                ".word 0x6fa0e049 // udot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "str	q8, [%[c_ptr], #0]\n"
                ".word 0x6fa0e071 // udot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                ".word 0x6f80e098 // udot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "str	q24, [%[c_ptr], #32]\n"

                ".word 0x6fa0e099 // udot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "str	q9, [%[c_ptr], #48]\n"
                ".word 0x6f80e84a // udot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                ".word 0x6f80e872 // udot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                ".word 0x6f80e89a // udot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                ".word 0x6fa0e84b // udot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                ".word 0x6fa0e873 // udot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                ".word 0x6fa0e89b // udot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                ".word 0x6f81e04c // udot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                ".word 0x6f81e074 // udot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                ".word 0x6f81e09c // udot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                ".word 0x6fa1e04d // udot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                ".word 0x6fa1e075 // udot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                ".word 0x6fa1e09d // udot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                ".word 0x6f81e84e // udot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                ".word 0x6f81e876 // udot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                ".word 0x6f81e89e // udot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                ".word 0x6fa1e84f // udot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                ".word 0x6fa1e877 // udot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                ".word 0x6fa1e89f // udot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "str	q15, [%[c_ptr], #336]\n"

                "b	3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                ".word 0x6f80e048 // udot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "ldr	%q[b2], [%[b_ptr], #32]\n"
                ".word 0x6f80e070 // udot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".word 0x6fa0e049 // udot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "str	q8, [%[c_ptr], #0]\n"
                ".word 0x6fa0e071 // udot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                "str	q16, [%[c_ptr], #16]\n"
                ".word 0x6f80e098 // udot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "add	%[b_ptr], %[b_ptr], #48\n"
                "add	%[a_ptr], %[a_ptr], #32\n"
                "str	q24, [%[c_ptr], #32]\n"
                ".word 0x6fa0e099 // udot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "str	q9, [%[c_ptr], #48]\n"

                ".word 0x6f80e84a // udot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "str	q17, [%[c_ptr], #64]\n"
                ".word 0x6f80e872 // udot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "str	q25, [%[c_ptr], #80]\n"
                ".word 0x6f80e89a // udot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "str	q10, [%[c_ptr], #96]\n"

                ".word 0x6fa0e84b // udot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "str	q18, [%[c_ptr], #112]\n"
                ".word 0x6fa0e873 // udot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "str	q26, [%[c_ptr], #128]\n"
                ".word 0x6fa0e89b // udot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "str	q11, [%[c_ptr], #144]\n"

                ".word 0x6f81e04c // udot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "str	q19, [%[c_ptr], #160]\n"
                ".word 0x6f81e074 // udot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "str	q27, [%[c_ptr], #176]\n"
                ".word 0x6f81e09c // udot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "str	q12, [%[c_ptr], #192]\n"

                ".word 0x6fa1e04d // udot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "str	q20, [%[c_ptr], #208]\n"
                ".word 0x6fa1e075 // udot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "str	q28, [%[c_ptr], #224]\n"
                ".word 0x6fa1e09d // udot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "str	q13, [%[c_ptr], #240]\n"

                ".word 0x6f81e84e // udot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "str	q21, [%[c_ptr], #256]\n"
                ".word 0x6f81e876 // udot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "str	q29, [%[c_ptr], #272]\n"
                ".word 0x6f81e89e // udot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "str	q14, [%[c_ptr], #288]\n"

                ".word 0x6fa1e84f // udot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "str	q22, [%[c_ptr], #304]\n"
                ".word 0x6fa1e877 // udot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "str	q30, [%[c_ptr], #320]\n"
                ".word 0x6fa1e89f // udot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "str	q15, [%[c_ptr], #336]\n"


                // Common tail
                "3:\n"
                "str	q23, [%[c_ptr], #352]\n"
                "str	q31, [%[c_ptr], #368]\n"
                "add	%[c_ptr], %[c_ptr], #384\n"

            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1),
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
