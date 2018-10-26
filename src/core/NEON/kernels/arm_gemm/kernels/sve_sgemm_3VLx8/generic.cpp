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
#ifdef __ARM_FEATURE_SVE

#include <arm_neon.h>
#include <arm_sve.h>

#include "../../asmlib.hpp"

// Kernel implementation.
//
// Assume that "Apanel" points to a chunk of A blocks (each size 8xK) in read-order.
// Assume that "Bpanel" points to a chunk of B blocks (each size 3VLxK) in read-order.
// Assume that "Cpanel" points to a chunk of C output blocks (each size
// 3VLx8), the chunks being arranged in a row major fashion.
//
// Note that the intent of this is that either ablocks or bblocks will be 1
// - this construction allows the output loop to proceed in either order.

namespace arm_gemm {

void sve_sgemm_3VLx8(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    // There's no predication inside the kernel, so get a true predicate to use everywhere.
    svbool_t ptrue = svptrue_b32();

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register svfloat32_t a0  asm("z0");
            register svfloat32_t a1  asm("z1");
            register svfloat32_t b0  asm("z2");
            register svfloat32_t b1  asm("z3");
            register svfloat32_t b2  asm("z4");
            register svfloat32_t a0a asm("z5");
            register svfloat32_t a1a asm("z6");

            // Note: All prefetches commented out for now, but left in place as documentation for how it was done on NEON.
            // Actual prefetches to be added once test hardware is available.
            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "mov	z8.s, #0\n"
                "ld1rqw	%[a0].S, %[ptrue]/Z, [%[a_ptr]]\n"
                "mov	z9.s, #0\n"
                "ld1w	%[b0].S, %[ptrue]/Z, [%[b_ptr]]\n"
                "mov	z10.s, #0\n"
                "ld1rqw	%[a1].S, %[ptrue]/Z, [%[a_ptr], #0x10]\n"
                "mov	z11.s, #0\n"
                "ld1w	%[b1].S, %[ptrue]/Z, [%[b_ptr], #1, MUL VL]\n"
                "mov	z12.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #64]")
                "mov	z13.s, #0\n"
                //ASM_PREFETCH("[%[a_ptr], #64]")
                "mov	z14.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #128]")
                "mov	z15.s, #0\n"
                //ASM_PREFETCH("[%[a_ptr], #128]")
                "mov	z16.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #192]")
                "mov	z17.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #256]")
                "mov	z18.s, #0\n"
                //ASM_PREFETCH("[%[a_ptr], #192]")
                "mov	z19.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #320]")
                "mov	z20.s, #0\n"
                //ASM_PREFETCH("[%[a_ptr], #256]")
                "mov	z21.s, #0\n"
                //ASM_PREFETCH("[%[b_ptr], #384]")
                "mov	z22.s, #0\n"
                "mov	z23.s, #0\n"
                "mov	z24.s, #0\n"
                "mov	z25.s, #0\n"
                "mov	z26.s, #0\n"
                "mov	z27.s, #0\n"
                "mov	z28.s, #0\n"
                "mov	z29.s, #0\n"
                "mov	z30.s, #0\n"
                "mov	z31.s, #0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz	%w[k], 4f\n"

                // Loop proper
                "1:\n"
                "fmla 	z8.s , %[b0].s, %[a0].s[0]\n"
                "fmla  	z9.s , %[b0].s, %[a0].s[1]\n"
                "ld1w	%[b2].s, %[ptrue]/Z, [%[b_ptr], #2, MUL VL]\n"
                "fmla	z10.s, %[b0].s, %[a0].s[2]\n"
                "fmla	z11.s, %[b0].s, %[a0].s[3]\n"
                "ld1rqw	%[a0a].s, %[ptrue]/Z, [%[a_ptr], #0x20]\n"
                "fmla 	z12.s, %[b0].s, %[a1].s[0]\n"
                "fmla	z13.s, %[b0].s, %[a1].s[1]\n"
                "ld1rqw	%[a1a].s, %[ptrue]/Z, [%[a_ptr], #0x30]\n"
                "fmla	z14.s, %[b0].s, %[a1].s[2]\n"
                "fmla	z15.s, %[b0].s, %[a1].s[3]\n"
                "ld1w	%[b0].s, %[ptrue]/Z, [%[b_ptr], #3, MUL VL]\n"

                "fmla	z16.s, %[b1].s, %[a0].s[0]\n"
                "fmla	z17.s, %[b1].s, %[a0].s[1]\n"
                //ASM_PREFETCH("[%[a_ptr], #320]")
                "fmla	z18.s, %[b1].s, %[a0].s[2]\n"
                "fmla	z19.s, %[b1].s, %[a0].s[3]\n"
                "fmla	z20.s, %[b1].s, %[a1].s[0]\n"
                "fmla	z21.s, %[b1].s, %[a1].s[1]\n"
                "fmla	z22.s, %[b1].s, %[a1].s[2]\n"
                "fmla	z23.s, %[b1].s, %[a1].s[3]\n"
                "ld1w	%[b1].s, %[ptrue]/Z, [%[b_ptr], #4, MUL VL]\n"

                "fmla	z24.s, %[b2].s, %[a0].s[0]\n"
                "fmla	z25.s, %[b2].s, %[a0].s[1]\n"
                //ASM_PREFETCH("[%[b_ptr], #448]")
                "fmla	z26.s, %[b2].s, %[a0].s[2]\n"
                "fmla	z27.s, %[b2].s, %[a0].s[3]\n"
                "fmla	z28.s, %[b2].s, %[a1].s[0]\n"
                "fmla	z29.s, %[b2].s, %[a1].s[1]\n"
                "fmla	z30.s, %[b2].s, %[a1].s[2]\n"
                "fmla	z31.s, %[b2].s, %[a1].s[3]\n"
                "ld1w	%[b2].s, %[ptrue]/Z, [%[b_ptr], #5, MUL VL]\n"

                "fmla 	z8.s , %[b0].s, %[a0a].s[0]\n"
                "fmla	z9.s , %[b0].s, %[a0a].s[1]\n"
                "ld1rqw	%[a0].s, %[ptrue]/Z, [%[a_ptr], #0x40]\n"
                "fmla	z10.s, %[b0].s, %[a0a].s[2]\n"
                "fmla	z11.s, %[b0].s, %[a0a].s[3]\n"
                "fmla 	z12.s, %[b0].s, %[a1a].s[0]\n"
                "ld1rqw	%[a1].s, %[ptrue]/Z, [%[a_ptr], #0x50]\n"
                "fmla	z13.s, %[b0].s, %[a1a].s[1]\n"
                "fmla	z14.s, %[b0].s, %[a1a].s[2]\n"
                "fmla	z15.s, %[b0].s, %[a1a].s[3]\n"
                "ld1w	%[b0].s, %[ptrue]/Z, [%[b_ptr], #6, MUL VL]\n"

                "fmla	z16.s, %[b1].s, %[a0a].s[0]\n"
                "fmla	z17.s, %[b1].s, %[a0a].s[1]\n"
                //ASM_PREFETCH("[%[b_ptr], #512]")
                "fmla	z18.s, %[b1].s, %[a0a].s[2]\n"
                "fmla	z19.s, %[b1].s, %[a0a].s[3]\n"
                "fmla	z20.s, %[b1].s, %[a1a].s[0]\n"
                "fmla	z21.s, %[b1].s, %[a1a].s[1]\n"
                "fmla	z22.s, %[b1].s, %[a1a].s[2]\n"
                "fmla	z23.s, %[b1].s, %[a1a].s[3]\n"
                "ld1w	%[b1].s, %[ptrue]/Z, [%[b_ptr], #7, MUL VL]\n"

                "fmla	z24.s, %[b2].s, %[a0a].s[0]\n"
                "fmla	z25.s, %[b2].s, %[a0a].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #0x40\n"
                "fmla	z26.s, %[b2].s, %[a0a].s[2]\n"
                "fmla	z27.s, %[b2].s, %[a0a].s[3]\n"
                "incb	%[b_ptr], ALL, MUL #6\n"
                "fmla	z28.s, %[b2].s, %[a1a].s[0]\n"
                "fmla	z29.s, %[b2].s, %[a1a].s[1]\n"
                "subs	%w[k], %w[k], #1\n"
                "fmla	z30.s, %[b2].s, %[a1a].s[2]\n"
                "fmla	z31.s, %[b2].s, %[a1a].s[3]\n"
                "bne	1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz	%w[oddk], 2f\n"

                // Detached final iteration (even K)
                "fmla 	z8.s , %[b0].s, %[a0].s[0]\n"
                "fmla	z9.s , %[b0].s, %[a0].s[1]\n"
                "ld1w	%[b2].s, %[ptrue]/Z, [%[b_ptr], #2, MUL VL]\n"
                "fmla	z10.s, %[b0].s, %[a0].s[2]\n"
                "fmla	z11.s, %[b0].s, %[a0].s[3]\n"
                "ld1rqw	%[a0a].s, %[ptrue]/Z, [%[a_ptr], #0x20]\n"
                "fmla 	z12.s, %[b0].s, %[a1].s[0]\n"
                "fmla	z13.s, %[b0].s, %[a1].s[1]\n"
                "ld1rqw	%[a1a].s, %[ptrue]/Z, [%[a_ptr], #0x30]\n"
                "fmla	z14.s, %[b0].s, %[a1].s[2]\n"
                "fmla	z15.s, %[b0].s, %[a1].s[3]\n"
                "ld1w	%[b0].s, %[ptrue]/Z, [%[b_ptr], #3, MUL VL]\n"

                "fmla	z16.s, %[b1].s, %[a0].s[0]\n"
                "fmla	z17.s, %[b1].s, %[a0].s[1]\n"
                "fmla	z18.s, %[b1].s, %[a0].s[2]\n"
                "fmla	z19.s, %[b1].s, %[a0].s[3]\n"
                "fmla	z20.s, %[b1].s, %[a1].s[0]\n"
                "fmla	z21.s, %[b1].s, %[a1].s[1]\n"
                "fmla	z22.s, %[b1].s, %[a1].s[2]\n"
                "fmla	z23.s, %[b1].s, %[a1].s[3]\n"
                "ld1w	%[b1].s, %[ptrue]/Z, [%[b_ptr], #4, MUL VL]\n"

                "fmla	z24.s, %[b2].s, %[a0].s[0]\n"
                "fmla	z25.s, %[b2].s, %[a0].s[1]\n"
                "add	%[a_ptr], %[a_ptr], #64\n"
                "fmla	z26.s, %[b2].s, %[a0].s[2]\n"
                "fmla	z27.s, %[b2].s, %[a0].s[3]\n"
                "fmla	z28.s, %[b2].s, %[a1].s[0]\n"
                "fmla	z29.s, %[b2].s, %[a1].s[1]\n"
                "fmla	z30.s, %[b2].s, %[a1].s[2]\n"
                "fmla	z31.s, %[b2].s, %[a1].s[3]\n"
                "ld1w	%[b2].s, %[ptrue]/Z, [%[b_ptr], #5, MUL VL]\n"

                "fmla 	z8.s , %[b0].s, %[a0a].s[0]\n"
                "fmla	z16.s, %[b1].s, %[a0a].s[0]\n"
                "incb	%[b_ptr], ALL, MUL #6\n"
                "fmla	z9.s , %[b0].s, %[a0a].s[1]\n"
                "st1w	z8.s, %[ptrue], [%[c_ptr]]\n"
                "fmla	z17.s, %[b1].s, %[a0a].s[1]\n"
                "st1w	z16.s, %[ptrue], [%[c_ptr], #1, MUL VL]\n"
                "fmla	z24.s, %[b2].s, %[a0a].s[0]\n"
                "st1w	z24.s, %[ptrue], [%[c_ptr], #2, MUL VL]\n"

                "fmla	z25.s, %[b2].s, %[a0a].s[1]\n"
                "st1w	z9.s, %[ptrue], [%[c_ptr], #3, MUL VL]\n"
                "fmla	z10.s, %[b0].s, %[a0a].s[2]\n"
                "st1w	z17.s, %[ptrue], [%[c_ptr], #4, MUL VL]\n"
                "fmla	z18.s, %[b1].s, %[a0a].s[2]\n"
                "st1w	z25.s, %[ptrue], [%[c_ptr], #5, MUL VL]\n"
                "fmla	z26.s, %[b2].s, %[a0a].s[2]\n"
                "st1w	z10.s, %[ptrue], [%[c_ptr], #6, MUL VL]\n"

                "fmla	z11.s, %[b0].s, %[a0a].s[3]\n"
                "st1w	z18.s, %[ptrue], [%[c_ptr], #7, MUL VL]\n"
                "incb	%[c_ptr], all, mul #12\n"
                "fmla	z19.s, %[b1].s, %[a0a].s[3]\n"
                "st1w	z26.s, %[ptrue], [%[c_ptr], #-4, MUL VL]\n"
                "fmla	z27.s, %[b2].s, %[a0a].s[3]\n"
                "st1w	z11.s, %[ptrue], [%[c_ptr], #-3, MUL VL]\n"

                "fmla 	z12.s, %[b0].s, %[a1a].s[0]\n"
                "st1w	z19.s, %[ptrue], [%[c_ptr], #-2, MUL VL]\n"
                "fmla	z20.s, %[b1].s, %[a1a].s[0]\n"
                "st1w	z27.s, %[ptrue], [%[c_ptr], #-1, MUL VL]\n"
                "fmla	z28.s, %[b2].s, %[a1a].s[0]\n"
                "st1w	z12.s, %[ptrue], [%[c_ptr]]\n"

                "fmla	z13.s, %[b0].s, %[a1a].s[1]\n"
                "st1w	z20.s, %[ptrue], [%[c_ptr], #1, MUL VL]\n"
                "fmla	z21.s, %[b1].s, %[a1a].s[1]\n"
                "st1w	z28.s, %[ptrue], [%[c_ptr], #2, MUL VL]\n"
                "fmla	z29.s, %[b2].s, %[a1a].s[1]\n"
                "st1w	z13.s, %[ptrue], [%[c_ptr], #3, MUL VL]\n"

                "fmla	z14.s, %[b0].s, %[a1a].s[2]\n"
                "st1w	z21.s, %[ptrue], [%[c_ptr], #4, MUL VL]\n"
                "fmla	z22.s, %[b1].s, %[a1a].s[2]\n"
                "st1w	z29.s, %[ptrue], [%[c_ptr], #5, MUL VL]\n"
                "fmla	z30.s, %[b2].s, %[a1a].s[2]\n"
                "st1w	z14.s, %[ptrue], [%[c_ptr], #6, MUL VL]\n"

                "fmla	z15.s, %[b0].s, %[a1a].s[3]\n"
                "st1w	z22.s, %[ptrue], [%[c_ptr], #7, MUL VL]\n"
                "incb	%[c_ptr], all, mul #12\n"
                "fmla	z23.s, %[b1].s, %[a1a].s[3]\n"
                "st1w	z30.s, %[ptrue], [%[c_ptr], #-4, MUL VL]\n"
                "fmla	z31.s, %[b2].s, %[a1a].s[3]\n"
                "st1w	z15.s, %[ptrue], [%[c_ptr], #-3, MUL VL]\n"

                "b	3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "fmla 	z8.s , %[b0].s, %[a0].s[0]\n"
                "ld1w	%[b2].s, %[ptrue]/Z, [%[b_ptr], #2, MUL VL]\n"
                "fmla	z16.s, %[b1].s, %[a0].s[0]\n"
                "fmla	z9.s , %[b0].s, %[a0].s[1]\n"
                "st1w	z8.s, %[ptrue], [%[c_ptr]]\n"
                "fmla	z17.s, %[b1].s, %[a0].s[1]\n"
                "st1w	z16.s, %[ptrue], [%[c_ptr], #1, MUL VL]\n"
                "fmla	z24.s, %[b2].s, %[a0].s[0]\n"
                "incb	%[b_ptr], all, mul #3\n"
                "add	%[a_ptr], %[a_ptr], #32\n"
                "st1w	z24.s, %[ptrue], [%[c_ptr], #2, MUL VL]\n"
                "fmla	z25.s, %[b2].s, %[a0].s[1]\n"
                "st1w	z9.s, %[ptrue], [%[c_ptr], #3, MUL VL]\n"

                "fmla	z10.s, %[b0].s, %[a0].s[2]\n"
                "st1w	z17.s, %[ptrue], [%[c_ptr], #4, MUL VL]\n"
                "fmla	z18.s, %[b1].s, %[a0].s[2]\n"
                "st1w	z25.s, %[ptrue], [%[c_ptr], #5, MUL VL]\n"
                "fmla	z26.s, %[b2].s, %[a0].s[2]\n"
                "st1w	z10.s, %[ptrue], [%[c_ptr], #6, MUL VL]\n"

                "fmla	z11.s, %[b0].s, %[a0].s[3]\n"
                "st1w	z18.s, %[ptrue], [%[c_ptr], #7, MUL VL]\n"
                "incb	%[c_ptr], all, mul #12\n"
                "fmla	z19.s, %[b1].s, %[a0].s[3]\n"
                "st1w	z26.s, %[ptrue], [%[c_ptr], #-4, MUL VL]\n"
                "fmla	z27.s, %[b2].s, %[a0].s[3]\n"
                "st1w	z11.s, %[ptrue], [%[c_ptr], #-3, MUL VL]\n"

                "fmla 	z12.s, %[b0].s, %[a1].s[0]\n"
                "st1w	z19.s, %[ptrue], [%[c_ptr], #-2, MUL VL]\n"
                "fmla	z20.s, %[b1].s, %[a1].s[0]\n"
                "st1w	z27.s, %[ptrue], [%[c_ptr], #-1, MUL VL]\n"
                "fmla	z28.s, %[b2].s, %[a1].s[0]\n"
                "st1w	z12.s, %[ptrue], [%[c_ptr]]\n"

                "fmla	z13.s, %[b0].s, %[a1].s[1]\n"
                "st1w	z20.s, %[ptrue], [%[c_ptr], #1, MUL VL]\n"
                "fmla	z21.s, %[b1].s, %[a1].s[1]\n"
                "st1w	z28.s, %[ptrue], [%[c_ptr], #2, MUL VL]\n"
                "fmla	z29.s, %[b2].s, %[a1].s[1]\n"
                "st1w	z13.s, %[ptrue], [%[c_ptr], #3, MUL VL]\n"

                "fmla	z14.s, %[b0].s, %[a1].s[2]\n"
                "st1w	z21.s, %[ptrue], [%[c_ptr], #4, MUL VL]\n"
                "fmla	z22.s, %[b1].s, %[a1].s[2]\n"
                "st1w	z29.s, %[ptrue], [%[c_ptr], #5, MUL VL]\n"
                "fmla	z30.s, %[b2].s, %[a1].s[2]\n"
                "st1w	z14.s, %[ptrue], [%[c_ptr], #6, MUL VL]\n"

                "fmla	z15.s, %[b0].s, %[a1].s[3]\n"
                "st1w	z22.s, %[ptrue], [%[c_ptr], #7, MUL VL]\n"
                "incb	%[c_ptr], all, mul #12\n"
                "fmla	z23.s, %[b1].s, %[a1].s[3]\n"
                "st1w	z30.s, %[ptrue], [%[c_ptr], #-4, MUL VL]\n"
                "fmla	z31.s, %[b2].s, %[a1].s[3]\n"
                "st1w	z15.s, %[ptrue], [%[c_ptr], #-3, MUL VL]\n"

                // Common tail
                "3:\n"
                "st1w	z23.s, %[ptrue], [%[c_ptr], #-2, MUL VL]\n"
                "st1w	z31.s, %[ptrue], [%[c_ptr], #-1, MUL VL]\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk), [ptrue] "Upl" (ptrue)
            : "x20", "x21", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18",
              "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
              "cc", "memory"
            );
        }
    }
}

} // namespace arm_gemm

#endif // __ARM_FEATURE_SVE
