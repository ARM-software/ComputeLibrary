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

// This should be possible on any AArch64 target, but some old compilers don't support __fp16 arguments.
#if defined(__aarch64__) && defined(__ARM_FP16_ARGS)

#include <arm_neon.h>

template<>
inline void MergeResults<12,8,false>(__fp16 *out, const float *in, int ldout, int y0, int ymax, int x0, int xmax, const __fp16 alpha, const __fp16 beta) {
    const float *inptr = in;
    prefetch_6x(inptr);
    prefetch_6x(inptr + 24);

    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

    for (int y=y0; y<ymax; y+=8) {
        __fp16 *outptr0 = out + (y * ldout) + x0;
        __fp16 *outptr1 = outptr0 + ldout;
        __fp16 *outptr2 = outptr1 + ldout;
        __fp16 *outptr3 = outptr2 + ldout;
        __fp16 *outptr4 = outptr3 + ldout;
        __fp16 *outptr5 = outptr4 + ldout;
        __fp16 *outptr6 = outptr5 + ldout;
        __fp16 *outptr7 = outptr6 + ldout;

        prefetch_2x(outptr0);
        prefetch_2x(outptr1);
        prefetch_2x(outptr2);
        prefetch_2x(outptr3);
        prefetch_2x(outptr4);
        prefetch_2x(outptr5);
        prefetch_2x(outptr6);
        prefetch_2x(outptr7);

        for (int i=x0; i<xmax; i+=12) {
            __fp16 dummyres[12];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+7) >= ymax) {
                switch ((y + 7) - ymax) {
                    case 6:
                        outptr1 = dummyres;
                        // fall through
                    case 5:
                        outptr2 = dummyres;
                        // fall through
                    case 4:
                        outptr3 = dummyres;
                        // fall through
                    case 3:
                        outptr4 = dummyres;
                        // fall through
                    case 2:
                        outptr5 = dummyres;
                        // fall through
                    case 1:
                        outptr6 = dummyres;
                        // fall through
                    case 0:
                        outptr7 = dummyres;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            if (beta == ((__fp16)0.0f)) {
                /* If beta==0, don't read the output. */
                /* For ragged X, manually copy over the valid results. */
                if ((i+11) >= xmax) {
                    for (int xi=0; xi<12; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = (alpha * inptr[xi]);
                            outptr0++;
                            *outptr1 = (alpha * inptr[xi + 12]);
                            outptr1++;
                            *outptr2 = (alpha * inptr[xi + 24]);
                            outptr2++;
                            *outptr3 = (alpha * inptr[xi + 36]);
                            outptr3++;
                            *outptr4 = (alpha * inptr[xi + 48]);
                            outptr4++;
                            *outptr5 = (alpha * inptr[xi + 60]);
                            outptr5++;
                            *outptr6 = (alpha * inptr[xi + 72]);
                            outptr6++;
                            *outptr7 = (alpha * inptr[xi + 84]);
                            outptr7++;
                        }
                    }
                    inptr += 96;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
                        // Rows 0-1
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #768]")
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FCVTN	v16.4h, v16.4s\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FCVTN2	v16.8h, v17.4s\n"
                        ASM_PREFETCH("[%[inptr], #960]")
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr0]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr0]], #8\n"
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr1]], #16\n"
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr1]], #8\n"

                        // Rows 2-3
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FCVTN	v16.4h, v16.4s\n"
                        ASM_PREFETCH("[%[outptr0], #64]")
                        "FCVTN2	v16.8h, v17.4s\n"
                        ASM_PREFETCH("[%[outptr1], #64]")
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr2]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr2]], #8\n"
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr3]], #16\n"
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr3]], #8\n"

                        // Rows 4-5
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr2], #64]")
                        "FCVTN	v16.4h, v16.4s\n"
                        ASM_PREFETCH("[%[outptr3], #64]")
                        "FCVTN2	v16.8h, v17.4s\n"
                        ASM_PREFETCH("[%[outptr4], #88]")
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr4]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr4]], #8\n"
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr5]], #16\n"
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr5]], #8\n"

                        // Rows 6-7
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr5], #64]")
                        "FCVTN	v16.4h, v16.4s\n"
                        ASM_PREFETCH("[%[outptr6], #88]")
                        "FCVTN2	v16.8h, v17.4s\n"
                        ASM_PREFETCH("[%[outptr7], #88]")
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr6]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr6]], #8\n"
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr7]], #16\n"
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr7]], #8\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                      [inptr] "+r" (inptr)
                    : [av] "w" (av), [bv] "w" (bv)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21"
                    );
                }
            } else {
                /* For ragged X, manually copy over the valid results. */
                if ((i+11) >= xmax) {
                    for (int xi=0; xi<12; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                            outptr0++;
                            *outptr1 = (alpha * inptr[xi + 12]) + (*outptr1 * beta);
                            outptr1++;
                            *outptr2 = (alpha * inptr[xi + 24]) + (*outptr2 * beta);
                            outptr2++;
                            *outptr3 = (alpha * inptr[xi + 36]) + (*outptr3 * beta);
                            outptr3++;
                            *outptr4 = (alpha * inptr[xi + 48]) + (*outptr4 * beta);
                            outptr4++;
                            *outptr5 = (alpha * inptr[xi + 60]) + (*outptr5 * beta);
                            outptr5++;
                            *outptr6 = (alpha * inptr[xi + 72]) + (*outptr6 * beta);
                            outptr6++;
                            *outptr7 = (alpha * inptr[xi + 84]) + (*outptr7 * beta);
                            outptr7++;
                        }
                    }
                    inptr += 96;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
                        // Rows 0-1
                        "LDR	q16, [%[outptr0]]\n"
                        "FCVTL2	v17.4s, v16.8h\n"
                        "LDR	d18, [%[outptr0], #16]\n"
                        "FCVTL	v16.4s, v16.4h\n"
                        "LDR	q19, [%[outptr1]]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDR	d21, [%[outptr1], #16]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "FCVTL	v18.4s, v18.4h\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FCVTL2	v20.4s, v19.8h\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FCVTL	v19.4s, v19.4h\n"
                        ASM_PREFETCH("[%[inptr], #768]")
                        "FCVTL	v21.4s, v21.4h\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[inptr], #960]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FCVTN	v16.4h, v16.4s\n"
                        "FCVTN2	v16.8h, v17.4s\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr0]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr0]], #8\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr1]], #16\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr1]], #8\n"

                        // Rows 2-3
                        "LDR	q16, [%[outptr2]]\n"
                        "FCVTL2	v17.4s, v16.8h\n"
                        "LDR	d18, [%[outptr2], #16]\n"
                        "FCVTL	v16.4s, v16.4h\n"
                        "LDR	q19, [%[outptr3]]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDR	d21, [%[outptr3], #16]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FCVTL	v18.4s, v18.4h\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FCVTL2	v20.4s, v19.8h\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FCVTL	v19.4s, v19.4h\n"
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "FCVTL	v21.4s, v21.4h\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[outptr0], #64]")
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[outptr1], #64]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FCVTN	v16.4h, v16.4s\n"
                        "FCVTN2	v16.8h, v17.4s\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr2]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr2]], #8\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr3]], #16\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr3]], #8\n"

                        // Rows 4-5
                        "LDR	q16, [%[outptr4]]\n"
                        "FCVTL2	v17.4s, v16.8h\n"
                        "LDR	d18, [%[outptr4], #16]\n"
                        "FCVTL	v16.4s, v16.4h\n"
                        "LDR	q19, [%[outptr5]]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDR	d21, [%[outptr5], #16]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FCVTL	v18.4s, v18.4h\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FCVTL2	v20.4s, v19.8h\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FCVTL	v19.4s, v19.4h\n"
                        ASM_PREFETCH("[%[outptr2], #64]")
                        "FCVTL	v21.4s, v21.4h\n"
                        ASM_PREFETCH("[%[outptr3], #64]")
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[outptr4], #88]")
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FCVTN	v16.4h, v16.4s\n"
                        "FCVTN2	v16.8h, v17.4s\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr4]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr4]], #8\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr5]], #16\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr5]], #8\n"

                        // Rows 6-7
                        "LDR	q16, [%[outptr6]]\n"
                        "FCVTL2	v17.4s, v16.8h\n"
                        "LDR	d18, [%[outptr6], #16]\n"
                        "FCVTL	v16.4s, v16.4h\n"
                        "LDR	q19, [%[outptr7]]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDR	d21, [%[outptr7], #16]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FCVTL	v18.4s, v18.4h\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FCVTL2	v20.4s, v19.8h\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FCVTL	v19.4s, v19.4h\n"
                        ASM_PREFETCH("[%[outptr5], #64]")
                        "FCVTL	v21.4s, v21.4h\n"
                        ASM_PREFETCH("[%[outptr6], #88]")
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        ASM_PREFETCH("[%[outptr7], #88]")
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "FCVTN	v16.4h, v16.4s\n"
                        "FCVTN2	v16.8h, v17.4s\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q16, [%[outptr6]], #16\n"
                        "FCVTN	v18.4h, v18.4s\n"
                        "STR	d18, [%[outptr6]], #8\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "FCVTN	v19.4h, v19.4s\n"
                        "FCVTN2	v19.8h, v20.4s\n"
                        "STR	q19, [%[outptr7]], #16\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "FCVTN	v21.4h, v21.4s\n"
                        "STR	d21, [%[outptr7]], #8\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                      [inptr] "+r" (inptr)
                    : [av] "w" (av), [bv] "w" (bv)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21"
                    );
                }
            }
        }
    }
}

#endif // __aarch64__ && __ARM_FP16_ARGS
