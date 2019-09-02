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

template<>
inline void MergeResults<12, 8, false>(float *out, const float *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const float alpha, const float beta) {
    const float *inptr = in;
    prefetch_6x(inptr);
    prefetch_6x(inptr + 96);

    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

    for (int y=y0; y<ymax; y+=8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        prefetch_2x(outptr0);
        prefetch_2x(outptr1);
        prefetch_2x(outptr2);
        prefetch_2x(outptr3);
        prefetch_2x(outptr4);
        prefetch_2x(outptr5);
        prefetch_2x(outptr6);
        prefetch_2x(outptr7);

        for (int i=x0; i<xmax; i+=12) {
            float dummyres[12];

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

            if (beta==0.0f) {
                /* If beta==0, don't read the original input at all. */

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
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        ASM_PREFETCH("[%[inptr], #768]")
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        ASM_PREFETCH("[%[inptr], #960]")
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FMUL	v16.4s, v0.4s, %[av].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FMUL	v17.4s, v1.4s, %[av].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMUL	v18.4s, v2.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMUL	v19.4s, v3.4s, %[av].4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMUL	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMUL	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
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
                        "LDP	q16, q17, [%[outptr0]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr1], #32]\n"
                        ASM_PREFETCH("[%[inptr], #768]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr1]], #16\n"

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr3], #32]\n"
                        ASM_PREFETCH("[%[inptr], #960]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr3]], #16\n"

                        // Rows 4-5
                        ASM_PREFETCH("[%[outptr0], #80]")
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr5], #32]\n"
                        ASM_PREFETCH("[%[outptr1], #80]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr2], #80]")
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr3], #80]")
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        ASM_PREFETCH("[%[outptr4], #80]")
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "FMUL	v16.4s, v16.4s, %[bv].4s\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "FMUL	v17.4s, v17.4s, %[bv].4s\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        "FMUL	v18.4s, v18.4s, %[bv].4s\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        ASM_PREFETCH("[%[outptr5], #80]")
                        "FMUL	v19.4s, v19.4s, %[bv].4s\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FMUL	v20.4s, v20.4s, %[bv].4s\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FMUL	v21.4s, v21.4s, %[bv].4s\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMLA	v16.4s, v0.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr6], #128]")
                        "FMLA	v17.4s, v1.4s, %[av].4s\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMLA	v18.4s, v2.4s, %[av].4s\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMLA	v19.4s, v3.4s, %[av].4s\n"
                        ASM_PREFETCH("[%[outptr7], #128]")
                        "FMLA	v20.4s, v4.4s, %[av].4s\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMLA	v21.4s, v5.4s, %[av].4s\n"
                        "STR	q21, [%[outptr7]], #16\n"
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

#endif // __aarch64__
