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

// AArch64 only, and either the FP16_KERNELS option set or the target explicitly supports FP16 vectors.
#if defined(__aarch64__) && (defined(FP16_KERNELS) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))

template<>
inline void MergeResults<24, 8>(__fp16 *out, const __fp16 *in, const int ldout, const int y0, const int ymax,
                         const int x0, const int xmax, const __fp16 alpha, const __fp16 beta) {
    const __fp16 *inptr = in;
    prefetch_6x(inptr);
    prefetch_6x(inptr + 48);

    float16x8_t va = vdupq_n_f16(alpha);
    float16x8_t vb = vdupq_n_f16(beta);

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

        for (int i=x0; i<xmax; i+=24) {
            __fp16 dummyres[24];

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

            if (beta == (__fp16)0.0f) {
                /* If beta===0, don't read the output. */

                /* For ragged X, manually copy over the valid results. */
                if ((i+23) >= xmax) {
                    for (int xi=0; xi<24; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = (alpha * inptr[xi]);
                            outptr0++;
                            *outptr1 = (alpha * inptr[xi + 24]);
                            outptr1++;
                            *outptr2 = (alpha * inptr[xi + 48]);
                            outptr2++;
                            *outptr3 = (alpha * inptr[xi + 72]);
                            outptr3++;
                            *outptr4 = (alpha * inptr[xi + 96]);
                            outptr4++;
                            *outptr5 = (alpha * inptr[xi + 120]);
                            outptr5++;
                            *outptr6 = (alpha * inptr[xi + 144]);
                            outptr6++;
                            *outptr7 = (alpha * inptr[xi + 168]);
                            outptr7++;
                        }
                    }
                    inptr += 192;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        ".arch	armv8.2-a+fp16\n"
#endif
                        // Rows 0-1
                        ASM_PREFETCH("[%[inptr], #768]")
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMUL	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FMUL	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FMUL	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMUL	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FMUL	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FMUL	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr1]], #16\n"
                        ASM_PREFETCH("[%[inptr], #960]")

                        // Rows 2-3
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMUL	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FMUL	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FMUL	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMUL	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr0], #80]")
                        "FMUL	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FMUL	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr3]], #16\n"
                        ASM_PREFETCH("[%[outptr1], #80]")

                        // Rows 4-5
                        ASM_PREFETCH("[%[outptr2], #80]")
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMUL	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr3], #80]")
                        "FMUL	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FMUL	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMUL	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr4], #80]")
                        "FMUL	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FMUL	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        ASM_PREFETCH("[%[outptr5], #80]")
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMUL	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr6], #128]")
                        "FMUL	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMUL	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMUL	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr7], #128]")
                        "FMUL	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMUL	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                      [inptr] "+r" (inptr)
                    : [va] "w" (va), [vb] "w" (vb)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21"
                    );
                }
            } else {
                /* For ragged X, manually copy over the valid results. */
                if ((i+23) >= xmax) {
                    for (int xi=0; xi<24; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                            outptr0++;
                            *outptr1 = (alpha * inptr[xi + 24]) + (*outptr1 * beta);
                            outptr1++;
                            *outptr2 = (alpha * inptr[xi + 48]) + (*outptr2 * beta);
                            outptr2++;
                            *outptr3 = (alpha * inptr[xi + 72]) + (*outptr3 * beta);
                            outptr3++;
                            *outptr4 = (alpha * inptr[xi + 96]) + (*outptr4 * beta);
                            outptr4++;
                            *outptr5 = (alpha * inptr[xi + 120]) + (*outptr5 * beta);
                            outptr5++;
                            *outptr6 = (alpha * inptr[xi + 144]) + (*outptr6 * beta);
                            outptr6++;
                            *outptr7 = (alpha * inptr[xi + 168]) + (*outptr7 * beta);
                            outptr7++;
                        }
                    }
                    inptr += 192;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        ".arch	armv8.2-a+fp16\n"
#endif
                        // Rows 0-1
                        "LDP	q16, q17, [%[outptr0]]\n"
                        "FMUL	v16.8h, v16.8h, %[vb].8h\n"
                        "LDR	q18, [%[outptr0], #32]\n"
                        "FMUL	v17.8h, v17.8h, %[vb].8h\n"
                        "LDP	q19, q20, [%[outptr1]]\n"
                        "FMUL	v18.8h, v18.8h, %[vb].8h\n"
                        ASM_PREFETCH("[%[inptr], #768]")
                        "LDR	q21, [%[outptr1], #32]\n"
                        "FMUL	v19.8h, v19.8h, %[vb].8h\n"
                        "LDP	q0,  q1,  [%[inptr]]\n"
                        "FMUL	v20.8h, v20.8h, %[vb].8h\n"
                        "LDP	q2,  q3,  [%[inptr], #32]\n"
                        "FMUL	v21.8h, v21.8h, %[vb].8h\n"
                        "LDP	q4,  q5,  [%[inptr], #64]\n"
                        "FMLA	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #832]")
                        "FMLA	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr0]], #32\n"
                        "FMLA	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr0]], #16\n"
                        "FMLA	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #896]")
                        "FMLA	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr1]], #32\n"
                        "FMLA	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr1]], #16\n"
                        ASM_PREFETCH("[%[inptr], #960]")

                        // Rows 2-3
                        "LDP	q16, q17, [%[outptr2]]\n"
                        "FMUL	v16.8h, v16.8h, %[vb].8h\n"
                        "LDR	q18, [%[outptr2], #32]\n"
                        "FMUL	v17.8h, v17.8h, %[vb].8h\n"
                        "LDP	q19, q20, [%[outptr3]]\n"
                        "FMUL	v18.8h, v18.8h, %[vb].8h\n"
                        ASM_PREFETCH("[%[inptr], #1024]")
                        "LDR	q21, [%[outptr3], #32]\n"
                        "FMUL	v19.8h, v19.8h, %[vb].8h\n"
                        "LDP	q0,  q1,  [%[inptr], #96]\n"
                        "FMUL	v20.8h, v20.8h, %[vb].8h\n"
                        "LDP	q2,  q3,  [%[inptr], #128]\n"
                        "FMUL	v21.8h, v21.8h, %[vb].8h\n"
                        "LDP	q4,  q5,  [%[inptr], #160]\n"
                        "FMLA	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[inptr], #1088]")
                        "FMLA	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr2]], #32\n"
                        "FMLA	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr2]], #16\n"
                        "FMLA	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr0], #80]")
                        "FMLA	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr3]], #32\n"
                        "FMLA	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr3]], #16\n"
                        ASM_PREFETCH("[%[outptr1], #80]")

                        // Rows 4-5
                        "LDP	q16, q17, [%[outptr4]]\n"
                        "FMUL	v16.8h, v16.8h, %[vb].8h\n"
                        "LDR	q18, [%[outptr4], #32]\n"
                        "FMUL	v17.8h, v17.8h, %[vb].8h\n"
                        "LDP	q19, q20, [%[outptr5]]\n"
                        "FMUL	v18.8h, v18.8h, %[vb].8h\n"
                        ASM_PREFETCH("[%[outptr2], #80]")
                        "LDR	q21, [%[outptr5], #32]\n"
                        "FMUL	v19.8h, v19.8h, %[vb].8h\n"
                        "LDP	q0,  q1,  [%[inptr], #192]\n"
                        "FMUL	v20.8h, v20.8h, %[vb].8h\n"
                        "LDP	q2,  q3,  [%[inptr], #224]\n"
                        "FMUL	v21.8h, v21.8h, %[vb].8h\n"
                        "LDP	q4,  q5,  [%[inptr], #256]\n"
                        "FMLA	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr3], #80]")
                        "FMLA	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr4]], #32\n"
                        "FMLA	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr4]], #16\n"
                        "FMLA	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr4], #80]")
                        "FMLA	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr5]], #32\n"
                        "FMLA	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr5]], #16\n"

                        // Rows 6-7
                        "LDP	q16, q17, [%[outptr6]]\n"
                        "FMUL	v16.8h, v16.8h, %[vb].8h\n"
                        "LDR	q18, [%[outptr6], #32]\n"
                        "FMUL	v17.8h, v17.8h, %[vb].8h\n"
                        "LDP	q19, q20, [%[outptr7]]\n"
                        ASM_PREFETCH("[%[outptr5], #80]")
                        "FMUL	v18.8h, v18.8h, %[vb].8h\n"
                        "LDR	q21, [%[outptr7], #32]\n"
                        "FMUL	v19.8h, v19.8h, %[vb].8h\n"
                        "LDP	q0,  q1,  [%[inptr], #288]\n"
                        "FMUL	v20.8h, v20.8h, %[vb].8h\n"
                        "LDP	q2,  q3,  [%[inptr], #320]\n"
                        "FMUL	v21.8h, v21.8h, %[vb].8h\n"
                        "LDP	q4,  q5,  [%[inptr], #352]\n"
                        "FMLA	v16.8h, v0.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr6], #128]")
                        "FMLA	v17.8h, v1.8h, %[va].8h\n"
                        "STP	q16, q17, [%[outptr6]], #32\n"
                        "FMLA	v18.8h, v2.8h, %[va].8h\n"
                        "STR	q18, [%[outptr6]], #16\n"
                        "FMLA	v19.8h, v3.8h, %[va].8h\n"
                        ASM_PREFETCH("[%[outptr7], #128]")
                        "FMLA	v20.8h, v4.8h, %[va].8h\n"
                        "STP	q19, q20, [%[outptr7]], #32\n"
                        "FMLA	v21.8h, v5.8h, %[va].8h\n"
                        "STR	q21, [%[outptr7]], #16\n"
                        "ADD	%[inptr], %[inptr], #384\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                      [inptr] "+r" (inptr)
                    : [va] "w" (va), [vb] "w" (vb)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21"
                    );
                }
            }
        }
    }
}

#endif // __aarch64__ && (FP16_KERNELS || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
