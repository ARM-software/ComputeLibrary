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

#ifdef __arm__

#include "../asmlib.hpp"

#include <arm_neon.h>

template<>
inline void MergeResults<8, 6>(float *out, const float *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const float alpha, const float beta) {
    const float *inptr = in;
//    prefetch_6x(inptr);
//    prefetch_6x(inptr + 96);

    float32x4_t av = vdupq_n_f32(alpha);
    float32x4_t bv = vdupq_n_f32(beta);

    for (int y=y0; y<ymax; y+=8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

//        prefetch_2x(outptr0);
//        prefetch_2x(outptr1);
//        prefetch_2x(outptr2);
//        prefetch_2x(outptr3);
//        prefetch_2x(outptr4);
//        prefetch_2x(outptr5);

        for (int i=x0; i<xmax; i+=8) {
            float dummyres[8];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                    case 3:
                        outptr2 = dummyres;
                    case 2:
                        outptr3 = dummyres;
                    case 1:
                        outptr4 = dummyres;
                    case 0:
                        outptr5 = dummyres;
                    default:
                        break;
                }
            }

            /* For ragged X, manually copy over the valid results. */
            if ((i+7) >= xmax) {
                for (int xi=0; xi<8; xi++) {
                    if ((i+xi) < xmax) {
                        *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                        outptr0++;
                        *outptr1 = (alpha * inptr[xi + 8]) + (*outptr1 * beta);
                        outptr1++;
                        *outptr2 = (alpha * inptr[xi + 16]) + (*outptr2 * beta);
                        outptr2++;
                        *outptr3 = (alpha * inptr[xi + 24]) + (*outptr3 * beta);
                        outptr3++;
                        *outptr4 = (alpha * inptr[xi + 32]) + (*outptr4 * beta);
                        outptr4++;
                        *outptr5 = (alpha * inptr[xi + 40]) + (*outptr5 * beta);
                        outptr5++;
                    }
                }
                inptr += 48;
            } else {
                /* Optimized routine to copy an entire block */
                __asm __volatile (
                    // Rows 0-1
                    "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                    "VMUL.f32	q4, q4, %q[bv]\n"
                    "VLD1.32	{d12-d15}, [%[outptr1]]\n"
                    "VMUL.f32	q5, q5, %q[bv]\n"
                    "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                    "VMUL.f32	q6, q6, %q[bv]\n"
                    "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                    "VMUL.f32	q7, q7, %q[bv]\n"

                    "VMLA.f32	q4, q0, %q[av]\n"
                    ASM_PREFETCH("[%[inptr], #352]")
                    "VMLA.f32	q5, q1, %q[av]\n"
                    "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                    ASM_PREFETCH("[%[inptr], #416]")
                    "VMLA.f32	q6, q2, %q[av]\n"
                    ASM_PREFETCH("[%[inptr], #480]")
                    "VMLA.f32	q7, q3, %q[av]\n"
                    "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                    // Rows 2-3
                    "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                    "VMUL.f32	q4, q4, %q[bv]\n"
                    "VLD1.32	{d12-d15}, [%[outptr3]]\n"
                    "VMUL.f32	q5, q5, %q[bv]\n"
                    "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                    "VMUL.f32	q6, q6, %q[bv]\n"
                    "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                    "VMUL.f32	q7, q7, %q[bv]\n"

                    "VMLA.f32	q4, q0, %q[av]\n"
                    ASM_PREFETCH("[%[outptr0], #96]")
                    "VMLA.f32	q5, q1, %q[av]\n"
                    "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                    ASM_PREFETCH("[%[outptr1], #96]")
                    "VMLA.f32	q6, q2, %q[av]\n"
                    ASM_PREFETCH("[%[outptr2], #96]")
                    "VMLA.f32	q7, q3, %q[av]\n"
                    "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                    // Rows 4-5
                    "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                    "VMUL.f32	q4, q4, %q[bv]\n"
                    "VLD1.32	{d12-d15}, [%[outptr5]]\n"
                    "VMUL.f32	q5, q5, %q[bv]\n"
                    "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                    "VMUL.f32	q6, q6, %q[bv]\n"
                    "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                    "VMUL.f32	q7, q7, %q[bv]\n"

                    "VMLA.f32	q4, q0, %q[av]\n"
                    ASM_PREFETCH("[%[outptr3], #96]")
                    "VMLA.f32	q5, q1, %q[av]\n"
                    "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                    ASM_PREFETCH("[%[outptr4], #96]")
                    "VMLA.f32	q6, q2, %q[av]\n"
                    ASM_PREFETCH("[%[outptr5], #128]")
                    "VMLA.f32	q7, q3, %q[av]\n"
                    "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                  [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [inptr] "+r" (inptr)
                : [av] "w" (av), [bv] "w" (bv)
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                );
            }
        }
    }
}

#endif // __arm__
