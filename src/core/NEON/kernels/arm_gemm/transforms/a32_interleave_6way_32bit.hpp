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

#ifdef __arm__

#include <arm_neon.h>

#include "../asmlib.hpp"

template<>
template<typename T>
inline void TransformImpl<6, 1, false, 4, 4, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint32_t *outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    uint32_t zerobuff[16] = { 0 }; // 8 for asm loop plus up to 7 for overflow loop

    for (int y=y0; y<ymax; y+=6) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;

        //prefetch_2x(inptr0);
        //prefetch_2x(inptr1);
        //prefetch_2x(inptr2);
        //prefetch_2x(inptr3);
        //prefetch_2x(inptr4);
        //prefetch_2x(inptr5);

        int x=(kmax-k0);
        for (;x>7;x-=8) {
            /* Cope with ragged cases by copying from a buffer of zeroes instead */
            if ((y + 5) >= ymax) {
                switch ((y + 5) - ymax) {
                    /* Everything falls through in here */
                    case 4:
                        inptr1 = zerobuff;
                        // fall through
                    case 3:
                        inptr2 = zerobuff;
                        // fall through
                    case 2:
                        inptr3 = zerobuff;
                        // fall through
                    case 1:
                        inptr4 = zerobuff;
                        // fall through
                    case 0:
                        inptr5 = zerobuff;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }


            __asm __volatile (
                // Load up 8 elements (2 vectors) from each of 8 sources.
                "VLD1.32	{d0-d3}, [%[inptr0]]!\n"   // q0=A0A1A2A3
                "VLD1.32	{d4-d7}, [%[inptr1]]!\n"   // q2=B0B1B2B3
                "VLD1.32	{d8-d11}, [%[inptr2]]!\n"  // q4=C0C1C2C3
                "VZIP.32	q0, q4\n"     // q0=A0C0A1C1, q4 = A2C2A3C3
                "VLD1.32	{d12-d15}, [%[inptr3]]!\n" // q6=D0D1D2D3
                "VZIP.32	q2, q6\n"     // q2=B0D0B1D1, q6 = B2D2B3D3
                "VLD1.32	{d16-d19}, [%[inptr4]]!\n"
                "VLD1.32	{d20-d23}, [%[inptr5]]!\n"
                "VZIP.32	q8, q10\n"    // q8=E0F0E1F1, q10 = E2F2E3F3
                ASM_PREFETCH("[%[inptr0], #128]")
                "VZIP.32	q0, q2\n"    // q0 = A0B0C0D0, q2 = A1B1C1D1

                // Store first elements
                "VST1.32	{d0-d1}, [%[outptr]]!\n"
                "VST1.32	{d16}, [%[outptr]]!\n"

                "VZIP.32	q4, q6\n"    // q4 = A2B2C2D2, q6 = A3B3C3D3

                // Store second elements
                "VST1.32	{d4-d5}, [%[outptr]]!\n"
                "VZIP.32	q1, q5\n"
                ASM_PREFETCH("[%[inptr1], #128]")
                "VST1.32	{d17}, [%[outptr]]!\n"
                "VZIP.32	q3, q7\n"

                // Store third elements
                "VZIP.32	q9, q11\n"
                "VST1.32	{d8-d9}, [%[outptr]]!\n"
                "VZIP.32	q1, q3\n"
                ASM_PREFETCH("[%[inptr2], #128]")
                "VST1.32	{d20}, [%[outptr]]!\n"

                // Store fourth elements
                "VZIP.32	q5, q7\n"
                "VST1.32	{d12-d13}, [%[outptr]]!\n"
                ASM_PREFETCH("[%[inptr3], #128]")
                "VST1.32	{d21}, [%[outptr]]!\n"

                // Fifth
                "VST1.32	{d2-d3}, [%[outptr]]!\n"
                ASM_PREFETCH("[%[inptr4], #128]")
                "VST1.32	{d18}, [%[outptr]]!\n"

                // Sixth
                "VST1.32	{d6-d7}, [%[outptr]]!\n"
                ASM_PREFETCH("[%[inptr5], #128]")
                "VST1.32	{d19}, [%[outptr]]!\n"

                // Seventh
                "VST1.32	{d10-d11}, [%[outptr]]!\n"
                "VST1.32	{d22}, [%[outptr]]!\n"

                // Eighth
                "VST1.32	{d14-d15}, [%[outptr]]!\n"
                "VST1.32	{d23}, [%[outptr]]!\n"

                : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
                  [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [outptr] "+r" (outptr)
                :
                : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "memory"
            );
        }

        for (;x>0;x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
        }
    }
}

#endif  // __arm__
