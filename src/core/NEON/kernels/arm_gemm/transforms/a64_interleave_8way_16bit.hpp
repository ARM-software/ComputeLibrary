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

#include <arm_neon.h>

#include "../asmlib.hpp"

template<>
template<typename T>
void TransformImpl<8, 1, false, 2, 2, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint16_t *outptr = (uint16_t *)out;
    const uint16_t *inptr = (const uint16_t *)in;

    uint16_t zerobuff[16] = { 0 }; // 8 for asm loop plus up to 7 for overflow loop

    for (int y=y0; y<ymax; y+=8) {
        const uint16_t *inptr0 = inptr + y * ldin + k0;
        const uint16_t *inptr1 = inptr0 + ldin;
        const uint16_t *inptr2 = inptr1 + ldin;
        const uint16_t *inptr3 = inptr2 + ldin;
        const uint16_t *inptr4 = inptr3 + ldin;
        const uint16_t *inptr5 = inptr4 + ldin;
        const uint16_t *inptr6 = inptr5 + ldin;
        const uint16_t *inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int x=(kmax-k0);
        for (;x>7;x-=8) {
            /* Cope with ragged cases by copying from a buffer of zeroes instead */
            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
                    /* Everything falls through in here */
                    case 6:
                        inptr1 = zerobuff;
                        // fall through
                    case 5:
                        inptr2 = zerobuff;
                        // fall through
                    case 4:
                        inptr3 = zerobuff;
                        // fall through
                    case 3:
                        inptr4 = zerobuff;
                        // fall through
                    case 2:
                        inptr5 = zerobuff;
                        // fall through
                    case 1:
                        inptr6 = zerobuff;
                        // fall through
                    case 0:
                        inptr7 = zerobuff;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            int skippf = (x & 31);
            __asm __volatile (
                // Load up 8 elements (1 vector) from each of 8 sources.
                "CBNZ	%w[skippf], 1f\n"
                ASM_PREFETCH("[%[inptr0], #128]")
                ASM_PREFETCH("[%[inptr1], #128]")
                ASM_PREFETCH("[%[inptr2], #128]")
                ASM_PREFETCH("[%[inptr3], #128]")
                "1:\n"

                "LDR	q0, [%[inptr0]], #16\n" // q0=A0A1A2A3A4A5A6A7
                "LDR	q4, [%[inptr4]], #16\n" // q8=E0E1E2E3E4E5E6E7
                "LDR	q2, [%[inptr2]], #16\n" // q4=C0C1C2C3...
                "LDR	q6, [%[inptr6]], #16\n"
                "ZIP1	v8.8h, v0.8h, v4.8h\n"  // q8=A0E0A1E1A2E2A3E3
                "ZIP2	v16.8h, v0.8h, v4.8h\n" // q16=A4E4A5E5A6E6A7E7
                "ZIP1	v9.8h, v2.8h, v6.8h\n"  // q9=C0G0C1G1C2G2C3G3
                "ZIP2	v17.8h, v2.8h, v6.8h\n" // q17=C4G4C5G5C6G6C7G7
                "LDR	q1, [%[inptr1]], #16\n" // q1=B0B1B2B3B4B5B6B7
                "LDR	q5, [%[inptr5]], #16\n"
                "LDR	q3, [%[inptr3]], #16\n" // q3=D0D1D2D3....
                "LDR	q7, [%[inptr7]], #16\n"
                "ZIP1	v10.8h, v1.8h, v5.8h\n" // q18=B0F0B1F1B2F2B3F3
                "ZIP2	v18.8h, v1.8h, v5.8h\n" // q18=B4F4B5F5B6F6B7F7
                "ZIP1	v11.8h, v3.8h, v7.8h\n" // q19=D0H0D1H1D2H2D3H3
                "ZIP2	v19.8h, v3.8h, v7.8h\n" // q19=D4H4D5H5D6H6D7H7

                "ZIP1	v12.8h,  v8.8h,  v9.8h\n" // q20=A0C0E0G0A1C1E1G1
                "ZIP2	v20.8h,  v8.8h,  v9.8h\n"
                "ZIP1	v13.8h, v10.8h, v11.8h\n" // q21=B0D0F0H0B1I1F1H1
                "ZIP2	v21.8h, v10.8h, v11.8h\n"

                "CBNZ	%w[skippf], 2f\n"
                ASM_PREFETCH("[%[inptr4], #112]")
                ASM_PREFETCH("[%[inptr5], #112]")
                ASM_PREFETCH("[%[inptr6], #112]")
                ASM_PREFETCH("[%[inptr7], #112]")
                "2:\n"

                "ZIP1	v22.8h, v16.8h, v17.8h\n"
                "ZIP2	v30.8h, v16.8h, v17.8h\n"
                "ZIP1	v23.8h, v18.8h, v19.8h\n"
                "ZIP2	v31.8h, v18.8h, v19.8h\n"

                "ZIP1	v14.8h, v12.8h, v13.8h\n" // q22=A0B0C0D0E0F0G0H0
                "ZIP2	v15.8h, v12.8h, v13.8h\n" // q23=A1B1C1D1E1F1G1H1
                "STP	q14, q15, [%[outptr]], #32\n" // Write back first two elements

                "ZIP1	v0.8h, v20.8h, v21.8h\n"
                "ZIP2	v1.8h, v20.8h, v21.8h\n"
                "STP	q0, q1, [%[outptr]], #32\n" // Write back next two elements

                "ZIP1	v2.8h, v22.8h, v23.8h\n"
                "ZIP2	v3.8h, v22.8h, v23.8h\n"
                "STP	q2, q3, [%[outptr]], #32\n" // Write back next two elements

                "ZIP1	v4.8h, v30.8h, v31.8h\n"
                "ZIP2	v5.8h, v30.8h, v31.8h\n"
                "STP	q4, q5, [%[outptr]], #32\n" // Write back last two elements
                : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
                  [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7), [outptr] "+r" (outptr)
                : [skippf] "r" (skippf)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                  "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory"
            );
        }

        for (;x>0;x--) {
            *outptr++ = *inptr0++;
            *outptr++ = *inptr1++;
            *outptr++ = *inptr2++;
            *outptr++ = *inptr3++;
            *outptr++ = *inptr4++;
            *outptr++ = *inptr5++;
            *outptr++ = *inptr6++;
            *outptr++ = *inptr7++;
        }
    }
}

#endif // __aarch64__
