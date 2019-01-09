/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)

#include <arm_neon.h>

#include "../asmlib.hpp"

template<>
template<typename T>
inline void TransformImpl<8, 1, false, 4, 4, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint32_t *outptr = (uint32_t *)out;
    const uint32_t *inptr = (uint32_t *)in;

    uint32_t zerobuff[16]; // 8 for asm loop plus up to 7 for overflow loop

    for (int y=y0; y<ymax; y+=8) {
        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;
        const uint32_t *inptr6 = inptr5 + ldin;
        const uint32_t *inptr7 = inptr6 + ldin;

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
                    case 5:
                        inptr2 = zerobuff;
                    case 4:
                        inptr3 = zerobuff;
                    case 3:
                        inptr4 = zerobuff;
                    case 2:
                        inptr5 = zerobuff;
                    case 1:
                        inptr6 = zerobuff;
                    case 0:
                        inptr7 = zerobuff;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            __asm __volatile (
                // Load up 8 elements (2 vectors) from each of 8 sources.
                "LDP        q0, q1, [%[inptr0]], #32\n" // q0=A0A1A2A3
                "LDP        q2, q3, [%[inptr1]], #32\n" // q2=B0B1B2B3
                "LDP        q4, q5, [%[inptr2]], #32\n" // q4=C0C1C2C3
                "ZIP1       v16.4s, v0.4s, v4.4s\n" // q16=A0C0A1C1
                ASM_PREFETCH("[%[inptr0], #128]")
                "LDP        q6, q7, [%[inptr3]], #32\n" // q6=D0D1D2D3
                "ZIP1       v17.4s, v2.4s, v6.4s\n" // q17=B0D0B1D1
                "LDP        q8, q9, [%[inptr4]], #32\n"
                "LDP        q10, q11, [%[inptr5]], #32\n"
                "LDP        q12, q13, [%[inptr6]], #32\n"
                "ZIP1       v18.4s, v8.4s, v12.4s\n"
                ASM_PREFETCH("[%[inptr1], #128]")
                "LDP        q14, q15, [%[inptr7]], #32\n"
                "ZIP1       v19.4s, v10.4s, v14.4s\n"

                "ZIP1       v20.4s, v16.4s, v17.4s\n" // q20=A0B0C0D0
                ASM_PREFETCH("[%[inptr2], #128]")
                "ZIP1       v21.4s, v18.4s, v19.4s\n"
                "ZIP2       v22.4s, v16.4s, v17.4s\n"
                "ZIP2       v23.4s, v18.4s, v19.4s\n"

                "ZIP2       v16.4s, v0.4s, v4.4s\n"
                ASM_PREFETCH("[%[inptr3], #128]")
                "ZIP2       v17.4s, v2.4s, v6.4s\n"
                "STP        q20, q21, [%[outptr]], #32\n" // Write back the first element of each source

                "ZIP2       v18.4s, v8.4s, v12.4s\n"
                "ZIP2       v19.4s, v10.4s, v14.4s\n"
                "STP        q22, q23, [%[outptr]], #32\n" // Write back the second element of each source

                "ZIP1       v20.4s, v16.4s, v17.4s\n"
                ASM_PREFETCH("[%[inptr4], #128]")
                "ZIP1       v21.4s, v18.4s, v19.4s\n"
                "ZIP2       v22.4s, v16.4s, v17.4s\n"
                "ZIP2       v23.4s, v18.4s, v19.4s\n"

                "ZIP1       v16.4s, v1.4s, v5.4s\n"
                ASM_PREFETCH("[%[inptr5], #128]")
                "ZIP1       v17.4s, v3.4s, v7.4s\n"
                "STP        q20, q21, [%[outptr]], #32\n" // Third element

                "ZIP1       v18.4s, v9.4s, v13.4s\n"
                "ZIP1       v19.4s, v11.4s, v15.4s\n"
                "STP        q22, q23, [%[outptr]], #32\n" // Fourth element

                "ZIP1       v20.4s, v16.4s, v17.4s\n"
                "ZIP1       v21.4s, v18.4s, v19.4s\n"
                "ZIP2       v22.4s, v16.4s, v17.4s\n"
                ASM_PREFETCH("[%[inptr6], #128]")
                "ZIP2       v23.4s, v18.4s, v19.4s\n"

                "ZIP2       v16.4s, v1.4s, v5.4s\n"
                "ZIP2       v17.4s, v3.4s, v7.4s\n"
                "STP        q20, q21, [%[outptr]], #32\n" // Fifth element

                "ZIP2       v18.4s, v9.4s, v13.4s\n"
                ASM_PREFETCH("[%[inptr7], #128]")
                "ZIP2       v19.4s, v11.4s, v15.4s\n"
                "STP        q22, q23, [%[outptr]], #32\n" // Sixth element

                "ZIP1       v20.4s, v16.4s, v17.4s\n"
                "ZIP1       v21.4s, v18.4s, v19.4s\n"
                "STP        q20, q21, [%[outptr]], #32\n" // Seventh element

                "ZIP2       v22.4s, v16.4s, v17.4s\n"
                "ZIP2       v23.4s, v18.4s, v19.4s\n"
                "STP        q22, q23, [%[outptr]], #32\n" // Eighth element
                : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
                  [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7), [outptr] "+r" (outptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                  "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory"
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

#endif  // __aarch64__ && !__ARM_FEATURE_SVE
