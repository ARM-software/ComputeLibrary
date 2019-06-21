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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS)

#include <arm_neon.h>

#include "../asmlib.hpp"

template<>
template<>
inline void TransformImpl<8, 1, false, 4, 2, false>::Transform(float *out, const __fp16 *in, int ldin, int y0, int ymax, int k0, int kmax) {
    float *outptr = out;
    const __fp16 *inptr = in;

    __fp16 zerobuff[16] = { 0 }; // 8 for asm loop plus up to 7 for overflow loop

    for (int y=y0; y<ymax; y+=8) {
        const __fp16 *inptr0 = inptr + y * ldin + k0;
        const __fp16 *inptr1 = inptr0 + ldin;
        const __fp16 *inptr2 = inptr1 + ldin;
        const __fp16 *inptr3 = inptr2 + ldin;
        const __fp16 *inptr4 = inptr3 + ldin;
        const __fp16 *inptr5 = inptr4 + ldin;
        const __fp16 *inptr6 = inptr5 + ldin;
        const __fp16 *inptr7 = inptr6 + ldin;

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

            __asm __volatile (
                // Load up 8 elements (2 vectors) from each of 8 sources.
                "LDR	q0, [%[inptr0]], #16\n"
                "LDR	q2, [%[inptr1]], #16\n"
                "FCVTL2	v1.4s, v0.8h\n"
                "FCVTL	v0.4s, v0.4h\n"
                "LDR	q4, [%[inptr2]], #16\n" // q4=C0C1C2C3
                "FCVTL2	v3.4s, v2.8h\n"
                "FCVTL	v2.4s, v2.4h\n"
                "FCVTL2	v5.4s, v4.8h\n"
                "FCVTL	v4.4s, v4.4h\n"
                "ZIP1	v16.4s, v0.4s, v4.4s\n" // q16=A0C0A1C1
                ASM_PREFETCH("[%[inptr0], #128]")
                "LDR	q6, [%[inptr3]], #16\n" // q6=D0D1D2D3
                "FCVTL2	v7.4s, v6.8h\n"
                "FCVTL	v6.4s, v6.4h\n"
                "ZIP1	v17.4s, v2.4s, v6.4s\n" // q17=B0D0B1D1
                "LDR	q8, [%[inptr4]], #16\n"
                "LDR	q10, [%[inptr5]], #16\n"
                "FCVTL2	v9.4s, v8.8h\n"
                "FCVTL	v8.4s, v8.4h\n"
                ASM_PREFETCH("[%[inptr1], #128]")
                "LDR	q12, [%[inptr6]], #16\n"
                "FCVTL2	v11.4s, v10.8h\n"
                "FCVTL	v10.4s, v10.4h\n"
                "FCVTL2	v13.4s, v12.8h\n"
                "FCVTL	v12.4s, v12.4h\n"
                "ZIP1	v18.4s, v8.4s, v12.4s\n"
                "LDR	q14, [%[inptr7]], #16\n"
                "FCVTL2	v15.4s, v14.8h\n"
                "FCVTL	v14.4s, v14.4h\n"
                "ZIP1	v19.4s, v10.4s, v14.4s\n"

                ASM_PREFETCH("[%[inptr2], #128]")
                "ZIP1	v20.4s, v16.4s, v17.4s\n" // q20=A0B0C0D0
                "ZIP1	v21.4s, v18.4s, v19.4s\n"
                "ZIP2	v22.4s, v16.4s, v17.4s\n"
                "ZIP2	v23.4s, v18.4s, v19.4s\n"
                ASM_PREFETCH("[%[inptr3], #128]")

                "ZIP2	v16.4s, v0.4s, v4.4s\n"
                "ZIP2	v17.4s, v2.4s, v6.4s\n"
                "STP	q20, q21, [%[outptr]], #32\n" // Write back the first element of each source

                "ZIP2	v18.4s, v8.4s, v12.4s\n"
                ASM_PREFETCH("[%[inptr4], #128]")
                "ZIP2	v19.4s, v10.4s, v14.4s\n"
                "STP	q22, q23, [%[outptr]], #32\n" // Write back the second element of each source

                "ZIP1	v20.4s, v16.4s, v17.4s\n"
                "ZIP1	v21.4s, v18.4s, v19.4s\n"
                ASM_PREFETCH("[%[inptr5], #128]")
                "ZIP2	v22.4s, v16.4s, v17.4s\n"
                "ZIP2	v23.4s, v18.4s, v19.4s\n"

                "ZIP1	v16.4s, v1.4s, v5.4s\n"
                "ZIP1	v17.4s, v3.4s, v7.4s\n"
                ASM_PREFETCH("[%[inptr6], #128]")
                "STP	q20, q21, [%[outptr]], #32\n" // Third element

                "ZIP1	v18.4s, v9.4s, v13.4s\n"
                "ZIP1	v19.4s, v11.4s, v15.4s\n"
                "STP	q22, q23, [%[outptr]], #32\n" // Fourth element
                ASM_PREFETCH("[%[inptr7], #128]")

                "ZIP1	v20.4s, v16.4s, v17.4s\n"
                "ZIP1	v21.4s, v18.4s, v19.4s\n"
                "ZIP2	v22.4s, v16.4s, v17.4s\n"
                "ZIP2	v23.4s, v18.4s, v19.4s\n"

                "ZIP2	v16.4s, v1.4s, v5.4s\n"
                "ZIP2	v17.4s, v3.4s, v7.4s\n"
                "STP	q20, q21, [%[outptr]], #32\n" // Fifth element

                "ZIP2	v18.4s, v9.4s, v13.4s\n"
                "ZIP2	v19.4s, v11.4s, v15.4s\n"
                "STP	q22, q23, [%[outptr]], #32\n" // Sixth element

                "ZIP1	v20.4s, v16.4s, v17.4s\n"
                "ZIP1	v21.4s, v18.4s, v19.4s\n"
                "STP	q20, q21, [%[outptr]], #32\n" // Seventh element

                "ZIP2	v22.4s, v16.4s, v17.4s\n"
                "ZIP2	v23.4s, v18.4s, v19.4s\n"
                "STP	q22, q23, [%[outptr]], #32\n" // Eighth element
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

#endif // __aarch64__ && __ARM_FP16_ARGS
