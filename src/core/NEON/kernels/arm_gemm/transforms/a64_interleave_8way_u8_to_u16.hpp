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
#pragma once

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS)

#include <arm_neon.h>
#include <cstdint>

#include "../asmlib.hpp"

template<>
template<>
inline void TransformImpl<8, 1, false, 2, 1, false>::Transform(uint16_t *out, const uint8_t *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint16_t *outptr = out;
    const uint8_t *inptr = in;
    bool first = true;

    uint8_t zerobuff[32] = { 0 }; // 16 for asm loop plus up to 15 for overflow loop

    for (int y=y0; y<ymax; y+=8) {
        const uint8_t *inptr0 = inptr + y * ldin + k0;
        const uint8_t *inptr1 = inptr0 + ldin;
        const uint8_t *inptr2 = inptr1 + ldin;
        const uint8_t *inptr3 = inptr2 + ldin;
        const uint8_t *inptr4 = inptr3 + ldin;
        const uint8_t *inptr5 = inptr4 + ldin;
        const uint8_t *inptr6 = inptr5 + ldin;
        const uint8_t *inptr7 = inptr6 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        prefetch_2x(inptr6);
        prefetch_2x(inptr7);

        int x=(kmax-k0);
        for (;(x>15) || first;x-=16) {
            /* Cope with ragged cases by copying from a buffer of zeroes instead */
            /* 'first' forces this to always run at least once, needed if the total size is <=7. */
            if ((y + 7) >= ymax) {
                switch ((y + 7) - ymax) {
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

            if (first) {
                if (x<=15) {
                    break;
                }

                first = false;
            }

            __asm __volatile (
                // Load up 16 elements (1 source vector, 2 destination vectors) from each of 8 sources.
                "LDR	q0, [%[inptr0]], #16\n"
                "LDR	q2, [%[inptr1]], #16\n"
                "USHLL2 v1.8h, v0.16b, #0\n"
                "USHLL  v0.8h, v0.8b, #0\n"
                "LDR	q4, [%[inptr2]], #16\n" // q4=C0C1C2C3
                "USHLL2 v3.8h, v2.16b, #0\n"
                "USHLL  v2.8h, v2.8b, #0\n"
                "USHLL2 v5.8h, v4.16b, #0\n"
                "USHLL  v4.8h, v4.8b, #0\n"
                "ZIP1	v16.8h, v0.8h, v4.8h\n" // q16=A0C0A1C1
                ASM_PREFETCH("[%[inptr0], #128]")
                "LDR	q6, [%[inptr3]], #16\n" // q6=D0D1D2D3
                "USHLL2 v7.8h, v6.16b, #0\n"
                "USHLL  v6.8h, v6.8b, #0\n"
                "ZIP1	v17.8h, v2.8h, v6.8h\n" // q17=B0D0B1D1
                "LDR	q8, [%[inptr4]], #16\n"
                "LDR	q10, [%[inptr5]], #16\n"
                "USHLL2 v9.8h, v8.16b, #0\n"
                "USHLL  v8.8h, v8.8b, #0\n"
                ASM_PREFETCH("[%[inptr1], #128]")
                "LDR	q12, [%[inptr6]], #16\n"
                "USHLL2 v11.8h, v10.16b, #0\n"
                "USHLL  v10.8h, v10.8b, #0\n"
                "USHLL2 v13.8h, v12.16b, #0\n"
                "USHLL  v12.8h, v12.8b, #0\n"
                "ZIP1	v18.8h, v8.8h, v12.8h\n"
                "LDR	q14, [%[inptr7]], #16\n"
                "USHLL2 v15.8h, v14.16b, #0\n"
                "USHLL  v14.8h, v14.8b, #0\n"
                "ZIP1	v19.8h, v10.8h, v14.8h\n"

                ASM_PREFETCH("[%[inptr2], #128]")
                "ZIP1	v20.8h, v16.8h, v17.8h\n" // q20=A0B0C0D0A1B1C1D1
                "ZIP1	v21.8h, v18.8h, v19.8h\n" // q21=E0F0G0H0E1F1G1H1
                "ZIP2	v22.8h, v16.8h, v17.8h\n" // q22=A2B2C2D2A3B3C3D3
                "ZIP2	v23.8h, v18.8h, v19.8h\n" // q23=E2F2G2H1E3F3G3H3
                ASM_PREFETCH("[%[inptr3], #128]")

                "ZIP2	v16.8h, v0.8h, v4.8h\n"
                "ZIP2	v17.8h, v2.8h, v6.8h\n"
                "TRN1	v24.2d, v20.2d, v21.2d\n"
                "TRN2	v25.2d, v20.2d, v21.2d\n"

                "ZIP2	v18.8h, v8.8h, v12.8h\n"
                ASM_PREFETCH("[%[inptr4], #128]")
                "ZIP2	v19.8h, v10.8h, v14.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Write back the first element of each source
                "TRN1	v24.2d, v22.2d, v23.2d\n"
                "TRN2	v25.2d, v22.2d, v23.2d\n"

                "ZIP1	v20.8h, v16.8h, v17.8h\n"
                "ZIP1	v21.8h, v18.8h, v19.8h\n"
                ASM_PREFETCH("[%[inptr5], #128]")
                "ZIP2	v22.8h, v16.8h, v17.8h\n"
                "ZIP2	v23.8h, v18.8h, v19.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Write back the second element of each source

                "ZIP1	v16.8h, v1.8h, v5.8h\n"
                "ZIP1	v17.8h, v3.8h, v7.8h\n"
                ASM_PREFETCH("[%[inptr6], #128]")
                "TRN1	v24.2d, v20.2d, v21.2d\n"
                "TRN2	v25.2d, v20.2d, v21.2d\n"

                "ZIP1	v18.8h, v9.8h, v13.8h\n"
                "ZIP1	v19.8h, v11.8h, v15.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Third element
                "TRN1	v24.2d, v22.2d, v23.2d\n"
                "TRN2	v25.2d, v22.2d, v23.2d\n"
                ASM_PREFETCH("[%[inptr7], #128]")

                "ZIP1	v20.8h, v16.8h, v17.8h\n"
                "ZIP1	v21.8h, v18.8h, v19.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Fourth element
                "ZIP2	v22.8h, v16.8h, v17.8h\n"
                "ZIP2	v23.8h, v18.8h, v19.8h\n"

                "ZIP2	v16.8h, v1.8h, v5.8h\n"
                "ZIP2	v17.8h, v3.8h, v7.8h\n"
                "TRN1	v24.2d, v20.2d, v21.2d\n"
                "TRN2	v25.2d, v20.2d, v21.2d\n"

                "ZIP2	v18.8h, v9.8h, v13.8h\n"
                "ZIP2	v19.8h, v11.8h, v15.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Fifth element
                "TRN1	v24.2d, v22.2d, v23.2d\n"
                "TRN2	v25.2d, v22.2d, v23.2d\n"

                "ZIP1	v20.8h, v16.8h, v17.8h\n"
                "ZIP1	v21.8h, v18.8h, v19.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Sixth element
                "TRN1	v24.2d, v20.2d, v21.2d\n"
                "TRN2	v25.2d, v20.2d, v21.2d\n"

                "ZIP2	v22.8h, v16.8h, v17.8h\n"
                "ZIP2	v23.8h, v18.8h, v19.8h\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Seventh element
                "TRN1	v24.2d, v22.2d, v23.2d\n"
                "TRN2	v25.2d, v22.2d, v23.2d\n"
                "STP	q24, q25, [%[outptr]], #32\n" // Eighth element
                : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
                  [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7), [outptr] "+r" (outptr)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                  "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "memory"
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
