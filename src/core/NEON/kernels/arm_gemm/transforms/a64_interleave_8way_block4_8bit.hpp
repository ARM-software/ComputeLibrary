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

#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)

#include <arm_neon.h>

#include "../asmlib.hpp"

template<>
template<typename T>
inline void TransformImpl<8, 4, false, 1, 1, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint8_t *outptr = reinterpret_cast<uint8_t *>(out);
    const uint8_t *inptr = reinterpret_cast<const uint8_t *>(in);
    bool first = true;

    /* Helper functions to copy blocks about used for odd case. */
    class t {
    public:
        static inline void copy_4_inc(uint8_t *&out, const uint8_t *&in) {
            uint32_t *out_word = reinterpret_cast<uint32_t *>(out);
            const uint32_t *in_word = reinterpret_cast<const uint32_t *>(in);

            *out_word++ = *in_word++;

            out = reinterpret_cast<uint8_t *>(out_word);
            in = reinterpret_cast<const uint8_t *>(in_word);
        }

        static inline void copy_pad(uint8_t *&out, const uint8_t *&in, size_t count) {
            for (unsigned int i=0; i<4; i++) {
                if (i < count) {
                    *out++ = *in++;
                } else {
                    *out++ = 0;
                }
            }
        }
    };

    uint8_t zerobuff[64] = { 0 }; // 32 for asm loop plus up to 31 for overflow loop

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
        for (;(x>31) || first;x-=32) {
            /* Cope with ragged cases by copying from a buffer of zeroes instead */
            /* 'first' forces this to always run at least once, needed if the total size is <=32. */
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
                if (x<=31) {
                    break;
                }

                first = false;
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

        // Copy any leftover blocks of 4 a complete block at a time.
        for (;x>4;x-=4) {
            t::copy_4_inc(outptr, inptr0);
            t::copy_4_inc(outptr, inptr1);
            t::copy_4_inc(outptr, inptr2);
            t::copy_4_inc(outptr, inptr3);
            t::copy_4_inc(outptr, inptr4);
            t::copy_4_inc(outptr, inptr5);
            t::copy_4_inc(outptr, inptr6);
            t::copy_4_inc(outptr, inptr7);
        }

        // Final block with padding, if any.
        if (x > 0) {
            t::copy_pad(outptr, inptr0, x);
            t::copy_pad(outptr, inptr1, x);
            t::copy_pad(outptr, inptr2, x);
            t::copy_pad(outptr, inptr3, x);
            t::copy_pad(outptr, inptr4, x);
            t::copy_pad(outptr, inptr5, x);
            t::copy_pad(outptr, inptr6, x);
            t::copy_pad(outptr, inptr7, x);
        }
    }
}

#endif  // __aarch64__ && !__ARM_FEATURE_SVE
