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
#include "../utils.hpp"

template<>
template<typename T>
void TransformImpl<4, 16, false, 1, 1, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax) {
    uint8_t *outptr = (uint8_t *)out;
    const uint8_t *inptr = (uint8_t *)in;

    uint8_t zerobuff[16] = { 0 };

    for (int y=y0; y<ymax; y+=4) {
        const uint8_t *inptr0 = inptr + y * ldin + k0;
        const uint8_t *inptr1 = inptr0 + ldin;
        const uint8_t *inptr2 = inptr1 + ldin;
        const uint8_t *inptr3 = inptr2 + ldin;

        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x=(kmax-k0);
        for (;x>15;x-=16) {
            /* Cope with ragged cases by copying from a buffer of zeroes instead */
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        // fall through
                    case 1:
                        inptr2 = zerobuff;
                        // fall through
                    case 0:
                        inptr3 = zerobuff;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            __asm __volatile (
                "LDR	q0, [%[inptr0]], #16\n"
                ASM_PREFETCH("[%[inptr0], #176]")
                "LDR	q1, [%[inptr1]], #16\n"
                ASM_PREFETCH("[%[inptr1], #176]")
                "STP	q0, q1, [%[outptr]], #32\n"
                "LDR	q0, [%[inptr2]], #16\n"
                ASM_PREFETCH("[%[inptr2], #176]")
                "LDR	q1, [%[inptr3]], #16\n"
                ASM_PREFETCH("[%[inptr3], #176]")
                "STP	q0, q1, [%[outptr]], #32\n"
                : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
                  [outptr] "+r" (outptr)
                :
                : "v0", "v1"
            );
        }

        if (x>0) {
            /* Need to duplicate this here, in case we didn't run the main loop. */
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        // fall through
                    case 1:
                        inptr2 = zerobuff;
                        // fall through
                    case 0:
                        inptr3 = zerobuff;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            /* We have to write out 16 values, copy as many legal values as there are and pad with 0 */
            auto f = [&outptr, x](const uint8_t *&p) {
                for (int i=0; i<16; i++) {
                    if (i < x) {
                        *outptr++ = *p++;
                    } else {
                        *outptr++ = 0;
                    }
                }
            };

            f(inptr0);
            f(inptr1);
            f(inptr2);
            f(inptr3);
        }
    }
}

#endif  // __aarch64__