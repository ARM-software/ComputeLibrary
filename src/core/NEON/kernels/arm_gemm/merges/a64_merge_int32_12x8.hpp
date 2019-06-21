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
inline void MergeResults<12, 8, false>(int32_t *out, const int32_t *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const int32_t alpha, const int32_t beta) {
    const int32_t *inptr = in;
    prefetch_6x(inptr);
    prefetch_6x(inptr + 96);

    int32x4_t alpha_value = vdupq_n_s32(alpha);
    int32x4_t beta_value = vdupq_n_s32(beta);

    for (int y=y0; y<ymax; y+=8) {
        int32_t *outptr0 = out + (y * ldout) + x0;
        int32_t *outptr1 = outptr0 + ldout;
        int32_t *outptr2 = outptr1 + ldout;
        int32_t *outptr3 = outptr2 + ldout;
        int32_t *outptr4 = outptr3 + ldout;
        int32_t *outptr5 = outptr4 + ldout;
        int32_t *outptr6 = outptr5 + ldout;
        int32_t *outptr7 = outptr6 + ldout;

        prefetch_2x(outptr0);
        prefetch_2x(outptr1);
        prefetch_2x(outptr2);
        prefetch_2x(outptr3);
        prefetch_2x(outptr4);
        prefetch_2x(outptr5);
        prefetch_2x(outptr6);
        prefetch_2x(outptr7);

        for (int i=x0; i<xmax; i+=12) {
            int32_t dummyres[12];

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
                  // Row 0
                  ASM_PREFETCH("[%x[outptr1], #192]")
                  "ldr q3, [%x[outptr0]]\n"
                  "ldr q4, [%x[outptr0], #0x10]\n"
                  "ldr q5, [%x[outptr0], #0x20]\n"
                  "mul v3.4s, v3.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr]]\n"
                  "mul v4.4s, v4.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x10]\n"
                  "mul v5.4s, v5.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x20]\n"
                  "mla v3.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q0, [%x[outptr1]]\n"
                  "mla v4.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q1, [%x[outptr1], #0x10]\n"
                  "mla v5.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q2, [%x[outptr1], #0x20]\n"

                  // Row 1
                  ASM_PREFETCH("[%x[outptr2], #192]")
                  "mul v0.4s, v0.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0x30]\n"
                  "str q3, [%x[outptr0]], #0x10\n"
                  "mul v1.4s, v1.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x40]\n"
                  "str q4, [%x[outptr0]], #0x10\n"
                  "mul v2.4s, v2.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x50]\n"
                  "str q5, [%x[outptr0]], #0x10\n"
                  "mla v0.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q3, [%x[outptr2]]\n"
                  "mla v1.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q4, [%x[outptr2], #0x10]\n"
                  "mla v2.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q5, [%x[outptr2], #0x20]\n"

                  // Row 2
                  ASM_PREFETCH("[%x[outptr3], #192]")
                  "mul v3.4s, v3.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0x60]\n"
                  "str q0, [%x[outptr1]], #0x10\n"
                  "mul v4.4s, v4.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x70]\n"
                  "str q1, [%x[outptr1]], #0x10\n"
                  "mul v5.4s, v5.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x80]\n"
                  "str q2, [%x[outptr1]], #0x10\n"
                  "mla v3.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q0, [%x[outptr3]]\n"
                  "mla v4.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q1, [%x[outptr3], #0x10]\n"
                  "mla v5.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q2, [%x[outptr3], #0x20]\n"

                  // Row 3
                  ASM_PREFETCH("[%x[outptr4], #192]")
                  "mul v0.4s, v0.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0x90]\n"
                  "str q3, [%x[outptr2]], #0x10\n"
                  "mul v1.4s, v1.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0xa0]\n"
                  "str q4, [%x[outptr2]], #0x10\n"
                  "mul v2.4s, v2.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0xb0]\n"
                  "str q5, [%x[outptr2]], #0x10\n"
                  "mla v0.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q3, [%x[outptr4]]\n"
                  "mla v1.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q4, [%x[outptr4], #0x10]\n"
                  "mla v2.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q5, [%x[outptr4], #0x20]\n"

                  // Row 4
                  ASM_PREFETCH("[%x[outptr5], #192]")
                  "mul v3.4s, v3.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0xc0]\n"
                  "str q0, [%x[outptr3]], #0x10\n"
                  "mul v4.4s, v4.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0xd0]\n"
                  "str q1, [%x[outptr3]], #0x10\n"
                  "mul v5.4s, v5.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0xe0]\n"
                  "str q2, [%x[outptr3]], #0x10\n"
                  "mla v3.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q0, [%x[outptr5]]\n"
                  "mla v4.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q1, [%x[outptr5], #0x10]\n"
                  "mla v5.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q2, [%x[outptr5], #0x20]\n"

                  // Row 5
                  ASM_PREFETCH("[%x[outptr6], #192]")
                  "mul v0.4s, v0.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0xf0]\n"
                  "str q3, [%x[outptr4]], #0x10\n"
                  "mul v1.4s, v1.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x100]\n"
                  "str q4, [%x[outptr4]], #0x10\n"
                  "mul v2.4s, v2.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x110]\n"
                  "str q5, [%x[outptr4]], #0x10\n"
                  "mla v0.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q3, [%x[outptr6]]\n"
                  "mla v1.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q4, [%x[outptr6], #0x10]\n"
                  "mla v2.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q5, [%x[outptr6], #0x20]\n"

                  // Row 6
                  ASM_PREFETCH("[%x[outptr7], #192]")
                  "mul v3.4s, v3.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0x120]\n"
                  "str q0, [%x[outptr5]], #0x10\n"
                  "mul v4.4s, v4.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x130]\n"
                  "str q1, [%x[outptr5]], #0x10\n"
                  "mul v5.4s, v5.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x140]\n"
                  "str q2, [%x[outptr5]], #0x10\n"
                  "mla v3.4s, v6.4s, %[alpha_value].4s\n"
                  "ldr q0, [%x[outptr7]]\n"
                  "mla v4.4s, v7.4s, %[alpha_value].4s\n"
                  "ldr q1, [%x[outptr7], #0x10]\n"
                  "mla v5.4s, v8.4s, %[alpha_value].4s\n"
                  "ldr q2, [%x[outptr7], #0x20]\n"

                  // Row 7
                  "mul v0.4s, v0.4s, %[beta_value].4s\n"
                  "ldr q6, [%x[inptr], #0x150]\n"
                  "str q3, [%x[outptr6]], #0x10\n"
                  "mul v1.4s, v1.4s, %[beta_value].4s\n"
                  "ldr q7, [%x[inptr], #0x160]\n"
                  "str q4, [%x[outptr6]], #0x10\n"
                  "mul v2.4s, v2.4s, %[beta_value].4s\n"
                  "ldr q8, [%x[inptr], #0x170]\n"
                  "str q5, [%x[outptr6]], #0x10\n"
                  "mla v0.4s, v6.4s, %[alpha_value].4s\n"
                  "mla v1.4s, v7.4s, %[alpha_value].4s\n"
                  "mla v2.4s, v8.4s, %[alpha_value].4s\n"
                  "str q0, [%x[outptr7]], #0x10\n"
                  "str q1, [%x[outptr7]], #0x10\n"
                  "str q2, [%x[outptr7]], #0x10\n"

                  "add %x[inptr], %x[inptr], #0x180\n"
                  : [outptr0] "+r" (outptr0),
                    [outptr1] "+r" (outptr1),
                    [outptr2] "+r" (outptr2),
                    [outptr3] "+r" (outptr3),
                    [outptr4] "+r" (outptr4),
                    [outptr5] "+r" (outptr5),
                    [outptr6] "+r" (outptr6),
                    [outptr7] "+r" (outptr7),
                    [inptr] "+r" (inptr)
                  : [alpha_value] "w" (alpha_value),
                    [beta_value] "w" (beta_value)
                  : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"
              );
            }
        }
    }
}

template<>
inline void MergeResults<12, 8>(uint32_t *out, const uint32_t *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const uint32_t alpha, const uint32_t beta) {
  // Since the above code uses only MUL and MLA instructions discard the "unsignedness" and proceed safely.
  MergeResults<12, 8>(reinterpret_cast<int32_t*>(out), reinterpret_cast<const int32_t*>(in), ldout, y0, ymax, x0, xmax, static_cast<int32_t>(alpha), static_cast<int32_t>(beta));
}

#endif // __aarch64__
