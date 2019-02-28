/*
 * Copyright (c) 2019 Arm Limited.
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
inline void MergeResults<12, 8, false>(float *out, const float *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const float alpha, const float beta)
{
    const float *inptr = in;

    for (int y=y0; y<ymax; y+=8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;
        float *outptr6 = outptr5 + ldout;
        float *outptr7 = outptr6 + ldout;

        const int height = ymax - y;

        for (int i=x0; i<xmax; i+=12) {
            if (beta==0.0f)
            {
                switch(height) {
                case 1:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr0], #0x10]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "add %[inptr], %[inptr], #0x180\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x10]\n"
                                "ldr q7, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x10]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "ldr q4, [%[inptr], #0x40]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr1], #0x10]\n"
                                "ldr q5, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr2], #0x10]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 4:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]);
                                    outptr3++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q4, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x10]\n"
                                "ldr q5, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x10]\n"
                                "ldr q6, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x10]\n"
                                "ldr q7, [%[inptr], #0xa0]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x10]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 5:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]);
                                    outptr3++;
                                    *outptr4 = (alpha * inptr[xi + 48]);
                                    outptr4++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q5, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr0], #0x10]\n"
                                "ldr q6, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr1], #0x10]\n"
                                "ldr q7, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr2], #0x10]\n"
                                "ldr q4, [%[inptr], #0xa0]\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr3], #0x10]\n"
                                "ldr q5, [%[inptr], #0xd0]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr4], #0x10]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "ldr q5, [%[inptr], #0xb0]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x20]\n"
                                "ldr q6, [%[inptr], #0xe0]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 6:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]);
                                    outptr3++;
                                    *outptr4 = (alpha * inptr[xi + 48]);
                                    outptr4++;
                                    *outptr5 = (alpha * inptr[xi + 60]);
                                    outptr5++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q6, [%[inptr], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x10]\n"
                                "ldr q7, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x10]\n"
                                "ldr q4, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x10]\n"
                                "ldr q5, [%[inptr], #0xa0]\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x10]\n"
                                "ldr q6, [%[inptr], #0xd0]\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x10]\n"
                                "ldr q7, [%[inptr], #0x100]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr5], #0x10]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "ldr q4, [%[inptr], #0xe0]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x20]\n"
                                "ldr q5, [%[inptr], #0x110]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 7:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]);
                                    outptr3++;
                                    *outptr4 = (alpha * inptr[xi + 48]);
                                    outptr4++;
                                    *outptr5 = (alpha * inptr[xi + 60]);
                                    outptr5++;
                                    *outptr6 = (alpha * inptr[xi + 72]);
                                    outptr6++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q6, [%[inptr], #0x120]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6]]\n"
                                "ldr q7, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "ldr q4, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr1], #0x10]\n"
                                "ldr q5, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr2], #0x10]\n"
                                "ldr q6, [%[inptr], #0xa0]\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr3], #0x10]\n"
                                "ldr q7, [%[inptr], #0xd0]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr4], #0x10]\n"
                                "ldr q4, [%[inptr], #0x100]\n"
                                "prfm PSTL1KEEP, [%[outptr6], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr5], #0x10]\n"
                                "ldr q5, [%[inptr], #0x130]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr6], #0x10]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "ldr q5, [%[inptr], #0xb0]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x20]\n"
                                "ldr q6, [%[inptr], #0xe0]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x20]\n"
                                "ldr q7, [%[inptr], #0x110]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr5], #0x20]\n"
                                "ldr q4, [%[inptr], #0x140]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr6], #0x20]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 8:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]);
                                    outptr3++;
                                    *outptr4 = (alpha * inptr[xi + 48]);
                                    outptr4++;
                                    *outptr5 = (alpha * inptr[xi + 60]);
                                    outptr5++;
                                    *outptr6 = (alpha * inptr[xi + 72]);
                                    outptr6++;
                                    *outptr7 = (alpha * inptr[xi + 84]);
                                    outptr7++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q4, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q6, [%[inptr], #0x120]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6]]\n"
                                "ldr q7, [%[inptr], #0x150]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7]]\n"
                                "ldr q4, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x10]\n"
                                "ldr q5, [%[inptr], #0x40]\n"
                                "prfm PSTL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x10]\n"
                                "ldr q6, [%[inptr], #0x70]\n"
                                "prfm PSTL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x10]\n"
                                "ldr q7, [%[inptr], #0xa0]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x10]\n"
                                "ldr q4, [%[inptr], #0xd0]\n"
                                "prfm PSTL1KEEP, [%[outptr6], #0x60]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x10]\n"
                                "ldr q5, [%[inptr], #0x100]\n"
                                "prfm PSTL1KEEP, [%[outptr7], #0x60]\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x10]\n"
                                "ldr q6, [%[inptr], #0x130]\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6], #0x10]\n"
                                "ldr q7, [%[inptr], #0x160]\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7], #0x10]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "ldr q4, [%[inptr], #0xe0]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x20]\n"
                                "ldr q5, [%[inptr], #0x110]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x20]\n"
                                "ldr q6, [%[inptr], #0x140]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fmul v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6], #0x20]\n"
                                "ldr q7, [%[inptr], #0x170]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "fmul v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7], #0x20]\n"
                                "add %[outptr7], %[outptr7], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;


                }
            }
            else
            {
                switch(height) {
                case 1:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                                    outptr0++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x10]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr0], #0x10]\n"
                                "ldr q10, [%[outptr0], #0x20]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]) + (*outptr1 * beta);
                                    outptr1++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x10]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x10]\n"
                                "ldr q11, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x40]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x10]\n"
                                "ldr q8, [%[outptr0], #0x20]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q9, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]) + (*outptr1 * beta);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]) + (*outptr2 * beta);
                                    outptr2++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x10]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "ldr q8, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x40]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr1], #0x10]\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x70]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr2], #0x10]\n"
                                "ldr q10, [%[outptr0], #0x20]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q11, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q8, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 4:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = (alpha * inptr[xi]) + (*outptr0 * beta);
                                    outptr0++;
                                    *outptr1 = (alpha * inptr[xi + 12]) + (*outptr1 * beta);
                                    outptr1++;
                                    *outptr2 = (alpha * inptr[xi + 24]) + (*outptr2 * beta);
                                    outptr2++;
                                    *outptr3 = (alpha * inptr[xi + 36]) + (*outptr3 * beta);
                                    outptr3++;
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr3]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q8, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x10]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x10]\n"
                                "ldr q9, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x40]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x10]\n"
                                "ldr q10, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x70]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x10]\n"
                                "ldr q11, [%[outptr3], #0x10]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xa0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x10]\n"
                                "ldr q8, [%[outptr0], #0x20]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q9, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q10, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q11, [%[outptr3], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 5:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
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
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr3]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q8, [%[outptr4]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q9, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x10]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr0], #0x10]\n"
                                "ldr q10, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x40]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr1], #0x10]\n"
                                "ldr q11, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x70]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr2], #0x10]\n"
                                "ldr q8, [%[outptr3], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xa0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr3], #0x10]\n"
                                "ldr q9, [%[outptr4], #0x10]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xd0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr4], #0x10]\n"
                                "ldr q10, [%[outptr0], #0x20]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q11, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q8, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "ldr q9, [%[outptr3], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xb0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x20]\n"
                                "ldr q10, [%[outptr4], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0xe0]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 6:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
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
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr3]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q8, [%[outptr4]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q9, [%[outptr5]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q10, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x10]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x10]\n"
                                "ldr q11, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x40]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x10]\n"
                                "ldr q8, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x70]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x10]\n"
                                "ldr q9, [%[outptr3], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xa0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x10]\n"
                                "ldr q10, [%[outptr4], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0xd0]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x10]\n"
                                "ldr q11, [%[outptr5], #0x10]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x100]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr5], #0x10]\n"
                                "ldr q8, [%[outptr0], #0x20]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q9, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q10, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q11, [%[outptr3], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xe0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x20]\n"
                                "ldr q9, [%[outptr5], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x110]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                case 7:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
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
                                }
                            }
                            inptr += 96;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr3]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q8, [%[outptr4]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q9, [%[outptr5]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q10, [%[outptr6]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x120]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6]]\n"
                                "ldr q11, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x10]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr0], #0x10]\n"
                                "ldr q8, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x40]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr1], #0x10]\n"
                                "ldr q9, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x70]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr2], #0x10]\n"
                                "ldr q10, [%[outptr3], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0xa0]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr3], #0x10]\n"
                                "ldr q11, [%[outptr4], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xd0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr4], #0x10]\n"
                                "ldr q8, [%[outptr5], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr6], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x100]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr5], #0x10]\n"
                                "ldr q9, [%[outptr6], #0x10]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x130]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr6], #0x10]\n"
                                "ldr q10, [%[outptr0], #0x20]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x20]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr0], #0x20]\n"
                                "ldr q11, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x50]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr1], #0x20]\n"
                                "ldr q8, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x80]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr2], #0x20]\n"
                                "ldr q9, [%[outptr3], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xb0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr3], #0x20]\n"
                                "ldr q10, [%[outptr4], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0xe0]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr4], #0x20]\n"
                                "ldr q11, [%[outptr5], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x110]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr5], #0x20]\n"
                                "ldr q8, [%[outptr6], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x140]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr6], #0x20]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 8:
                    {
                        if ((i+11) >= xmax)
                        {
                            for (int xi=0; xi<12; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
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
                                "ldr q8, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x180]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr]]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0]]\n"
                                "ldr q9, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x240]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x30]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1]]\n"
                                "ldr q10, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x1c0]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x60]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2]]\n"
                                "ldr q11, [%[outptr3]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x280]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x90]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3]]\n"
                                "ldr q8, [%[outptr4]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xc0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4]]\n"
                                "ldr q9, [%[outptr5]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0xf0]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5]]\n"
                                "ldr q10, [%[outptr6]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x200]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x120]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6]]\n"
                                "ldr q11, [%[outptr7]]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x60]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x150]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7]]\n"
                                "ldr q8, [%[outptr0], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x10]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x10]\n"
                                "ldr q9, [%[outptr1], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr4], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x40]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x10]\n"
                                "ldr q10, [%[outptr2], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr5], #0x60]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x70]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x10]\n"
                                "ldr q11, [%[outptr3], #0x10]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x2c0]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xa0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x10]\n"
                                "ldr q8, [%[outptr4], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr6], #0x60]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xd0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x10]\n"
                                "ldr q9, [%[outptr5], #0x10]\n"
                                "prfm PLDL1KEEP, [%[outptr7], #0x60]\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x100]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x10]\n"
                                "ldr q10, [%[outptr6], #0x10]\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x130]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6], #0x10]\n"
                                "ldr q11, [%[outptr7], #0x10]\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x160]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7], #0x10]\n"
                                "ldr q8, [%[outptr0], #0x20]\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0x20]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr0], #0x20]\n"
                                "ldr q9, [%[outptr1], #0x20]\n"
                                "add %[outptr0], %[outptr0], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x50]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr1], #0x20]\n"
                                "ldr q10, [%[outptr2], #0x20]\n"
                                "add %[outptr1], %[outptr1], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x80]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr2], #0x20]\n"
                                "ldr q11, [%[outptr3], #0x20]\n"
                                "add %[outptr2], %[outptr2], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0xb0]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr3], #0x20]\n"
                                "ldr q8, [%[outptr4], #0x20]\n"
                                "add %[outptr3], %[outptr3], #0x30\n"
                                "fmul v8.4s, v8.4s, %[beta].s[0]\n"
                                "ldr q4, [%[inptr], #0xe0]\n"
                                "fmla v8.4s, v4.4s, %[alpha].s[0]\n"
                                "str q8, [%[outptr4], #0x20]\n"
                                "ldr q9, [%[outptr5], #0x20]\n"
                                "add %[outptr4], %[outptr4], #0x30\n"
                                "fmul v9.4s, v9.4s, %[beta].s[0]\n"
                                "ldr q5, [%[inptr], #0x110]\n"
                                "fmla v9.4s, v5.4s, %[alpha].s[0]\n"
                                "str q9, [%[outptr5], #0x20]\n"
                                "ldr q10, [%[outptr6], #0x20]\n"
                                "add %[outptr5], %[outptr5], #0x30\n"
                                "fmul v10.4s, v10.4s, %[beta].s[0]\n"
                                "ldr q6, [%[inptr], #0x140]\n"
                                "fmla v10.4s, v6.4s, %[alpha].s[0]\n"
                                "str q10, [%[outptr6], #0x20]\n"
                                "ldr q11, [%[outptr7], #0x20]\n"
                                "add %[outptr6], %[outptr6], #0x30\n"
                                "fmul v11.4s, v11.4s, %[beta].s[0]\n"
                                "ldr q7, [%[inptr], #0x170]\n"
                                "fmla v11.4s, v7.4s, %[alpha].s[0]\n"
                                "str q11, [%[outptr7], #0x20]\n"
                                "add %[outptr7], %[outptr7], #0x30\n"
                                "add %[inptr], %[inptr], #0x180\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3), [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [outptr6] "+r" (outptr6), [outptr7] "+r" (outptr7),
                              [inptr] "+r" (inptr)
                            : [alpha] "w" (alpha), [beta] "w" (beta)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory"
                            );
                        }
                    }
                    break;


                }
            }
        }
    }
}

#endif // __aarch64__
