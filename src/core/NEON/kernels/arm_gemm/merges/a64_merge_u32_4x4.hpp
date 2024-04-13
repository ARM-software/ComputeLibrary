/*
 * Copyright (c) 2019-2020 Arm Limited.
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
void MergeResults<4, 4, false>(uint32_t *out, const uint32_t *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const uint32_t *bias, Activation , bool append)
{
    const uint32_t *inptr = in;
    uint32_t nullbias[4];


    if (!append && !bias)
    {
        memset(nullbias, 0, (4 * sizeof(uint32_t)));
    }

    for (int y=y0; y<ymax; y+=4)
    {
        uint32_t *outptr0 = out + (y * ldout) + x0;
        uint32_t *outptr1 = outptr0 + ldout;
        uint32_t *outptr2 = outptr1 + ldout;
        uint32_t *outptr3 = outptr2 + ldout;

        const int height = ymax - y;

        for (int i=x0; i<xmax; i+=4)
        {
            if (append)
            {
                switch(height)
                {
                case 1:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 += inptr[xi];
                                    outptr0++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q10, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x20]\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v10.4s, v10.4s, v2.4s\n"
                                "str q10, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            :
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 += inptr[xi];
                                    outptr0++;
                                    *outptr1 += inptr[xi + 4];
                                    outptr1++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q10, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q3, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x20]\n"
                                "add v10.4s, v10.4s, v2.4s\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v11.4s, v11.4s, v3.4s\n"
                                "str q10, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "str q11, [%[outptr1]]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            :
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 += inptr[xi];
                                    outptr0++;
                                    *outptr1 += inptr[xi + 4];
                                    outptr1++;
                                    *outptr2 += inptr[xi + 8];
                                    outptr2++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q10, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q3, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x20]\n"
                                "add v10.4s, v10.4s, v2.4s\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "ldr q4, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v11.4s, v11.4s, v3.4s\n"
                                "str q10, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "add v12.4s, v12.4s, v4.4s\n"
                                "str q11, [%[outptr1]]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                                "str q12, [%[outptr2]]\n"
                                "add %[outptr2], %[outptr2], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            :
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 4:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 += inptr[xi];
                                    outptr0++;
                                    *outptr1 += inptr[xi + 4];
                                    outptr1++;
                                    *outptr2 += inptr[xi + 8];
                                    outptr2++;
                                    *outptr3 += inptr[xi + 12];
                                    outptr3++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[outptr0]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q10, [%[inptr]]\n"
                                "prfm PLDL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q3, [%[outptr1]]\n"
                                "prfm PLDL1KEEP, [%[outptr1], #0x20]\n"
                                "add v10.4s, v10.4s, v2.4s\n"
                                "ldr q11, [%[inptr], #0x10]\n"
                                "ldr q4, [%[outptr2]]\n"
                                "prfm PLDL1KEEP, [%[outptr2], #0x20]\n"
                                "ldr q12, [%[inptr], #0x20]\n"
                                "prfm PLDL1KEEP, [%[outptr3], #0x20]\n"
                                "add v11.4s, v11.4s, v3.4s\n"
                                "str q10, [%[outptr0]]\n"
                                "ldr q5, [%[outptr3]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "add v12.4s, v12.4s, v4.4s\n"
                                "str q11, [%[outptr1]]\n"
                                "ldr q13, [%[inptr], #0x30]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "str q12, [%[outptr2]]\n"
                                "add %[outptr2], %[outptr2], #0x10\n"
                                "add v13.4s, v13.4s, v5.4s\n"
                                "str q13, [%[outptr3]]\n"
                                "add %[outptr3], %[outptr3], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            :
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;


                }
            }
            else
            {
                const uint32_t *biasptr = bias ? bias + i : nullbias;

                switch(height)
                {
                case 1:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = biasptr[xi] + inptr[xi];
                                    outptr0++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[biasptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q11, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x20]\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v11.4s, v11.4s, v2.4s\n"
                                "str q11, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                case 2:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = biasptr[xi] + inptr[xi];
                                    outptr0++;
                                    *outptr1 = biasptr[xi] + inptr[xi + 4];
                                    outptr1++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[biasptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q11, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x20]\n"
                                "add v11.4s, v11.4s, v2.4s\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v12.4s, v12.4s, v2.4s\n"
                                "str q11, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "str q12, [%[outptr1]]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                case 3:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = biasptr[xi] + inptr[xi];
                                    outptr0++;
                                    *outptr1 = biasptr[xi] + inptr[xi + 4];
                                    outptr1++;
                                    *outptr2 = biasptr[xi] + inptr[xi + 8];
                                    outptr2++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[biasptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q11, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x20]\n"
                                "add v11.4s, v11.4s, v2.4s\n"
                                "ldr q13, [%[inptr], #0x20]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x20]\n"
                                "add v12.4s, v12.4s, v2.4s\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "add v13.4s, v13.4s, v2.4s\n"
                                "str q11, [%[outptr0]]\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "str q12, [%[outptr1]]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                                "str q13, [%[outptr2]]\n"
                                "add %[outptr2], %[outptr2], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
                            );
                        }
                    }
                    break;

                default:
                case 4:
                    {
                        if ((i+3) >= xmax)
                        {
                            for (int xi=0; xi<3; xi++)
                            {
                                if ((i+xi) < xmax)
                                {
                                    *outptr0 = biasptr[xi] + inptr[xi];
                                    outptr0++;
                                    *outptr1 = biasptr[xi] + inptr[xi + 4];
                                    outptr1++;
                                    *outptr2 = biasptr[xi] + inptr[xi + 8];
                                    outptr2++;
                                    *outptr3 = biasptr[xi] + inptr[xi + 12];
                                    outptr3++;
                                }
                            }
                            inptr += 16;
                        } else {
                            /* Optimized routine to copy an entire block */
                            __asm __volatile (
                                "ldr q2, [%[biasptr]]\n"
                                "prfm PLDL1KEEP, [%[inptr], #0x40]\n"
                                "ldr q11, [%[inptr]]\n"
                                "prfm PSTL1KEEP, [%[outptr0], #0x20]\n"
                                "ldr q12, [%[inptr], #0x10]\n"
                                "prfm PSTL1KEEP, [%[outptr1], #0x20]\n"
                                "add v11.4s, v11.4s, v2.4s\n"
                                "ldr q13, [%[inptr], #0x20]\n"
                                "ldr q14, [%[inptr], #0x30]\n"
                                "prfm PSTL1KEEP, [%[outptr2], #0x20]\n"
                                "add v12.4s, v12.4s, v2.4s\n"
                                "str q11, [%[outptr0]]\n"
                                "add v13.4s, v13.4s, v2.4s\n"
                                "add %[outptr0], %[outptr0], #0x10\n"
                                "add v14.4s, v14.4s, v2.4s\n"
                                "str q12, [%[outptr1]]\n"
                                "add %[outptr1], %[outptr1], #0x10\n"
                                "prfm PSTL1KEEP, [%[outptr3], #0x20]\n"
                                "add %[inptr], %[inptr], #0x40\n"
                                "str q13, [%[outptr2]]\n"
                                "add %[outptr2], %[outptr2], #0x10\n"
                                "str q14, [%[outptr3]]\n"
                                "add %[outptr3], %[outptr3], #0x10\n"
                            : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                              [inptr] "+r" (inptr)
                            : [biasptr] "r" (biasptr)
                            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "memory"
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
