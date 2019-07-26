/*
 * Copyright (c) 2018-2019 Arm Limited.
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

#ifdef __ARM_FEATURE_SVE

template<>
template<typename T>
inline void TransformImpl<8, 1, false, 4, 4, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax)
{
    uint32_t *master_outptr = reinterpret_cast<uint32_t *>(out);
    const uint32_t *inptr = reinterpret_cast<const uint32_t *>(in);

    for (int y=y0; y<ymax; y+=8)
    {
        const int height = ymax-y;
        const long inwidth = (kmax - k0);
        const long outwidth = inwidth * 8;
        long inpos = 0;
        long outpos = 0;

        uint32_t *outptr = master_outptr;
        master_outptr += outwidth;

        const uint32_t *inptr0 = inptr + y * ldin + k0;
        const uint32_t *inptr1 = inptr0 + ldin;
        const uint32_t *inptr2 = inptr1 + ldin;
        const uint32_t *inptr3 = inptr2 + ldin;
        const uint32_t *inptr4 = inptr3 + ldin;
        const uint32_t *inptr5 = inptr4 + ldin;
        const uint32_t *inptr6 = inptr5 + ldin;
        const uint32_t *inptr7 = inptr6 + ldin;

        switch(height)
        {
            case 1:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip1 z0.s, z8.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z1.s, z8.s, z4.s\n"
                    "zip1 z2.s, z9.s, z4.s\n"
                    "zip2 z3.s, z9.s, z4.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z4.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.s, z3.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 2:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.s, #0\n"
                    "mov z14.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip1 z0.s, z8.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z1.s, z8.s, z4.s\n"
                    "zip1 z2.s, z9.s, z4.s\n"
                    "zip2 z3.s, z9.s, z4.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "zip1 z6.s, z11.s, z14.s\n"
                    "zip2 z7.s, z11.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 3:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.s, #0\n"
                    "mov z14.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "zip1 z6.s, z11.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z7.s, z11.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 4:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "ld1w z3.s, p0/z, [%[inptr3], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z4.s\n"
                    "zip2 z15.s, z3.s, z4.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 5:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z5.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "ld1w z3.s, p0/z, [%[inptr3], %[inpos], LSL #2]\n"
                    "ld1w z4.s, p0/z, [%[inptr4], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z5.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z5.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z5.s\n"
                    "zip2 z15.s, z3.s, z5.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 6:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z6.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "ld1w z3.s, p0/z, [%[inptr3], %[inpos], LSL #2]\n"
                    "ld1w z4.s, p0/z, [%[inptr4], %[inpos], LSL #2]\n"
                    "ld1w z5.s, p0/z, [%[inptr5], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z6.s\n"
                    "zip2 z15.s, z3.s, z6.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            case 7:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z7.s, #0\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "ld1w z3.s, p0/z, [%[inptr3], %[inpos], LSL #2]\n"
                    "ld1w z4.s, p0/z, [%[inptr4], %[inpos], LSL #2]\n"
                    "ld1w z5.s, p0/z, [%[inptr5], %[inpos], LSL #2]\n"
                    "ld1w z6.s, p0/z, [%[inptr6], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [inptr6] "+r" (inptr6)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;

            default:
            case 8:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.s, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "ld1w z0.s, p0/z, [%[inptr0], %[inpos], LSL #2]\n"
                    "ld1w z1.s, p0/z, [%[inptr1], %[inpos], LSL #2]\n"
                    "ld1w z2.s, p0/z, [%[inptr2], %[inpos], LSL #2]\n"
                    "ld1w z3.s, p0/z, [%[inptr3], %[inpos], LSL #2]\n"
                    "ld1w z4.s, p0/z, [%[inptr4], %[inpos], LSL #2]\n"
                    "ld1w z5.s, p0/z, [%[inptr5], %[inpos], LSL #2]\n"
                    "ld1w z6.s, p0/z, [%[inptr6], %[inpos], LSL #2]\n"
                    "ld1w z7.s, p0/z, [%[inptr7], %[inpos], LSL #2]\n"
                    "incw %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p1.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.s, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.s, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.s, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1w z8.s, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.s, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1w z9.s, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incw %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1w z10.s, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.s, %[outpos], %[outwidth]\n"
                    "st1w z11.s, p3, [%[outptr], #3, MUL VL]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z12.s, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.s, %[outpos], %[outwidth]\n"
                    "incw %[outpos], all, mul #1\n"
                    "st1w z13.s, p5, [%[outptr], #5, MUL VL]\n"
                    "st1w z14.s, p6, [%[outptr], #6, MUL VL]\n"
                    "st1w z15.s, p7, [%[outptr], #7, MUL VL]\n"
                    "addvl %[outptr], %[outptr], #8\n"
                    "b 1b\n"
                    "2:\n"
                : [inpos] "+r" (inpos), [outpos] "+r" (outpos), [outptr] "+r" (outptr), [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3), [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [inptr6] "+r" (inptr6), [inptr7] "+r" (inptr7)
                : [outwidth] "r" (outwidth), [inwidth] "r" (inwidth)
                : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "cc", "memory"
                );
                break;


        }
    }
}

#endif // __ARM_FEATURE_SVE
