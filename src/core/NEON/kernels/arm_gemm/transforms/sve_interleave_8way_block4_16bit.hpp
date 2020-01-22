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

#ifdef __ARM_FEATURE_SVE

template<>
template<typename T>
inline void TransformImpl<8, 4, false, 2, 2, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax)
{
    uint16_t *master_outptr = reinterpret_cast<uint16_t *>(out);
    const uint16_t *inptr = reinterpret_cast<const uint16_t *>(in);

    for (int y=y0; y<ymax; y+=8)
    {
        const int height = ymax-y;
        const long inwidth = (kmax - k0);
        const long outwidth = ((inwidth + 3) / 4) * 32;
        long inpos = 0;
        long outpos = 0;

        uint16_t *outptr = master_outptr;
        master_outptr += outwidth;

        const uint16_t *inptr0 = inptr + y * ldin + k0;
        const uint16_t *inptr1 = inptr0 + ldin;
        const uint16_t *inptr2 = inptr1 + ldin;
        const uint16_t *inptr3 = inptr2 + ldin;
        const uint16_t *inptr4 = inptr3 + ldin;
        const uint16_t *inptr5 = inptr4 + ldin;
        const uint16_t *inptr6 = inptr5 + ldin;
        const uint16_t *inptr7 = inptr6 + ldin;

        switch(height)
        {
            case 1:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip1 z0.d, z8.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z1.d, z8.d, z4.d\n"
                    "zip1 z2.d, z9.d, z4.d\n"
                    "zip2 z3.d, z9.d, z4.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip1 z10.d, z1.d, z4.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z4.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip1 z12.d, z2.d, z4.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.d, z2.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z4.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.d, z3.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.h, #0\n"
                    "mov z14.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip1 z10.d, z1.d, z4.d\n"
                    "zip2 z11.d, z1.d, z4.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip1 z0.d, z8.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z1.d, z8.d, z4.d\n"
                    "zip1 z2.d, z9.d, z4.d\n"
                    "zip2 z3.d, z9.d, z4.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "zip1 z6.d, z11.d, z14.d\n"
                    "zip2 z7.d, z11.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.h, #0\n"
                    "mov z14.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z4.d\n"
                    "zip2 z11.d, z1.d, z4.d\n"
                    "zip1 z12.d, z2.d, z4.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "zip1 z6.d, z11.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z7.d, z11.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "ld1h z3.h, p0/z, [%[inptr3], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z4.d\n"
                    "zip2 z11.d, z1.d, z4.d\n"
                    "zip1 z12.d, z2.d, z4.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z4.d\n"
                    "zip2 z15.d, z3.d, z4.d\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z6.d, z11.d, z15.d\n"
                    "zip2 z7.d, z11.d, z15.d\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z5.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "ld1h z3.h, p0/z, [%[inptr3], %[inpos], LSL #1]\n"
                    "ld1h z4.h, p0/z, [%[inptr4], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z5.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z5.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z5.d\n"
                    "zip2 z15.d, z3.d, z5.d\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z6.d, z11.d, z15.d\n"
                    "zip2 z7.d, z11.d, z15.d\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z6.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "ld1h z3.h, p0/z, [%[inptr3], %[inpos], LSL #1]\n"
                    "ld1h z4.h, p0/z, [%[inptr4], %[inpos], LSL #1]\n"
                    "ld1h z5.h, p0/z, [%[inptr5], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z6.d\n"
                    "zip2 z15.d, z3.d, z6.d\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z6.d, z11.d, z15.d\n"
                    "zip2 z7.d, z11.d, z15.d\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z7.h, #0\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "ld1h z3.h, p0/z, [%[inptr3], %[inpos], LSL #1]\n"
                    "ld1h z4.h, p0/z, [%[inptr4], %[inpos], LSL #1]\n"
                    "ld1h z5.h, p0/z, [%[inptr5], %[inpos], LSL #1]\n"
                    "ld1h z6.h, p0/z, [%[inptr6], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z6.d, z11.d, z15.d\n"
                    "zip2 z7.d, z11.d, z15.d\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.h, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "ld1h z0.h, p0/z, [%[inptr0], %[inpos], LSL #1]\n"
                    "ld1h z1.h, p0/z, [%[inptr1], %[inpos], LSL #1]\n"
                    "ld1h z2.h, p0/z, [%[inptr2], %[inpos], LSL #1]\n"
                    "ld1h z3.h, p0/z, [%[inptr3], %[inpos], LSL #1]\n"
                    "ld1h z4.h, p0/z, [%[inptr4], %[inpos], LSL #1]\n"
                    "ld1h z5.h, p0/z, [%[inptr5], %[inpos], LSL #1]\n"
                    "ld1h z6.h, p0/z, [%[inptr6], %[inpos], LSL #1]\n"
                    "ld1h z7.h, p0/z, [%[inptr7], %[inpos], LSL #1]\n"
                    "inch %[inpos], all, mul #1\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p0.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p1.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "zip1 z0.d, z8.d, z12.d\n"
                    "whilelt p2.h, %[outpos], %[outwidth]\n"
                    "zip2 z1.d, z8.d, z12.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z2.d, z9.d, z13.d\n"
                    "zip2 z3.d, z9.d, z13.d\n"
                    "zip1 z4.d, z10.d, z14.d\n"
                    "whilelt p3.h, %[outpos], %[outwidth]\n"
                    "zip2 z5.d, z10.d, z14.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z6.d, z11.d, z15.d\n"
                    "zip2 z7.d, z11.d, z15.d\n"
                    "zip1 z8.d, z0.d, z4.d\n"
                    "whilelt p4.h, %[outpos], %[outwidth]\n"
                    "zip2 z9.d, z0.d, z4.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip1 z10.d, z1.d, z5.d\n"
                    "st1h z8.h, p0, [%[outptr]]\n"
                    "zip2 z11.d, z1.d, z5.d\n"
                    "zip1 z12.d, z2.d, z6.d\n"
                    "whilelt p5.h, %[outpos], %[outwidth]\n"
                    "zip2 z13.d, z2.d, z6.d\n"
                    "st1h z9.h, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.d, z3.d, z7.d\n"
                    "inch %[outpos], all, mul #1\n"
                    "zip2 z15.d, z3.d, z7.d\n"
                    "st1h z10.h, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.h, %[outpos], %[outwidth]\n"
                    "st1h z11.h, p3, [%[outptr], #3, MUL VL]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z12.h, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.h, %[outpos], %[outwidth]\n"
                    "inch %[outpos], all, mul #1\n"
                    "st1h z13.h, p5, [%[outptr], #5, MUL VL]\n"
                    "st1h z14.h, p6, [%[outptr], #6, MUL VL]\n"
                    "st1h z15.h, p7, [%[outptr], #7, MUL VL]\n"
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
