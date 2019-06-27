/*
 * Copyright (c) 2018 - 2019 Arm Limited.
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
inline void TransformImpl<8, 4, false, 1, 1, false>::Transform(T *out, const T *in, int ldin, int y0, int ymax, int k0, int kmax)
{
    uint8_t *master_outptr = reinterpret_cast<uint8_t *>(out);
    const uint8_t *inptr = reinterpret_cast<const uint8_t *>(in);

    for (int y=y0; y<ymax; y+=8)
    {
        const int height = ymax-y;
        const long inwidth = (kmax - k0);
        const long outwidth = ((inwidth + 3) / 4) * 32;
        long inpos = 0;
        long outpos = 0;

        uint8_t *outptr = master_outptr;
        master_outptr += outwidth;

        const uint8_t *inptr0 = inptr + y * ldin + k0;
        const uint8_t *inptr1 = inptr0 + ldin;
        const uint8_t *inptr2 = inptr1 + ldin;
        const uint8_t *inptr3 = inptr2 + ldin;
        const uint8_t *inptr4 = inptr3 + ldin;
        const uint8_t *inptr5 = inptr4 + ldin;
        const uint8_t *inptr6 = inptr5 + ldin;
        const uint8_t *inptr7 = inptr6 + ldin;

        switch(height)
        {
            case 1:
                __asm __volatile(
                    "1:\n"
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip1 z0.s, z8.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z1.s, z8.s, z4.s\n"
                    "zip1 z2.s, z9.s, z4.s\n"
                    "zip2 z3.s, z9.s, z4.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z4.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.s, z3.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.b, #0\n"
                    "mov z14.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip1 z0.s, z8.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z1.s, z8.s, z4.s\n"
                    "zip1 z2.s, z9.s, z4.s\n"
                    "zip2 z3.s, z9.s, z4.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "zip1 z6.s, z11.s, z14.s\n"
                    "zip2 z7.s, z11.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.b, #0\n"
                    "mov z14.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "zip1 z6.s, z11.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z7.s, z11.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z4.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "ld1b z3.b, p0/z, [%[inptr3], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z4.s\n"
                    "zip2 z11.s, z1.s, z4.s\n"
                    "zip1 z12.s, z2.s, z4.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z4.s\n"
                    "zip2 z15.s, z3.s, z4.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z5.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "ld1b z3.b, p0/z, [%[inptr3], %[inpos]]\n"
                    "ld1b z4.b, p0/z, [%[inptr4], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z5.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z5.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z5.s\n"
                    "zip2 z15.s, z3.s, z5.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z6.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "ld1b z3.b, p0/z, [%[inptr3], %[inpos]]\n"
                    "ld1b z4.b, p0/z, [%[inptr4], %[inpos]]\n"
                    "ld1b z5.b, p0/z, [%[inptr5], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z6.s\n"
                    "zip2 z15.s, z3.s, z6.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "mov z7.b, #0\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "ld1b z3.b, p0/z, [%[inptr3], %[inpos]]\n"
                    "ld1b z4.b, p0/z, [%[inptr4], %[inpos]]\n"
                    "ld1b z5.b, p0/z, [%[inptr5], %[inpos]]\n"
                    "ld1b z6.b, p0/z, [%[inptr6], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
                    "whilelt p0.b, %[inpos], %[inwidth]\n"
                    "b.none 2f\n"
                    "ld1b z0.b, p0/z, [%[inptr0], %[inpos]]\n"
                    "ld1b z1.b, p0/z, [%[inptr1], %[inpos]]\n"
                    "ld1b z2.b, p0/z, [%[inptr2], %[inpos]]\n"
                    "ld1b z3.b, p0/z, [%[inptr3], %[inpos]]\n"
                    "ld1b z4.b, p0/z, [%[inptr4], %[inpos]]\n"
                    "ld1b z5.b, p0/z, [%[inptr5], %[inpos]]\n"
                    "ld1b z6.b, p0/z, [%[inptr6], %[inpos]]\n"
                    "ld1b z7.b, p0/z, [%[inptr7], %[inpos]]\n"
                    "incb %[inpos], all, mul #1\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p0.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p1.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "zip1 z0.s, z8.s, z12.s\n"
                    "whilelt p2.b, %[outpos], %[outwidth]\n"
                    "zip2 z1.s, z8.s, z12.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z2.s, z9.s, z13.s\n"
                    "zip2 z3.s, z9.s, z13.s\n"
                    "zip1 z4.s, z10.s, z14.s\n"
                    "whilelt p3.b, %[outpos], %[outwidth]\n"
                    "zip2 z5.s, z10.s, z14.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z6.s, z11.s, z15.s\n"
                    "zip2 z7.s, z11.s, z15.s\n"
                    "zip1 z8.s, z0.s, z4.s\n"
                    "whilelt p4.b, %[outpos], %[outwidth]\n"
                    "zip2 z9.s, z0.s, z4.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip1 z10.s, z1.s, z5.s\n"
                    "st1b z8.b, p0, [%[outptr]]\n"
                    "zip2 z11.s, z1.s, z5.s\n"
                    "zip1 z12.s, z2.s, z6.s\n"
                    "whilelt p5.b, %[outpos], %[outwidth]\n"
                    "zip2 z13.s, z2.s, z6.s\n"
                    "st1b z9.b, p1, [%[outptr], #1, MUL VL]\n"
                    "zip1 z14.s, z3.s, z7.s\n"
                    "incb %[outpos], all, mul #1\n"
                    "zip2 z15.s, z3.s, z7.s\n"
                    "st1b z10.b, p2, [%[outptr], #2, MUL VL]\n"
                    "whilelt p6.b, %[outpos], %[outwidth]\n"
                    "st1b z11.b, p3, [%[outptr], #3, MUL VL]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z12.b, p4, [%[outptr], #4, MUL VL]\n"
                    "whilelt p7.b, %[outpos], %[outwidth]\n"
                    "incb %[outpos], all, mul #1\n"
                    "st1b z13.b, p5, [%[outptr], #5, MUL VL]\n"
                    "st1b z14.b, p6, [%[outptr], #6, MUL VL]\n"
                    "st1b z15.b, p7, [%[outptr], #7, MUL VL]\n"
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
