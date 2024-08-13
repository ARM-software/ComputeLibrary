/*
 * Copyright (c) 2021, 2023-2024 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace {

void sve_transpose_interleave_12VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 12 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p2.b\n"
      "1:"  // Main row loop: Head
      "mov x28, %x[in]\n"
      "mov x27, %x[width]\n"
      "cnth x26, ALL, MUL #6\n"
      "cmp %x[height], #0x3\n"
      "mov x25, %x[out]\n"
      "add x24, x28, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add %x[in], x22, %x[in_stride]\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel x23, x23, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "cmp x27, x26\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z16.s }, p2/Z, [x28]\n"
      "ld1w { z22.s }, p2/Z, [x28, #1, MUL VL]\n"
      "mov x21, x25\n"
      "add x25, x25, %x[out_stride]\n"
      "ld1w { z30.s }, p2/Z, [x28, #2, MUL VL]\n"
      "ld1w { z11.s }, p2/Z, [x28, #3, MUL VL]\n"
      "mov x20, x25\n"
      "sub x27, x27, x26\n"
      "ld1w { z23.s }, p2/Z, [x28, #4, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x28, #5, MUL VL]\n"
      "cmp x27, x26\n"
      "add x25, x25, %x[out_stride]\n"
      "ld1w { z17.s }, p2/Z, [x28, #6, MUL VL]\n"
      "ld1w { z0.s }, p2/Z, [x28, #7, MUL VL]\n"
      "addvl x28, x28, #12\n"
      "ld1w { z10.s }, p2/Z, [x23]\n"
      "ld1w { z14.s }, p2/Z, [x23, #1, MUL VL]\n"
      "ld1w { z12.s }, p2/Z, [x23, #2, MUL VL]\n"
      "ld1w { z13.s }, p2/Z, [x23, #3, MUL VL]\n"
      "ld1w { z29.s }, p2/Z, [x23, #4, MUL VL]\n"
      "ld1w { z31.s }, p2/Z, [x23, #5, MUL VL]\n"
      "ld1w { z19.s }, p2/Z, [x23, #6, MUL VL]\n"
      "ld1w { z1.s }, p2/Z, [x23, #7, MUL VL]\n"
      "addvl x23, x23, #12\n"
      "zip1 z26.s, z16.s, z10.s\n"
      "ld1w { z2.s }, p2/Z, [x28, #-4, MUL VL]\n"
      "ld1w { z24.s }, p2/Z, [x28, #-3, MUL VL]\n"
      "zip2 z15.s, z16.s, z10.s\n"
      "zip1 z6.s, z22.s, z14.s\n"
      "ld1w { z27.s }, p2/Z, [x24]\n"
      "ld1w { z18.s }, p2/Z, [x24, #1, MUL VL]\n"
      "zip2 z28.s, z22.s, z14.s\n"
      "zip1 z25.s, z30.s, z12.s\n"
      "ld1w { z21.s }, p2/Z, [x24, #2, MUL VL]\n"
      "ld1w { z3.s }, p2/Z, [x24, #3, MUL VL]\n"
      "zip2 z7.s, z30.s, z12.s\n"
      "zip1 z9.s, z11.s, z13.s\n"
      "ld1w { z4.s }, p2/Z, [x24, #4, MUL VL]\n"
      "ld1w { z22.s }, p2/Z, [x24, #5, MUL VL]\n"
      "zip2 z16.s, z11.s, z13.s\n"
      ".inst 0x658aab4c  // bfcvt z12.h, p2/M, z26.s\n"
      "ld1w { z14.s }, p2/Z, [x24, #6, MUL VL]\n"
      "ld1w { z30.s }, p2/Z, [x24, #7, MUL VL]\n"
      "addvl x24, x24, #12\n"
      ".inst 0x658aa9ef  // bfcvt z15.h, p2/M, z15.s\n"
      "ld1w { z26.s }, p2/Z, [x23, #-4, MUL VL]\n"
      "ld1w { z5.s }, p2/Z, [x23, #-3, MUL VL]\n"
      ".inst 0x658aa8cd  // bfcvt z13.h, p2/M, z6.s\n"
      ".inst 0x658aab8a  // bfcvt z10.h, p2/M, z28.s\n"
      "ld1w { z28.s }, p2/Z, [x22]\n"
      "ld1w { z8.s }, p2/Z, [x22, #1, MUL VL]\n"
      ".inst 0x658aab39  // bfcvt z25.h, p2/M, z25.s\n"
      ".inst 0x658aa8e6  // bfcvt z6.h, p2/M, z7.s\n"
      "ld1w { z11.s }, p2/Z, [x22, #2, MUL VL]\n"
      ".inst 0x658aa927  // bfcvt z7.h, p2/M, z9.s\n"
      ".inst 0x658aaa10  // bfcvt z16.h, p2/M, z16.s\n"
      "zip1 z9.s, z23.s, z29.s\n"
      "zip2 z23.s, z23.s, z29.s\n"
      "zip1 z29.s, z27.s, z28.s\n"
      "zip2 z27.s, z27.s, z28.s\n"
      "ld1w { z28.s }, p2/Z, [x22, #3, MUL VL]\n"
      ".inst 0x658aa929  // bfcvt z9.h, p2/M, z9.s\n"
      ".inst 0x658aaaf7  // bfcvt z23.h, p2/M, z23.s\n"
      ".inst 0x648aabac  // bfcvtnt z12.h, p2/M, z29.s\n"
      "ld1w { z29.s }, p2/Z, [x22, #4, MUL VL]\n"
      ".inst 0x648aab6f  // bfcvtnt z15.h, p2/M, z27.s\n"
      "zip1 z27.s, z18.s, z8.s\n"
      "zip2 z8.s, z18.s, z8.s\n"
      "ld1w { z18.s }, p2/Z, [x22, #5, MUL VL]\n"
      ".inst 0x648aab6d  // bfcvtnt z13.h, p2/M, z27.s\n"
      "ld1w { z27.s }, p2/Z, [x22, #6, MUL VL]\n"
      ".inst 0x648aa90a  // bfcvtnt z10.h, p2/M, z8.s\n"
      "zip1 z8.s, z21.s, z11.s\n"
      "zip2 z21.s, z21.s, z11.s\n"
      "ld1w { z11.s }, p2/Z, [x22, #7, MUL VL]\n"
      "addvl x22, x22, #12\n"
      ".inst 0x648aa919  // bfcvtnt z25.h, p2/M, z8.s\n"
      "ld1w { z8.s }, p2/Z, [x28, #-2, MUL VL]\n"
      ".inst 0x648aaaa6  // bfcvtnt z6.h, p2/M, z21.s\n"
      "zip1 z21.s, z3.s, z28.s\n"
      "zip2 z3.s, z3.s, z28.s\n"
      "ld1w { z28.s }, p2/Z, [x28, #-1, MUL VL]\n"
      ".inst 0x648aaaa7  // bfcvtnt z7.h, p2/M, z21.s\n"
      "ld1w { z21.s }, p2/Z, [x24, #-4, MUL VL]\n"
      ".inst 0x648aa870  // bfcvtnt z16.h, p2/M, z3.s\n"
      "zip1 z3.s, z20.s, z31.s\n"
      "zip2 z31.s, z20.s, z31.s\n"
      "zip1 z20.s, z17.s, z19.s\n"
      "zip2 z17.s, z17.s, z19.s\n"
      "zip1 z19.s, z0.s, z1.s\n"
      "zip2 z1.s, z0.s, z1.s\n"
      "zip1 z0.s, z2.s, z26.s\n"
      "zip2 z2.s, z2.s, z26.s\n"
      "zip1 z26.s, z24.s, z5.s\n"
      "zip2 z24.s, z24.s, z5.s\n"
      "zip1 z5.s, z4.s, z29.s\n"
      "zip2 z4.s, z4.s, z29.s\n"
      "ld1w { z29.s }, p2/Z, [x24, #-3, MUL VL]\n"
      ".inst 0x658aa863  // bfcvt z3.h, p2/M, z3.s\n"
      ".inst 0x658aabff  // bfcvt z31.h, p2/M, z31.s\n"
      ".inst 0x658aaa94  // bfcvt z20.h, p2/M, z20.s\n"
      ".inst 0x658aaa31  // bfcvt z17.h, p2/M, z17.s\n"
      ".inst 0x658aaa73  // bfcvt z19.h, p2/M, z19.s\n"
      ".inst 0x658aa821  // bfcvt z1.h, p2/M, z1.s\n"
      ".inst 0x658aa800  // bfcvt z0.h, p2/M, z0.s\n"
      ".inst 0x658aa842  // bfcvt z2.h, p2/M, z2.s\n"
      ".inst 0x658aab5a  // bfcvt z26.h, p2/M, z26.s\n"
      ".inst 0x658aab18  // bfcvt z24.h, p2/M, z24.s\n"
      ".inst 0x648aa8a9  // bfcvtnt z9.h, p2/M, z5.s\n"
      "ld1w { z5.s }, p2/Z, [x23, #-2, MUL VL]\n"
      ".inst 0x648aa897  // bfcvtnt z23.h, p2/M, z4.s\n"
      "zip1 z4.s, z22.s, z18.s\n"
      "zip2 z22.s, z22.s, z18.s\n"
      "ld1w { z18.s }, p2/Z, [x23, #-1, MUL VL]\n"
      ".inst 0x648aa883  // bfcvtnt z3.h, p2/M, z4.s\n"
      "ld1w { z4.s }, p2/Z, [x22, #-4, MUL VL]\n"
      ".inst 0x648aaadf  // bfcvtnt z31.h, p2/M, z22.s\n"
      "zip1 z22.s, z14.s, z27.s\n"
      "zip2 z14.s, z14.s, z27.s\n"
      "ld1w { z27.s }, p2/Z, [x22, #-3, MUL VL]\n"
      ".inst 0x648aaad4  // bfcvtnt z20.h, p2/M, z22.s\n"
      "ld1w { z22.s }, p2/Z, [x24, #-2, MUL VL]\n"
      ".inst 0x648aa9d1  // bfcvtnt z17.h, p2/M, z14.s\n"
      "zip1 z14.s, z30.s, z11.s\n"
      "zip2 z11.s, z30.s, z11.s\n"
      "ld1w { z30.s }, p2/Z, [x24, #-1, MUL VL]\n"
      ".inst 0x648aa9d3  // bfcvtnt z19.h, p2/M, z14.s\n"
      "ld1w { z14.s }, p2/Z, [x22, #-2, MUL VL]\n"
      ".inst 0x648aa961  // bfcvtnt z1.h, p2/M, z11.s\n"
      "ld1w { z11.s }, p2/Z, [x22, #-1, MUL VL]\n"
      "st1h { z12.h }, p2, [x21]\n"
      "zip1 z12.s, z21.s, z4.s\n"
      "zip2 z21.s, z21.s, z4.s\n"
      "zip1 z4.s, z29.s, z27.s\n"
      "zip2 z29.s, z29.s, z27.s\n"
      "st1h { z15.h }, p2, [x21, #1, MUL VL]\n"
      "zip1 z27.s, z8.s, z5.s\n"
      "zip2 z8.s, z8.s, z5.s\n"
      "st1h { z13.h }, p2, [x21, #2, MUL VL]\n"
      "zip1 z5.s, z28.s, z18.s\n"
      "zip2 z28.s, z28.s, z18.s\n"
      "st1h { z10.h }, p2, [x21, #3, MUL VL]\n"
      "st1h { z25.h }, p2, [x21, #4, MUL VL]\n"
      ".inst 0x648aa980  // bfcvtnt z0.h, p2/M, z12.s\n"
      ".inst 0x648aaaa2  // bfcvtnt z2.h, p2/M, z21.s\n"
      "st1h { z6.h }, p2, [x21, #5, MUL VL]\n"
      ".inst 0x648aa89a  // bfcvtnt z26.h, p2/M, z4.s\n"
      ".inst 0x648aabb8  // bfcvtnt z24.h, p2/M, z29.s\n"
      "st1h { z7.h }, p2, [x21, #6, MUL VL]\n"
      ".inst 0x658aab7b  // bfcvt z27.h, p2/M, z27.s\n"
      "zip1 z25.s, z22.s, z14.s\n"
      "st1h { z16.h }, p2, [x21, #7, MUL VL]\n"
      "addvl x21, x21, #12\n"
      ".inst 0x658aa906  // bfcvt z6.h, p2/M, z8.s\n"
      "zip2 z4.s, z22.s, z14.s\n"
      ".inst 0x658aa8b2  // bfcvt z18.h, p2/M, z5.s\n"
      "zip1 z22.s, z30.s, z11.s\n"
      ".inst 0x658aab95  // bfcvt z21.h, p2/M, z28.s\n"
      "zip2 z16.s, z30.s, z11.s\n"
      "st1h { z9.h }, p2, [x21, #-4, MUL VL]\n"
      "st1h { z23.h }, p2, [x21, #-3, MUL VL]\n"
      ".inst 0x648aab3b  // bfcvtnt z27.h, p2/M, z25.s\n"
      ".inst 0x648aa886  // bfcvtnt z6.h, p2/M, z4.s\n"
      "st1h { z3.h }, p2, [x21, #-2, MUL VL]\n"
      ".inst 0x648aaad2  // bfcvtnt z18.h, p2/M, z22.s\n"
      "st1h { z31.h }, p2, [x21, #-1, MUL VL]\n"
      ".inst 0x648aaa15  // bfcvtnt z21.h, p2/M, z16.s\n"
      "st1h { z20.h }, p2, [x20]\n"
      "st1h { z17.h }, p2, [x20, #1, MUL VL]\n"
      "st1h { z19.h }, p2, [x20, #2, MUL VL]\n"
      "st1h { z1.h }, p2, [x20, #3, MUL VL]\n"
      "st1h { z0.h }, p2, [x20, #4, MUL VL]\n"
      "st1h { z2.h }, p2, [x20, #5, MUL VL]\n"
      "st1h { z26.h }, p2, [x20, #6, MUL VL]\n"
      "st1h { z24.h }, p2, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #12\n"
      "st1h { z27.h }, p2, [x20, #-4, MUL VL]\n"
      "st1h { z6.h }, p2, [x20, #-3, MUL VL]\n"
      "st1h { z18.h }, p2, [x20, #-2, MUL VL]\n"
      "st1h { z21.h }, p2, [x20, #-1, MUL VL]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x27, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x21, x27\n"
      "mov x20, x25\n"
      "decd x27, ALL, MUL #12\n"
      "add x25, x25, %x[out_stride]\n"
      "whilelt p1.s, XZR, x21\n"
      "decw x21\n"
      "whilelt p0.s, XZR, x21\n"
      "decw x21\n"
      "ld1w { z19.s }, p1/Z, [x28]\n"
      "ld1w { z18.s }, p1/Z, [x23]\n"
      "ld1w { z30.s }, p1/Z, [x24]\n"
      "ld1w { z29.s }, p1/Z, [x22]\n"
      "whilelt p1.s, XZR, x21\n"
      "decw x21\n"
      "ld1w { z21.s }, p0/Z, [x28, #1, MUL VL]\n"
      "ld1w { z17.s }, p0/Z, [x23, #1, MUL VL]\n"
      "ld1w { z28.s }, p0/Z, [x24, #1, MUL VL]\n"
      "ld1w { z27.s }, p0/Z, [x22, #1, MUL VL]\n"
      "zip1 z16.s, z19.s, z18.s\n"
      "zip2 z26.s, z19.s, z18.s\n"
      "whilelt p0.s, XZR, x21\n"
      "decw x21\n"
      "ld1w { z20.s }, p1/Z, [x28, #2, MUL VL]\n"
      "ld1w { z19.s }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1w { z25.s }, p1/Z, [x24, #2, MUL VL]\n"
      "ld1w { z24.s }, p1/Z, [x22, #2, MUL VL]\n"
      "zip1 z18.s, z21.s, z17.s\n"
      "zip2 z23.s, z21.s, z17.s\n"
      ".inst 0x658aaa0a  // bfcvt z10.h, p2/M, z16.s\n"
      "zip1 z9.s, z30.s, z29.s\n"
      "whilelt p1.s, XZR, x21\n"
      "decw x21\n"
      "ld1w { z17.s }, p0/Z, [x28, #3, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x23, #3, MUL VL]\n"
      "zip1 z22.s, z20.s, z19.s\n"
      "zip2 z21.s, z20.s, z19.s\n"
      "ld1w { z20.s }, p0/Z, [x24, #3, MUL VL]\n"
      "ld1w { z19.s }, p0/Z, [x22, #3, MUL VL]\n"
      ".inst 0x658aab48  // bfcvt z8.h, p2/M, z26.s\n"
      "zip2 z7.s, z30.s, z29.s\n"
      "whilelt p0.s, XZR, x21\n"
      "ld1w { z6.s }, p1/Z, [x28, #4, MUL VL]\n"
      "ld1w { z5.s }, p1/Z, [x23, #4, MUL VL]\n"
      ".inst 0x658aaa44  // bfcvt z4.h, p2/M, z18.s\n"
      "zip1 z18.s, z17.s, z16.s\n"
      "zip2 z17.s, z17.s, z16.s\n"
      "ld1w { z3.s }, p1/Z, [x24, #4, MUL VL]\n"
      "ld1w { z2.s }, p1/Z, [x22, #4, MUL VL]\n"
      "zip1 z1.s, z28.s, z27.s\n"
      ".inst 0x658aaae0  // bfcvt z0.h, p2/M, z23.s\n"
      "cmp x27, #0x0\n"
      "ld1w { z31.s }, p0/Z, [x28, #5, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x23, #5, MUL VL]\n"
      "ld1w { z30.s }, p0/Z, [x24, #5, MUL VL]\n"
      "zip2 z29.s, z28.s, z27.s\n"
      ".inst 0x658aaadc  // bfcvt z28.h, p2/M, z22.s\n"
      "ld1w { z27.s }, p0/Z, [x22, #5, MUL VL]\n"
      "zip1 z23.s, z25.s, z24.s\n"
      ".inst 0x658aaaba  // bfcvt z26.h, p2/M, z21.s\n"
      "addvl x28, x28, #6\n"
      "zip2 z22.s, z25.s, z24.s\n"
      ".inst 0x658aaa59  // bfcvt z25.h, p2/M, z18.s\n"
      "addvl x24, x24, #6\n"
      "addvl x23, x23, #6\n"
      "zip1 z21.s, z20.s, z19.s\n"
      ".inst 0x658aaa38  // bfcvt z24.h, p2/M, z17.s\n"
      "addvl x22, x22, #6\n"
      "zip2 z20.s, z20.s, z19.s\n"
      "zip1 z19.s, z6.s, z5.s\n"
      "zip2 z18.s, z6.s, z5.s\n"
      "zip1 z17.s, z31.s, z16.s\n"
      "zip2 z16.s, z31.s, z16.s\n"
      ".inst 0x648aa92a  // bfcvtnt z10.h, p2/M, z9.s\n"
      ".inst 0x648aa8e8  // bfcvtnt z8.h, p2/M, z7.s\n"
      ".inst 0x648aa824  // bfcvtnt z4.h, p2/M, z1.s\n"
      ".inst 0x648aaba0  // bfcvtnt z0.h, p2/M, z29.s\n"
      ".inst 0x648aaafc  // bfcvtnt z28.h, p2/M, z23.s\n"
      ".inst 0x648aaada  // bfcvtnt z26.h, p2/M, z22.s\n"
      ".inst 0x648aaab9  // bfcvtnt z25.h, p2/M, z21.s\n"
      "st1h { z10.h }, p2, [x20]\n"
      ".inst 0x648aaa98  // bfcvtnt z24.h, p2/M, z20.s\n"
      ".inst 0x658aaa77  // bfcvt z23.h, p2/M, z19.s\n"
      "st1h { z8.h }, p2, [x20, #1, MUL VL]\n"
      "zip1 z22.s, z3.s, z2.s\n"
      ".inst 0x658aaa55  // bfcvt z21.h, p2/M, z18.s\n"
      "st1h { z4.h }, p2, [x20, #2, MUL VL]\n"
      "zip2 z20.s, z3.s, z2.s\n"
      ".inst 0x658aaa33  // bfcvt z19.h, p2/M, z17.s\n"
      "st1h { z0.h }, p2, [x20, #3, MUL VL]\n"
      "zip1 z18.s, z30.s, z27.s\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "st1h { z28.h }, p2, [x20, #4, MUL VL]\n"
      "zip2 z16.s, z30.s, z27.s\n"
      "st1h { z26.h }, p2, [x20, #5, MUL VL]\n"
      ".inst 0x648aaad7  // bfcvtnt z23.h, p2/M, z22.s\n"
      "st1h { z25.h }, p2, [x20, #6, MUL VL]\n"
      ".inst 0x648aaa95  // bfcvtnt z21.h, p2/M, z20.s\n"
      "st1h { z24.h }, p2, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #12\n"
      ".inst 0x648aaa53  // bfcvtnt z19.h, p2/M, z18.s\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z23.h }, p2, [x20, #-4, MUL VL]\n"
      "st1h { z21.h }, p2, [x20, #-3, MUL VL]\n"
      "st1h { z19.h }, p2, [x20, #-2, MUL VL]\n"
      "st1h { z17.h }, p2, [x20, #-1, MUL VL]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #12\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace
template<>
void Transform<12, 4, true, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_12VL_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}


#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
