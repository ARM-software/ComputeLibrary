/*
 * Copyright (c) 2021 Arm Limited.
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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#pragma once

#ifdef __ARM_FEATURE_SVE


namespace {

void sve_transpose_interleave_8VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 8 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p1.b\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[out]\n"
      "add x23, x25, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "cmp %x[height], #0x3\n"
      "csel x21, x21, %x[pad_row], GT\n"
      "csel x22, x22, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x23, x23, %x[pad_row], GT\n"
      "sub %x[height], %x[height], #0x4\n"
      "mov x20, %x[width]\n"
      "cntb x19, ALL, MUL #8\n"
      "cmp x20, x19\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1b { z8.b }, p1/Z, [x25]\n"
      "sub x20, x20, x19\n"
      "ld1b { z24.b }, p1/Z, [x25, #1, MUL VL]\n"
      "cmp x20, x19\n"
      "ld1b { z27.b }, p1/Z, [x25, #2, MUL VL]\n"
      "ld1b { z25.b }, p1/Z, [x25, #3, MUL VL]\n"
      "ld1b { z7.b }, p1/Z, [x25, #4, MUL VL]\n"
      "ld1b { z3.b }, p1/Z, [x25, #5, MUL VL]\n"
      "ld1b { z14.b }, p1/Z, [x25, #6, MUL VL]\n"
      "ld1b { z13.b }, p1/Z, [x25, #7, MUL VL]\n"
      "addvl x25, x25, #8\n"
      "ld1b { z16.b }, p1/Z, [x23]\n"
      "ld1b { z12.b }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1b { z15.b }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1b { z11.b }, p1/Z, [x23, #3, MUL VL]\n"
      "ld1b { z4.b }, p1/Z, [x23, #4, MUL VL]\n"
      "ld1b { z5.b }, p1/Z, [x23, #5, MUL VL]\n"
      "ld1b { z26.b }, p1/Z, [x23, #6, MUL VL]\n"
      "ld1b { z30.b }, p1/Z, [x23, #7, MUL VL]\n"
      "addvl x23, x23, #8\n"
      "ld1b { z22.b }, p1/Z, [x22]\n"
      "zip1 z21.b, z8.b, z22.b\n"
      "ld1b { z2.b }, p1/Z, [x22, #1, MUL VL]\n"
      "zip2 z20.b, z8.b, z22.b\n"
      "ld1b { z18.b }, p1/Z, [x22, #2, MUL VL]\n"
      "ld1b { z17.b }, p1/Z, [x22, #3, MUL VL]\n"
      "zip1 z10.b, z24.b, z2.b\n"
      "ld1b { z22.b }, p1/Z, [x22, #4, MUL VL]\n"
      "zip2 z9.b, z24.b, z2.b\n"
      "ld1b { z6.b }, p1/Z, [x22, #5, MUL VL]\n"
      "zip1 z0.b, z27.b, z18.b\n"
      "ld1b { z1.b }, p1/Z, [x22, #6, MUL VL]\n"
      "zip2 z28.b, z27.b, z18.b\n"
      "ld1b { z23.b }, p1/Z, [x22, #7, MUL VL]\n"
      "addvl x22, x22, #8\n"
      "zip1 z31.b, z25.b, z17.b\n"
      "ld1b { z19.b }, p1/Z, [x21]\n"
      "zip2 z8.b, z25.b, z17.b\n"
      "ld1b { z2.b }, p1/Z, [x21, #1, MUL VL]\n"
      "zip1 z27.b, z7.b, z22.b\n"
      "ld1b { z29.b }, p1/Z, [x21, #2, MUL VL]\n"
      "zip2 z7.b, z7.b, z22.b\n"
      "ld1b { z24.b }, p1/Z, [x21, #3, MUL VL]\n"
      "zip1 z18.b, z16.b, z19.b\n"
      "ld1b { z25.b }, p1/Z, [x21, #4, MUL VL]\n"
      "zip1 z17.b, z21.b, z18.b\n"
      "ld1b { z22.b }, p1/Z, [x21, #5, MUL VL]\n"
      "zip2 z18.b, z21.b, z18.b\n"
      "ld1b { z21.b }, p1/Z, [x21, #6, MUL VL]\n"
      "zip2 z16.b, z16.b, z19.b\n"
      "ld1b { z19.b }, p1/Z, [x21, #7, MUL VL]\n"
      "addvl x21, x21, #8\n"
      "st1b { z17.b }, p1, [x24]\n"
      "zip1 z17.b, z20.b, z16.b\n"
      "zip2 z20.b, z20.b, z16.b\n"
      "st1b { z18.b }, p1, [x24, #1, MUL VL]\n"
      "zip1 z16.b, z12.b, z2.b\n"
      "st1b { z17.b }, p1, [x24, #2, MUL VL]\n"
      "zip1 z17.b, z10.b, z16.b\n"
      "st1b { z20.b }, p1, [x24, #3, MUL VL]\n"
      "zip2 z16.b, z10.b, z16.b\n"
      "st1b { z17.b }, p1, [x24, #4, MUL VL]\n"
      "zip2 z17.b, z12.b, z2.b\n"
      "st1b { z16.b }, p1, [x24, #5, MUL VL]\n"
      "zip1 z16.b, z9.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #6, MUL VL]\n"
      "zip2 z16.b, z9.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "zip1 z18.b, z15.b, z29.b\n"
      "zip2 z17.b, z15.b, z29.b\n"
      "zip1 z16.b, z0.b, z18.b\n"
      "st1b { z16.b }, p1, [x24]\n"
      "zip2 z16.b, z0.b, z18.b\n"
      "st1b { z16.b }, p1, [x24, #1, MUL VL]\n"
      "zip1 z16.b, z28.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #2, MUL VL]\n"
      "zip2 z16.b, z28.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #3, MUL VL]\n"
      "zip1 z17.b, z11.b, z24.b\n"
      "zip1 z16.b, z31.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #4, MUL VL]\n"
      "zip2 z16.b, z31.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #5, MUL VL]\n"
      "zip2 z17.b, z11.b, z24.b\n"
      "zip1 z16.b, z8.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #6, MUL VL]\n"
      "zip2 z16.b, z8.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "zip1 z18.b, z4.b, z25.b\n"
      "zip2 z17.b, z4.b, z25.b\n"
      "zip1 z16.b, z27.b, z18.b\n"
      "st1b { z16.b }, p1, [x24]\n"
      "zip2 z16.b, z27.b, z18.b\n"
      "st1b { z16.b }, p1, [x24, #1, MUL VL]\n"
      "zip1 z16.b, z7.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #2, MUL VL]\n"
      "zip2 z16.b, z7.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #3, MUL VL]\n"
      "zip1 z18.b, z3.b, z6.b\n"
      "zip1 z17.b, z5.b, z22.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #4, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #5, MUL VL]\n"
      "zip2 z18.b, z3.b, z6.b\n"
      "zip2 z17.b, z5.b, z22.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #6, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "zip1 z18.b, z14.b, z1.b\n"
      "zip1 z17.b, z26.b, z21.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #1, MUL VL]\n"
      "zip2 z18.b, z14.b, z1.b\n"
      "zip2 z17.b, z26.b, z21.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #2, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #3, MUL VL]\n"
      "zip1 z18.b, z13.b, z23.b\n"
      "zip1 z17.b, z30.b, z19.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #4, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #5, MUL VL]\n"
      "zip2 z18.b, z13.b, z23.b\n"
      "zip2 z17.b, z30.b, z19.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #6, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x20, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x19, x20\n"
      "decw x20, ALL, MUL #8\n"
      "whilelt p0.b, XZR, x19\n"
      "ld1b { z17.b }, p0/Z, [x25]\n"
      "ld1b { z25.b }, p0/Z, [x23]\n"
      "decb x19\n"
      "ld1b { z16.b }, p0/Z, [x22]\n"
      "zip1 z18.b, z17.b, z16.b\n"
      "ld1b { z24.b }, p0/Z, [x21]\n"
      "whilelt p0.b, XZR, x19\n"
      "zip2 z23.b, z17.b, z16.b\n"
      "ld1b { z22.b }, p0/Z, [x25, #1, MUL VL]\n"
      "addvl x25, x25, #2\n"
      "zip1 z16.b, z25.b, z24.b\n"
      "ld1b { z21.b }, p0/Z, [x23, #1, MUL VL]\n"
      "addvl x23, x23, #2\n"
      "zip1 z17.b, z18.b, z16.b\n"
      "ld1b { z20.b }, p0/Z, [x22, #1, MUL VL]\n"
      "addvl x22, x22, #2\n"
      "zip2 z18.b, z18.b, z16.b\n"
      "ld1b { z19.b }, p0/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "zip2 z16.b, z25.b, z24.b\n"
      "st1b { z17.b }, p1, [x24]\n"
      "cmp x20, #0x0\n"
      "zip1 z17.b, z23.b, z16.b\n"
      "st1b { z18.b }, p1, [x24, #1, MUL VL]\n"
      "zip2 z16.b, z23.b, z16.b\n"
      "st1b { z17.b }, p1, [x24, #2, MUL VL]\n"
      "zip1 z18.b, z22.b, z20.b\n"
      "st1b { z16.b }, p1, [x24, #3, MUL VL]\n"
      "zip1 z17.b, z21.b, z19.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #4, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #5, MUL VL]\n"
      "zip2 z18.b, z22.b, z20.b\n"
      "zip2 z17.b, z21.b, z19.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #6, MUL VL]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "addvl %x[out], %x[out], #8\n"
      "cmp %x[height], #0x1\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<8, 4, true, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_8VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<8, 4, true, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_8VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif
