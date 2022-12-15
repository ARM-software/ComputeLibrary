/*
 * Copyright (c) 2021, 2023 Arm Limited.
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


namespace {

void sve_transpose_interleave_8VL_2x2(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    uint16_t *pad_row = reinterpret_cast<uint16_t *>(alloca(width * sizeof(uint16_t)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(uint16_t));
    }

    size_t out_stride = 8 * roundup<size_t>(height, 2) * get_vector_length<uint16_t>();

    __asm__ __volatile__(
      "cmp %x[height], #0x4\n"
      "ptrue p4.b\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x28, %x[in]\n"
      "mov x27, %x[width]\n"
      "cnth x26, ALL, MUL #8\n"
      "add x25, x28, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "cmp x27, x26\n"
      "add %x[in], x23, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1h { z30.h }, p4/Z, [x28]\n"
      "ld1h { z12.h }, p4/Z, [x28, #1, MUL VL]\n"
      "mov x21, x22\n"
      "add x22, x22, %x[out_stride]\n"
      "ld1h { z31.h }, p4/Z, [x28, #2, MUL VL]\n"
      "ld1h { z18.h }, p4/Z, [x28, #3, MUL VL]\n"
      "mov x20, x22\n"
      "sub x27, x27, x26\n"
      "ld1h { z20.h }, p4/Z, [x25]\n"
      "ld1h { z17.h }, p4/Z, [x25, #1, MUL VL]\n"
      "zip1 z3.h, z30.h, z20.h\n"
      "zip2 z21.h, z30.h, z20.h\n"
      "ld1h { z26.h }, p4/Z, [x25, #2, MUL VL]\n"
      "ld1h { z23.h }, p4/Z, [x25, #3, MUL VL]\n"
      "zip1 z13.h, z12.h, z17.h\n"
      "zip2 z0.h, z12.h, z17.h\n"
      "ld1h { z2.h }, p4/Z, [x28, #4, MUL VL]\n"
      "ld1h { z24.h }, p4/Z, [x28, #5, MUL VL]\n"
      "zip1 z12.h, z31.h, z26.h\n"
      "zip2 z14.h, z31.h, z26.h\n"
      "ld1h { z17.h }, p4/Z, [x28, #6, MUL VL]\n"
      "ld1h { z29.h }, p4/Z, [x28, #7, MUL VL]\n"
      "zip1 z16.h, z18.h, z23.h\n"
      "zip2 z15.h, z18.h, z23.h\n"
      "ld1h { z9.h }, p4/Z, [x25, #4, MUL VL]\n"
      "ld1h { z18.h }, p4/Z, [x25, #5, MUL VL]\n"
      "zip1 z11.h, z2.h, z9.h\n"
      "zip2 z5.h, z2.h, z9.h\n"
      "ld1h { z7.h }, p4/Z, [x25, #6, MUL VL]\n"
      "ld1h { z2.h }, p4/Z, [x25, #7, MUL VL]\n"
      "zip1 z10.h, z24.h, z18.h\n"
      "zip2 z6.h, z24.h, z18.h\n"
      "ld1h { z19.h }, p4/Z, [x24]\n"
      "ld1h { z18.h }, p4/Z, [x24, #1, MUL VL]\n"
      "zip1 z9.h, z17.h, z7.h\n"
      "zip2 z4.h, z17.h, z7.h\n"
      "ld1h { z24.h }, p4/Z, [x24, #2, MUL VL]\n"
      "ld1h { z22.h }, p4/Z, [x24, #3, MUL VL]\n"
      "zip1 z7.h, z29.h, z2.h\n"
      "zip2 z8.h, z29.h, z2.h\n"
      "ld1h { z25.h }, p4/Z, [x24, #4, MUL VL]\n"
      "ld1h { z17.h }, p4/Z, [x24, #5, MUL VL]\n"
      "cmp x27, x26\n"
      "addvl x28, x28, #8\n"
      "ld1h { z2.h }, p4/Z, [x24, #6, MUL VL]\n"
      "ld1h { z30.h }, p4/Z, [x24, #7, MUL VL]\n"
      "addvl x25, x25, #8\n"
      "addvl x24, x24, #8\n"
      "ld1h { z20.h }, p4/Z, [x23]\n"
      "ld1h { z27.h }, p4/Z, [x23, #1, MUL VL]\n"
      "zip1 z31.h, z19.h, z20.h\n"
      "zip2 z29.h, z19.h, z20.h\n"
      "ld1h { z26.h }, p4/Z, [x23, #2, MUL VL]\n"
      "ld1h { z23.h }, p4/Z, [x23, #3, MUL VL]\n"
      "zip1 z28.h, z18.h, z27.h\n"
      "zip2 z1.h, z18.h, z27.h\n"
      "ld1h { z20.h }, p4/Z, [x23, #4, MUL VL]\n"
      "ld1h { z19.h }, p4/Z, [x23, #5, MUL VL]\n"
      "zip1 z27.h, z24.h, z26.h\n"
      "zip2 z26.h, z24.h, z26.h\n"
      "ld1h { z18.h }, p4/Z, [x23, #6, MUL VL]\n"
      "ld1h { z24.h }, p4/Z, [x23, #7, MUL VL]\n"
      "st1h { z3.h }, p4, [x21]\n"
      "zip1 z3.h, z22.h, z23.h\n"
      "st1h { z21.h }, p4, [x21, #1, MUL VL]\n"
      "zip2 z22.h, z22.h, z23.h\n"
      "addvl x23, x23, #8\n"
      "zip1 z23.h, z25.h, z20.h\n"
      "st1h { z13.h }, p4, [x21, #2, MUL VL]\n"
      "zip2 z25.h, z25.h, z20.h\n"
      "zip1 z21.h, z17.h, z19.h\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z0.h }, p4, [x21, #3, MUL VL]\n"
      "zip2 z20.h, z17.h, z19.h\n"
      "zip1 z19.h, z2.h, z18.h\n"
      "st1h { z12.h }, p4, [x21, #4, MUL VL]\n"
      "zip2 z18.h, z2.h, z18.h\n"
      "zip1 z17.h, z30.h, z24.h\n"
      "st1h { z14.h }, p4, [x21, #5, MUL VL]\n"
      "zip2 z13.h, z30.h, z24.h\n"
      "st1h { z16.h }, p4, [x21, #6, MUL VL]\n"
      "st1h { z15.h }, p4, [x21, #7, MUL VL]\n"
      "addvl x21, x21, #16\n"
      "st1h { z31.h }, p4, [x21, #-8, MUL VL]\n"
      "st1h { z29.h }, p4, [x21, #-7, MUL VL]\n"
      "st1h { z28.h }, p4, [x21, #-6, MUL VL]\n"
      "st1h { z1.h }, p4, [x21, #-5, MUL VL]\n"
      "st1h { z27.h }, p4, [x21, #-4, MUL VL]\n"
      "st1h { z26.h }, p4, [x21, #-3, MUL VL]\n"
      "st1h { z3.h }, p4, [x21, #-2, MUL VL]\n"
      "st1h { z22.h }, p4, [x21, #-1, MUL VL]\n"
      "st1h { z11.h }, p4, [x20]\n"
      "st1h { z5.h }, p4, [x20, #1, MUL VL]\n"
      "st1h { z10.h }, p4, [x20, #2, MUL VL]\n"
      "st1h { z6.h }, p4, [x20, #3, MUL VL]\n"
      "st1h { z9.h }, p4, [x20, #4, MUL VL]\n"
      "st1h { z4.h }, p4, [x20, #5, MUL VL]\n"
      "st1h { z7.h }, p4, [x20, #6, MUL VL]\n"
      "st1h { z8.h }, p4, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1h { z23.h }, p4, [x20, #-8, MUL VL]\n"
      "st1h { z25.h }, p4, [x20, #-7, MUL VL]\n"
      "st1h { z21.h }, p4, [x20, #-6, MUL VL]\n"
      "st1h { z20.h }, p4, [x20, #-5, MUL VL]\n"
      "st1h { z19.h }, p4, [x20, #-4, MUL VL]\n"
      "st1h { z18.h }, p4, [x20, #-3, MUL VL]\n"
      "st1h { z17.h }, p4, [x20, #-2, MUL VL]\n"
      "st1h { z13.h }, p4, [x20, #-1, MUL VL]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x27, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x27\n"
      "whilelt p3.h, XZR, x20\n"
      "ld1h { z20.h }, p3/Z, [x28]\n"
      "ld1h { z19.h }, p3/Z, [x25]\n"
      "dech x20\n"
      "whilelt p2.h, XZR, x20\n"
      "ld1h { z18.h }, p2/Z, [x28, #1, MUL VL]\n"
      "ld1h { z17.h }, p2/Z, [x25, #1, MUL VL]\n"
      "dech x20\n"
      "whilelt p1.h, XZR, x20\n"
      "ld1h { z25.h }, p1/Z, [x28, #2, MUL VL]\n"
      "ld1h { z16.h }, p1/Z, [x25, #2, MUL VL]\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "ld1h { z0.h }, p0/Z, [x28, #3, MUL VL]\n"
      "ld1h { z24.h }, p0/Z, [x25, #3, MUL VL]\n"
      "mov x20, x22\n"
      "decw x27, ALL, MUL #8\n"
      "ld1h { z31.h }, p3/Z, [x24]\n"
      "ld1h { z30.h }, p2/Z, [x24, #1, MUL VL]\n"
      "ld1h { z29.h }, p1/Z, [x24, #2, MUL VL]\n"
      "ld1h { z28.h }, p0/Z, [x24, #3, MUL VL]\n"
      "zip1 z23.h, z20.h, z19.h\n"
      "zip2 z22.h, z20.h, z19.h\n"
      "ld1h { z21.h }, p3/Z, [x23]\n"
      "ld1h { z27.h }, p2/Z, [x23, #1, MUL VL]\n"
      "zip1 z20.h, z18.h, z17.h\n"
      "zip2 z19.h, z18.h, z17.h\n"
      "ld1h { z18.h }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1h { z26.h }, p0/Z, [x23, #3, MUL VL]\n"
      "zip1 z17.h, z25.h, z16.h\n"
      "zip2 z16.h, z25.h, z16.h\n"
      "zip1 z25.h, z0.h, z24.h\n"
      "zip2 z24.h, z0.h, z24.h\n"
      "st1h { z23.h }, p4, [x20]\n"
      "cmp x27, #0x0\n"
      "st1h { z22.h }, p4, [x20, #1, MUL VL]\n"
      "addvl x28, x28, #4\n"
      "addvl x25, x25, #4\n"
      "zip1 z23.h, z31.h, z21.h\n"
      "st1h { z20.h }, p4, [x20, #2, MUL VL]\n"
      "addvl x24, x24, #4\n"
      "addvl x23, x23, #4\n"
      "zip2 z22.h, z31.h, z21.h\n"
      "st1h { z19.h }, p4, [x20, #3, MUL VL]\n"
      "zip1 z21.h, z30.h, z27.h\n"
      "zip2 z20.h, z30.h, z27.h\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z17.h }, p4, [x20, #4, MUL VL]\n"
      "zip1 z19.h, z29.h, z18.h\n"
      "zip2 z18.h, z29.h, z18.h\n"
      "st1h { z16.h }, p4, [x20, #5, MUL VL]\n"
      "zip1 z17.h, z28.h, z26.h\n"
      "zip2 z16.h, z28.h, z26.h\n"
      "st1h { z25.h }, p4, [x20, #6, MUL VL]\n"
      "st1h { z24.h }, p4, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1h { z23.h }, p4, [x20, #-8, MUL VL]\n"
      "st1h { z22.h }, p4, [x20, #-7, MUL VL]\n"
      "st1h { z21.h }, p4, [x20, #-6, MUL VL]\n"
      "st1h { z20.h }, p4, [x20, #-5, MUL VL]\n"
      "st1h { z19.h }, p4, [x20, #-4, MUL VL]\n"
      "st1h { z18.h }, p4, [x20, #-3, MUL VL]\n"
      "st1h { z17.h }, p4, [x20, #-2, MUL VL]\n"
      "st1h { z16.h }, p4, [x20, #-1, MUL VL]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x4\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x28, %x[in]\n"
      "mov x21, %x[width]\n"
      "cnth x20, ALL, MUL #8\n"
      "add x25, x28, %x[in_stride]\n"
      "cmp %x[height], #0x1\n"
      "add %x[in], x25, %x[in_stride]\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "cmp x21, x20\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x2\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1h { z17.h }, p4/Z, [x28]\n"
      "ld1h { z20.h }, p4/Z, [x28, #1, MUL VL]\n"
      "sub x21, x21, x20\n"
      "cmp x21, x20\n"
      "ld1h { z23.h }, p4/Z, [x28, #2, MUL VL]\n"
      "ld1h { z19.h }, p4/Z, [x28, #3, MUL VL]\n"
      "ld1h { z16.h }, p4/Z, [x25]\n"
      "ld1h { z18.h }, p4/Z, [x25, #1, MUL VL]\n"
      "zip1 z0.h, z17.h, z16.h\n"
      "zip2 z22.h, z17.h, z16.h\n"
      "ld1h { z17.h }, p4/Z, [x25, #2, MUL VL]\n"
      "ld1h { z16.h }, p4/Z, [x25, #3, MUL VL]\n"
      "zip1 z31.h, z20.h, z18.h\n"
      "zip2 z30.h, z20.h, z18.h\n"
      "ld1h { z21.h }, p4/Z, [x28, #4, MUL VL]\n"
      "ld1h { z20.h }, p4/Z, [x28, #5, MUL VL]\n"
      "zip1 z29.h, z23.h, z17.h\n"
      "zip2 z28.h, z23.h, z17.h\n"
      "ld1h { z27.h }, p4/Z, [x28, #6, MUL VL]\n"
      "ld1h { z26.h }, p4/Z, [x28, #7, MUL VL]\n"
      "zip1 z25.h, z19.h, z16.h\n"
      "zip2 z24.h, z19.h, z16.h\n"
      "ld1h { z19.h }, p4/Z, [x25, #4, MUL VL]\n"
      "ld1h { z18.h }, p4/Z, [x25, #5, MUL VL]\n"
      "addvl x28, x28, #8\n"
      "zip1 z23.h, z21.h, z19.h\n"
      "ld1h { z17.h }, p4/Z, [x25, #6, MUL VL]\n"
      "ld1h { z16.h }, p4/Z, [x25, #7, MUL VL]\n"
      "st1h { z0.h }, p4, [x22]\n"
      "addvl x25, x25, #8\n"
      "st1h { z22.h }, p4, [x22, #1, MUL VL]\n"
      "zip2 z22.h, z21.h, z19.h\n"
      "zip1 z21.h, z20.h, z18.h\n"
      "st1h { z31.h }, p4, [x22, #2, MUL VL]\n"
      "zip2 z20.h, z20.h, z18.h\n"
      "zip1 z19.h, z27.h, z17.h\n"
      "st1h { z30.h }, p4, [x22, #3, MUL VL]\n"
      "zip2 z18.h, z27.h, z17.h\n"
      "zip1 z17.h, z26.h, z16.h\n"
      "st1h { z29.h }, p4, [x22, #4, MUL VL]\n"
      "zip2 z16.h, z26.h, z16.h\n"
      "st1h { z28.h }, p4, [x22, #5, MUL VL]\n"
      "st1h { z25.h }, p4, [x22, #6, MUL VL]\n"
      "st1h { z24.h }, p4, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z23.h }, p4, [x22]\n"
      "st1h { z22.h }, p4, [x22, #1, MUL VL]\n"
      "st1h { z21.h }, p4, [x22, #2, MUL VL]\n"
      "st1h { z20.h }, p4, [x22, #3, MUL VL]\n"
      "st1h { z19.h }, p4, [x22, #4, MUL VL]\n"
      "st1h { z18.h }, p4, [x22, #5, MUL VL]\n"
      "st1h { z17.h }, p4, [x22, #6, MUL VL]\n"
      "st1h { z16.h }, p4, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "whilelt p0.h, XZR, x20\n"
      "ld1h { z22.h }, p0/Z, [x28]\n"
      "ld1h { z21.h }, p0/Z, [x25]\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "ld1h { z20.h }, p0/Z, [x28, #1, MUL VL]\n"
      "ld1h { z19.h }, p0/Z, [x25, #1, MUL VL]\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "ld1h { z18.h }, p0/Z, [x28, #2, MUL VL]\n"
      "ld1h { z17.h }, p0/Z, [x25, #2, MUL VL]\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "ld1h { z24.h }, p0/Z, [x28, #3, MUL VL]\n"
      "ld1h { z23.h }, p0/Z, [x25, #3, MUL VL]\n"
      "decw x21, ALL, MUL #8\n"
      "cmp x21, #0x0\n"
      "zip1 z16.h, z22.h, z21.h\n"
      "zip2 z22.h, z22.h, z21.h\n"
      "addvl x28, x28, #4\n"
      "addvl x25, x25, #4\n"
      "zip1 z21.h, z20.h, z19.h\n"
      "zip2 z20.h, z20.h, z19.h\n"
      "zip1 z19.h, z18.h, z17.h\n"
      "zip2 z18.h, z18.h, z17.h\n"
      "st1h { z16.h }, p4, [x22]\n"
      "zip1 z17.h, z24.h, z23.h\n"
      "zip2 z16.h, z24.h, z23.h\n"
      "st1h { z22.h }, p4, [x22, #1, MUL VL]\n"
      "st1h { z21.h }, p4, [x22, #2, MUL VL]\n"
      "st1h { z20.h }, p4, [x22, #3, MUL VL]\n"
      "st1h { z19.h }, p4, [x22, #4, MUL VL]\n"
      "st1h { z18.h }, p4, [x22, #5, MUL VL]\n"
      "st1h { z17.h }, p4, [x22, #6, MUL VL]\n"
      "st1h { z16.h }, p4, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 7b\n"
      "12:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<8, 2, true, VLType::SVE>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_8VL_2x2(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(bfloat16) / 2,
        stride * sizeof(bfloat16),
        (kmax-k0)
    );
}

#endif
