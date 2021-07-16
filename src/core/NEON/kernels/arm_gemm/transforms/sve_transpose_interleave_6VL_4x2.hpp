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

void sve_transpose_interleave_6VL_4x2(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    uint32_t *pad_row = reinterpret_cast<uint32_t *>(alloca(width * sizeof(uint32_t)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(uint32_t));
    }

    size_t out_stride = 6 * roundup<size_t>(height, 2) * get_vector_length<uint16_t>();

    __asm__ __volatile__(
      "ptrue p2.b\n"
      "cmp %x[height], #0x4\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x27, %x[in]\n"
      "mov x26, %x[out]\n"
      "add x25, x27, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add %x[in], x23, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x4\n"
      "mov x22, %x[width]\n"
      "cntw x21, ALL, MUL #6\n"
      "cmp x22, x21\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z19.s }, p2/Z, [x27]\n"
      "mov x20, x26\n"
      "ld1w { z18.s }, p2/Z, [x27, #1, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "ld1w { z21.s }, p2/Z, [x27, #2, MUL VL]\n"
      "mov x19, x26\n"
      "ld1w { z26.s }, p2/Z, [x27, #3, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "ld1w { z25.s }, p2/Z, [x27, #4, MUL VL]\n"
      "sub x22, x22, x21\n"
      "ld1w { z24.s }, p2/Z, [x27, #5, MUL VL]\n"
      "addvl x27, x27, #6\n"
      "ld1w { z16.s }, p2/Z, [x25]\n"
      "zip1 z23.s, z19.s, z16.s\n"
      "ld1w { z17.s }, p2/Z, [x25, #1, MUL VL]\n"
      "cmp x22, x21\n"
      "zip2 z9.s, z19.s, z16.s\n"
      "ld1w { z20.s }, p2/Z, [x25, #2, MUL VL]\n"
      "ld1w { z19.s }, p2/Z, [x25, #3, MUL VL]\n"
      "zip1 z8.s, z18.s, z17.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #4, MUL VL]\n"
      "zip2 z7.s, z18.s, z17.s\n"
      "ld1w { z18.s }, p2/Z, [x25, #5, MUL VL]\n"
      "addvl x25, x25, #6\n"
      "zip1 z6.s, z21.s, z20.s\n"
      "ld1w { z17.s }, p2/Z, [x24]\n"
      "zip2 z5.s, z21.s, z20.s\n"
      "ld1w { z22.s }, p2/Z, [x24, #1, MUL VL]\n"
      "zip1 z4.s, z26.s, z19.s\n"
      "ld1w { z21.s }, p2/Z, [x24, #2, MUL VL]\n"
      "zip2 z3.s, z26.s, z19.s\n"
      "ld1w { z2.s }, p2/Z, [x24, #3, MUL VL]\n"
      "zip1 z1.s, z25.s, z16.s\n"
      "ld1w { z0.s }, p2/Z, [x24, #4, MUL VL]\n"
      "zip2 z31.s, z25.s, z16.s\n"
      "ld1w { z30.s }, p2/Z, [x24, #5, MUL VL]\n"
      "addvl x24, x24, #6\n"
      "zip1 z29.s, z24.s, z18.s\n"
      "ld1w { z16.s }, p2/Z, [x23]\n"
      "zip2 z28.s, z24.s, z18.s\n"
      "ld1w { z20.s }, p2/Z, [x23, #1, MUL VL]\n"
      "ld1w { z19.s }, p2/Z, [x23, #2, MUL VL]\n"
      "zip1 z27.s, z17.s, z16.s\n"
      "ld1w { z18.s }, p2/Z, [x23, #3, MUL VL]\n"
      "zip2 z26.s, z17.s, z16.s\n"
      "ld1w { z17.s }, p2/Z, [x23, #4, MUL VL]\n"
      "zip1 z25.s, z22.s, z20.s\n"
      "ld1w { z16.s }, p2/Z, [x23, #5, MUL VL]\n"
      "addvl x23, x23, #6\n"
      "zip2 z24.s, z22.s, z20.s\n"
      "st1w { z23.s }, p2, [x20]\n"
      "zip1 z23.s, z21.s, z19.s\n"
      "st1w { z9.s }, p2, [x20, #1, MUL VL]\n"
      "zip2 z22.s, z21.s, z19.s\n"
      "st1w { z8.s }, p2, [x20, #2, MUL VL]\n"
      "zip1 z21.s, z2.s, z18.s\n"
      "st1w { z7.s }, p2, [x20, #3, MUL VL]\n"
      "zip2 z20.s, z2.s, z18.s\n"
      "st1w { z6.s }, p2, [x20, #4, MUL VL]\n"
      "zip1 z19.s, z0.s, z17.s\n"
      "st1w { z5.s }, p2, [x20, #5, MUL VL]\n"
      "zip2 z18.s, z0.s, z17.s\n"
      "st1w { z27.s }, p2, [x20, #6, MUL VL]\n"
      "zip1 z17.s, z30.s, z16.s\n"
      "st1w { z26.s }, p2, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #12\n"
      "zip2 z16.s, z30.s, z16.s\n"
      "st1w { z25.s }, p2, [x20, #-4, MUL VL]\n"
      "st1w { z24.s }, p2, [x20, #-3, MUL VL]\n"
      "st1w { z23.s }, p2, [x20, #-2, MUL VL]\n"
      "st1w { z22.s }, p2, [x20, #-1, MUL VL]\n"
      "st1w { z4.s }, p2, [x19]\n"
      "st1w { z3.s }, p2, [x19, #1, MUL VL]\n"
      "st1w { z1.s }, p2, [x19, #2, MUL VL]\n"
      "st1w { z31.s }, p2, [x19, #3, MUL VL]\n"
      "st1w { z29.s }, p2, [x19, #4, MUL VL]\n"
      "st1w { z28.s }, p2, [x19, #5, MUL VL]\n"
      "st1w { z21.s }, p2, [x19, #6, MUL VL]\n"
      "st1w { z20.s }, p2, [x19, #7, MUL VL]\n"
      "addvl x19, x19, #12\n"
      "st1w { z19.s }, p2, [x19, #-4, MUL VL]\n"
      "st1w { z18.s }, p2, [x19, #-3, MUL VL]\n"
      "st1w { z17.s }, p2, [x19, #-2, MUL VL]\n"
      "st1w { z16.s }, p2, [x19, #-1, MUL VL]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x22, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x22\n"
      "mov x19, x26\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z18.s }, p0/Z, [x27]\n"
      "ld1w { z16.s }, p0/Z, [x25]\n"
      "zip1 z28.s, z18.s, z16.s\n"
      "ld1w { z17.s }, p0/Z, [x24]\n"
      "decw x20\n"
      "zip2 z27.s, z18.s, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x23]\n"
      "whilelt p1.s, XZR, x20\n"
      "zip1 z26.s, z17.s, z16.s\n"
      "ld1w { z18.s }, p1/Z, [x27, #1, MUL VL]\n"
      "decw x20\n"
      "zip2 z25.s, z17.s, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]\n"
      "whilelt p0.s, XZR, x20\n"
      "zip1 z24.s, z18.s, z16.s\n"
      "ld1w { z17.s }, p0/Z, [x27, #2, MUL VL]\n"
      "addvl x27, x27, #3\n"
      "zip2 z23.s, z18.s, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]\n"
      "addvl x25, x25, #3\n"
      "zip1 z22.s, z17.s, z16.s\n"
      "ld1w { z18.s }, p1/Z, [x24, #1, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "zip2 z21.s, z17.s, z16.s\n"
      "ld1w { z20.s }, p0/Z, [x24, #2, MUL VL]\n"
      "addvl x24, x24, #3\n"
      "ld1w { z17.s }, p1/Z, [x23, #1, MUL VL]\n"
      "zip1 z19.s, z18.s, z17.s\n"
      "ld1w { z16.s }, p0/Z, [x23, #2, MUL VL]\n"
      "addvl x23, x23, #3\n"
      "zip2 z18.s, z18.s, z17.s\n"
      "st1w { z28.s }, p2, [x19]\n"
      "decd x22, ALL, MUL #6\n"
      "zip1 z17.s, z20.s, z16.s\n"
      "st1w { z27.s }, p2, [x19, #1, MUL VL]\n"
      "cmp x22, #0x0\n"
      "zip2 z16.s, z20.s, z16.s\n"
      "st1w { z24.s }, p2, [x19, #2, MUL VL]\n"
      "st1w { z23.s }, p2, [x19, #3, MUL VL]\n"
      "st1w { z22.s }, p2, [x19, #4, MUL VL]\n"
      "st1w { z21.s }, p2, [x19, #5, MUL VL]\n"
      "st1w { z26.s }, p2, [x19, #6, MUL VL]\n"
      "st1w { z25.s }, p2, [x19, #7, MUL VL]\n"
      "addvl x19, x19, #12\n"
      "st1w { z19.s }, p2, [x19, #-4, MUL VL]\n"
      "st1w { z18.s }, p2, [x19, #-3, MUL VL]\n"
      "st1w { z17.s }, p2, [x19, #-2, MUL VL]\n"
      "st1w { z16.s }, p2, [x19, #-1, MUL VL]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "addvl %x[out], %x[out], #12\n"
      "cmp %x[height], #0x4\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip

      "7:"  // Tail row loop: Head
      "mov x27, %x[in]\n"
      "mov x26, %x[out]\n"
      "add x25, x27, %x[in_stride]\n"
      "add %x[in], x25, %x[in_stride]\n"
      "cmp %x[height], #0x1\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "sub %x[height], %x[height], #0x2\n"
      "mov x20, %x[width]\n"
      "cntw x19, ALL, MUL #6\n"
      "cmp x20, x19\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1w { z19.s }, p2/Z, [x27]\n"
      "sub x20, x20, x19\n"
      "ld1w { z18.s }, p2/Z, [x27, #1, MUL VL]\n"
      "cmp x20, x19\n"
      "ld1w { z29.s }, p2/Z, [x27, #2, MUL VL]\n"
      "ld1w { z28.s }, p2/Z, [x27, #3, MUL VL]\n"
      "ld1w { z27.s }, p2/Z, [x27, #4, MUL VL]\n"
      "ld1w { z26.s }, p2/Z, [x27, #5, MUL VL]\n"
      "addvl x27, x27, #6\n"
      "ld1w { z16.s }, p2/Z, [x25]\n"
      "zip1 z25.s, z19.s, z16.s\n"
      "ld1w { z17.s }, p2/Z, [x25, #1, MUL VL]\n"
      "zip2 z24.s, z19.s, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #2, MUL VL]\n"
      "ld1w { z23.s }, p2/Z, [x25, #3, MUL VL]\n"
      "zip1 z20.s, z18.s, z17.s\n"
      "ld1w { z22.s }, p2/Z, [x25, #4, MUL VL]\n"
      "zip2 z19.s, z18.s, z17.s\n"
      "ld1w { z21.s }, p2/Z, [x25, #5, MUL VL]\n"
      "addvl x25, x25, #6\n"
      "zip1 z18.s, z29.s, z16.s\n"
      "st1w { z25.s }, p2, [x26]\n"
      "zip2 z17.s, z29.s, z16.s\n"
      "st1w { z24.s }, p2, [x26, #1, MUL VL]\n"
      "zip1 z16.s, z28.s, z23.s\n"
      "st1w { z20.s }, p2, [x26, #2, MUL VL]\n"
      "zip2 z20.s, z28.s, z23.s\n"
      "st1w { z19.s }, p2, [x26, #3, MUL VL]\n"
      "zip1 z19.s, z27.s, z22.s\n"
      "st1w { z18.s }, p2, [x26, #4, MUL VL]\n"
      "zip2 z18.s, z27.s, z22.s\n"
      "st1w { z17.s }, p2, [x26, #5, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "zip1 z17.s, z26.s, z21.s\n"
      "st1w { z16.s }, p2, [x26]\n"
      "zip2 z16.s, z26.s, z21.s\n"
      "st1w { z20.s }, p2, [x26, #1, MUL VL]\n"
      "st1w { z19.s }, p2, [x26, #2, MUL VL]\n"
      "st1w { z18.s }, p2, [x26, #3, MUL VL]\n"
      "st1w { z17.s }, p2, [x26, #4, MUL VL]\n"
      "st1w { z16.s }, p2, [x26, #5, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x20, 11f\n"
      "10:"  // Tail row loop: Column loop
      "mov x19, x20\n"
      "decd x20, ALL, MUL #6\n"
      "whilelt p0.s, XZR, x19\n"
      "ld1w { z17.s }, p0/Z, [x27]\n"
      "ld1w { z16.s }, p0/Z, [x25]\n"
      "zip1 z22.s, z17.s, z16.s\n"
      "decw x19\n"
      "zip2 z21.s, z17.s, z16.s\n"
      "whilelt p0.s, XZR, x19\n"
      "ld1w { z17.s }, p0/Z, [x27, #1, MUL VL]\n"
      "decw x19\n"
      "ld1w { z16.s }, p0/Z, [x25, #1, MUL VL]\n"
      "zip1 z20.s, z17.s, z16.s\n"
      "whilelt p0.s, XZR, x19\n"
      "ld1w { z19.s }, p0/Z, [x27, #2, MUL VL]\n"
      "zip2 z18.s, z17.s, z16.s\n"
      "addvl x27, x27, #3\n"
      "ld1w { z16.s }, p0/Z, [x25, #2, MUL VL]\n"
      "zip1 z17.s, z19.s, z16.s\n"
      "st1w { z22.s }, p2, [x26]\n"
      "addvl x25, x25, #3\n"
      "zip2 z16.s, z19.s, z16.s\n"
      "st1w { z21.s }, p2, [x26, #1, MUL VL]\n"
      "cmp x20, #0x0\n"
      "st1w { z20.s }, p2, [x26, #2, MUL VL]\n"
      "st1w { z18.s }, p2, [x26, #3, MUL VL]\n"
      "st1w { z17.s }, p2, [x26, #4, MUL VL]\n"
      "st1w { z16.s }, p2, [x26, #5, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "addvl %x[out], %x[out], #6\n"
      "cmp %x[height], #0x1\n"
      "bge 7b\n"
      "12:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<6, 2, true, VLType::SVE>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_6VL_4x2(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
