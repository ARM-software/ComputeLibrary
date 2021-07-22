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

void sve_transpose_interleave_1VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 1 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p1.b\n"
      "cmp %x[height], #0x8\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x9, %x[in]\n"
      "mov x28, %x[out]\n"
      "add x27, x9, %x[in_stride]\n"
      "add x26, x27, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x8\n"
      "mov x20, %x[width]\n"
      "cntb x19, ALL, MUL #2\n"
      "cmp x20, x19\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1b { z17.b }, p1/Z, [x9]\n"
      "sub x20, x20, x19\n"
      "ld1b { z3.b }, p1/Z, [x9, #1, MUL VL]\n"
      "addvl x9, x9, #2\n"
      "ld1b { z21.b }, p1/Z, [x27]\n"
      "cmp x20, x19\n"
      "ld1b { z2.b }, p1/Z, [x27, #1, MUL VL]\n"
      "addvl x27, x27, #2\n"
      "ld1b { z16.b }, p1/Z, [x26]\n"
      "zip1 z20.b, z17.b, z16.b\n"
      "ld1b { z1.b }, p1/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "zip2 z19.b, z17.b, z16.b\n"
      "ld1b { z17.b }, p1/Z, [x25]\n"
      "ld1b { z0.b }, p1/Z, [x25, #1, MUL VL]\n"
      "zip1 z31.b, z3.b, z1.b\n"
      "ld1b { z30.b }, p1/Z, [x24]\n"
      "addvl x25, x25, #2\n"
      "zip1 z16.b, z21.b, z17.b\n"
      "ld1b { z29.b }, p1/Z, [x24, #1, MUL VL]\n"
      "addvl x24, x24, #2\n"
      "zip1 z18.b, z20.b, z16.b\n"
      "ld1b { z28.b }, p1/Z, [x23]\n"
      "zip2 z27.b, z20.b, z16.b\n"
      "ld1b { z26.b }, p1/Z, [x23, #1, MUL VL]\n"
      "addvl x23, x23, #2\n"
      "zip2 z17.b, z21.b, z17.b\n"
      "ld1b { z16.b }, p1/Z, [x22]\n"
      "zip1 z25.b, z2.b, z0.b\n"
      "ld1b { z24.b }, p1/Z, [x22, #1, MUL VL]\n"
      "addvl x22, x22, #2\n"
      "zip1 z23.b, z19.b, z17.b\n"
      "ld1b { z22.b }, p1/Z, [x21]\n"
      "zip2 z20.b, z19.b, z17.b\n"
      "ld1b { z21.b }, p1/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "zip1 z19.b, z30.b, z16.b\n"
      "st1b { z18.b }, p1, [x28]\n"
      "zip2 z18.b, z30.b, z16.b\n"
      "zip1 z17.b, z28.b, z22.b\n"
      "zip1 z16.b, z19.b, z17.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z16.b, z19.b, z17.b\n"
      "st1b { z27.b }, p1, [x28]\n"
      "zip2 z17.b, z28.b, z22.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z23.b }, p1, [x28]\n"
      "zip2 z17.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z16.b, z31.b, z25.b\n"
      "st1b { z20.b }, p1, [x28]\n"
      "zip1 z19.b, z29.b, z24.b\n"
      "st1b { z17.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z18.b, z26.b, z21.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z17.b, z31.b, z25.b\n"
      "zip1 z16.b, z19.b, z18.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z16.b, z19.b, z18.b\n"
      "st1b { z17.b }, p1, [x28]\n"
      "zip2 z20.b, z3.b, z1.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z19.b, z2.b, z0.b\n"
      "zip2 z18.b, z29.b, z24.b\n"
      "zip1 z16.b, z20.b, z19.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z17.b, z26.b, z21.b\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z16.b, z20.b, z19.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "add x28, x28, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x20, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.b, XZR, x20\n"
      "ld1b { z18.b }, p0/Z, [x9]\n"
      "incd x9, ALL, MUL #2\n"
      "ld1b { z17.b }, p0/Z, [x27]\n"
      "incd x27, ALL, MUL #2\n"
      "ld1b { z16.b }, p0/Z, [x26]\n"
      "zip1 z18.b, z18.b, z16.b\n"
      "ld1b { z16.b }, p0/Z, [x25]\n"
      "incd x26, ALL, MUL #2\n"
      "zip1 z16.b, z17.b, z16.b\n"
      "ld1b { z17.b }, p0/Z, [x24]\n"
      "incd x25, ALL, MUL #2\n"
      "zip1 z19.b, z18.b, z16.b\n"
      "ld1b { z18.b }, p0/Z, [x23]\n"
      "incd x24, ALL, MUL #2\n"
      "ld1b { z16.b }, p0/Z, [x22]\n"
      "zip1 z17.b, z17.b, z16.b\n"
      "ld1b { z16.b }, p0/Z, [x21]\n"
      "incd x23, ALL, MUL #2\n"
      "zip1 z16.b, z18.b, z16.b\n"
      "st1b { z19.b }, p1, [x28]\n"
      "incd x22, ALL, MUL #2\n"
      "zip1 z16.b, z17.b, z16.b\n"
      "st1b { z16.b }, p1, [x28, #1, MUL VL]\n"
      "incd x21, ALL, MUL #2\n"
      "add x28, x28, %x[out_stride]\n"
      "decw x20\n"
      "cmp x20, #0x0\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "addvl %x[out], %x[out], #2\n"
      "cmp %x[height], #0x8\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip

      "7:"  // Tail row loop: Head
      "mov x9, %x[in]\n"
      "mov x28, %x[out]\n"
      "add x27, x9, %x[in_stride]\n"
      "add x26, x27, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add %x[in], x25, %x[in_stride]\n"
      "cmp %x[height], #0x3\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "csel x26, x26, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x27, x27, %x[pad_row], GT\n"
      "sub %x[height], %x[height], #0x4\n"
      "mov x20, %x[width]\n"
      "cntb x19, ALL, MUL #2\n"
      "cmp x20, x19\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1b { z19.b }, p1/Z, [x9]\n"
      "sub x20, x20, x19\n"
      "ld1b { z18.b }, p1/Z, [x9, #1, MUL VL]\n"
      "addvl x9, x9, #2\n"
      "ld1b { z25.b }, p1/Z, [x27]\n"
      "cmp x20, x19\n"
      "ld1b { z24.b }, p1/Z, [x27, #1, MUL VL]\n"
      "addvl x27, x27, #2\n"
      "ld1b { z17.b }, p1/Z, [x26]\n"
      "zip1 z23.b, z19.b, z17.b\n"
      "ld1b { z16.b }, p1/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "zip2 z22.b, z19.b, z17.b\n"
      "ld1b { z21.b }, p1/Z, [x25]\n"
      "ld1b { z20.b }, p1/Z, [x25, #1, MUL VL]\n"
      "zip1 z19.b, z18.b, z16.b\n"
      "addvl x25, x25, #2\n"
      "zip2 z18.b, z18.b, z16.b\n"
      "zip1 z17.b, z25.b, z21.b\n"
      "zip1 z16.b, z23.b, z17.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z16.b, z23.b, z17.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z17.b, z25.b, z21.b\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z16.b, z22.b, z17.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z16.b, z22.b, z17.b\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z17.b, z24.b, z20.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z16.b, z19.b, z17.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z16.b, z19.b, z17.b\n"
      "add x28, x28, %x[out_stride]\n"
      "zip2 z17.b, z24.b, z20.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "add x28, x28, %x[out_stride]\n"
      "zip1 z16.b, z18.b, z17.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "zip2 z16.b, z18.b, z17.b\n"
      "add x28, x28, %x[out_stride]\n"
      "st1b { z16.b }, p1, [x28]\n"
      "add x28, x28, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x20, 11f\n"
      "10:"  // Tail row loop: Column loop
      "whilelt p0.b, XZR, x20\n"
      "ld1b { z17.b }, p0/Z, [x9]\n"
      "incd x9, ALL, MUL #2\n"
      "ld1b { z18.b }, p0/Z, [x27]\n"
      "incd x27, ALL, MUL #2\n"
      "ld1b { z16.b }, p0/Z, [x26]\n"
      "zip1 z17.b, z17.b, z16.b\n"
      "ld1b { z16.b }, p0/Z, [x25]\n"
      "incd x26, ALL, MUL #2\n"
      "zip1 z16.b, z18.b, z16.b\n"
      "incd x25, ALL, MUL #2\n"
      "decw x20\n"
      "zip1 z16.b, z17.b, z16.b\n"
      "st1b { z16.b }, p1, [x28]\n"
      "add x28, x28, %x[out_stride]\n"
      "cmp x20, #0x0\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "addvl %x[out], %x[out], #1\n"
      "cmp %x[height], #0x1\n"
      "bge 7b\n"
      "12:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x9", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<1, 4, true, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_1VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<1, 4, true, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_1VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif
