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

void sve_transpose_interleave_1VL(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 1 * height * get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "ptrue p1.b\n"
      "cmp %x[height], #0x4\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[out]\n"
      "add x23, x25, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x4\n"
      "mov x20, %x[width]\n"
      "cntw x19, ALL, MUL #2\n"
      "cmp x20, x19\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z23.s }, p1/Z, [x25]\n"
      "sub x20, x20, x19\n"
      "ld1w { z22.s }, p1/Z, [x25, #1, MUL VL]\n"
      "addvl x25, x25, #2\n"
      "ld1w { z21.s }, p1/Z, [x23]\n"
      "cmp x20, x19\n"
      "ld1w { z20.s }, p1/Z, [x23, #1, MUL VL]\n"
      "addvl x23, x23, #2\n"
      "ld1w { z19.s }, p1/Z, [x22]\n"
      "ld1w { z18.s }, p1/Z, [x22, #1, MUL VL]\n"
      "addvl x22, x22, #2\n"
      "ld1w { z17.s }, p1/Z, [x21]\n"
      "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "st1w { z23.s }, p1, [x24]\n"
      "st1w { z21.s }, p1, [x24, #1, MUL VL]\n"
      "st1w { z19.s }, p1, [x24, #2, MUL VL]\n"
      "st1w { z17.s }, p1, [x24, #3, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "st1w { z22.s }, p1, [x24]\n"
      "st1w { z20.s }, p1, [x24, #1, MUL VL]\n"
      "st1w { z18.s }, p1, [x24, #2, MUL VL]\n"
      "st1w { z16.s }, p1, [x24, #3, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x20, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z19.s }, p0/Z, [x25]\n"
      "addvl x25, x25, #1\n"
      "ld1w { z18.s }, p0/Z, [x23]\n"
      "addvl x23, x23, #1\n"
      "ld1w { z17.s }, p0/Z, [x22]\n"
      "addvl x22, x22, #1\n"
      "ld1w { z16.s }, p0/Z, [x21]\n"
      "addvl x21, x21, #1\n"
      "st1w { z19.s }, p1, [x24]\n"
      "decw x20\n"
      "st1w { z18.s }, p1, [x24, #1, MUL VL]\n"
      "cmp x20, #0x0\n"
      "st1w { z17.s }, p1, [x24, #2, MUL VL]\n"
      "st1w { z16.s }, p1, [x24, #3, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "addvl %x[out], %x[out], #4\n"
      "cmp %x[height], #0x4\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip

      "7:"  // Tail row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[out]\n"
      "add %x[in], x25, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x1\n"
      "mov x20, %x[width]\n"
      "cntw x19, ALL, MUL #2\n"
      "cmp x20, x19\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1w { z17.s }, p1/Z, [x25]\n"
      "sub x20, x20, x19\n"
      "ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]\n"
      "addvl x25, x25, #2\n"
      "cmp x20, x19\n"
      "st1w { z17.s }, p1, [x24]\n"
      "add x24, x24, %x[out_stride]\n"
      "st1w { z16.s }, p1, [x24]\n"
      "add x24, x24, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x20, 11f\n"
      "10:"  // Tail row loop: Column loop
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z16.s }, p0/Z, [x25]\n"
      "addvl x25, x25, #1\n"
      "decw x20\n"
      "st1w { z16.s }, p1, [x24]\n"
      "add x24, x24, %x[out_stride]\n"
      "cmp x20, #0x0\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "addvl %x[out], %x[out], #1\n"
      "cmp %x[height], #0x1\n"
      "bge 7b\n"
      "12:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23"
    );
}

} // anonymous namespace

template<>
void Transform<1, 1, true, VLType::SVE>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_1VL(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
