/*
 * Copyright (c) 2022-2023 Arm Limited.
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

void sme_transpose_interleave_1VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 1 * roundup<size_t>(height, 4) * sme::get_vector_length<uint32_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p1.b\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "cmp %x[height], #0x3\n"
      "add %x[in], x23, %x[in_stride]\n"
      "csel x23, x23, %x[pad_row], GT\n"
      "csel x24, x24, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "mov x22, %x[width]\n"
      "cntb x21\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "cmp x22, x21\n"
      "mov x20, %x[out]\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1b { z17.b }, p1/Z, [x26]\n"
      "sub x22, x22, x21\n"
      "cmp x22, x21\n"
      "ld1b { z18.b }, p1/Z, [x25]\n"
      "addvl x26, x26, #1\n"
      "addvl x25, x25, #1\n"
      "ld1b { z16.b }, p1/Z, [x24]\n"
      "zip1 z20.b, z17.b, z16.b\n"
      "zip2 z19.b, z17.b, z16.b\n"
      "addvl x24, x24, #1\n"
      "ld1b { z16.b }, p1/Z, [x23]\n"
      "zip1 z17.b, z18.b, z16.b\n"
      "zip2 z18.b, z18.b, z16.b\n"
      "addvl x23, x23, #1\n"
      "zip1 z16.b, z20.b, z17.b\n"
      "st1b { z16.b }, p1, [x20]\n"
      "add x20, x20, %x[out_stride]\n"
      "zip2 z16.b, z20.b, z17.b\n"
      "st1b { z16.b }, p1, [x20]\n"
      "add x20, x20, %x[out_stride]\n"
      "zip1 z17.b, z19.b, z18.b\n"
      "zip2 z16.b, z19.b, z18.b\n"
      "st1b { z17.b }, p1, [x20]\n"
      "add x20, x20, %x[out_stride]\n"
      "st1b { z16.b }, p1, [x20]\n"
      "add x20, x20, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x22, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.b, XZR, x22\n"
      "ld1b { z17.b }, p0/Z, [x26]\n"
      "decw x22\n"
      "ld1b { z18.b }, p0/Z, [x25]\n"
      "cmp x22, #0x0\n"
      "incd x26, ALL, MUL #2\n"
      "ld1b { z16.b }, p0/Z, [x24]\n"
      "zip1 z17.b, z17.b, z16.b\n"
      "incd x25, ALL, MUL #2\n"
      "incd x24, ALL, MUL #2\n"
      "ld1b { z16.b }, p0/Z, [x23]\n"
      "zip1 z16.b, z18.b, z16.b\n"
      "incd x23, ALL, MUL #2\n"
      "zip1 z16.b, z17.b, z16.b\n"
      "st1b { z16.b }, p1, [x20]\n"
      "add x20, x20, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #1\n"
      "bge 1b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<1, 4, true, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_1VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<1, 4, true, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_1VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif
