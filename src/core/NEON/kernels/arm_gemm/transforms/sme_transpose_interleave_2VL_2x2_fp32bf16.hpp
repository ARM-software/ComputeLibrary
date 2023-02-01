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

void sme_transpose_interleave_2VL_2x2_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 2 * roundup<size_t>(height, 2) * sme::get_vector_length<uint16_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cmp %x[height], #0x4\n"
      "ptrue p2.b\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "mov x23, %x[width]\n"
      "cnth x20, ALL, MUL #2\n"
      "add x21, x24, %x[in_stride]\n"
      "cmp x23, x20\n"
      "add %x[in], x21, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z16.s }, p2/Z, [x26]\n"
      ".inst 0x658aaa18  // bfcvt z24.h, p2/M, z16.s\n"
      "sub x23, x23, x20\n"
      "cmp x23, x20\n"
      "ld1w { z16.s }, p2/Z, [x26, #1, MUL VL]\n"
      ".inst 0x658aaa17  // bfcvt z23.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x24]\n"
      ".inst 0x658aaa16  // bfcvt z22.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x24, #1, MUL VL]\n"
      ".inst 0x658aaa15  // bfcvt z21.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x26, #2, MUL VL]\n"
      ".inst 0x658aaa14  // bfcvt z20.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x26, #3, MUL VL]\n"
      ".inst 0x658aaa13  // bfcvt z19.h, p2/M, z16.s\n"
      "addvl x26, x26, #4\n"
      "ld1w { z16.s }, p2/Z, [x24, #2, MUL VL]\n"
      ".inst 0x658aaa12  // bfcvt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x24, #3, MUL VL]\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "addvl x24, x24, #4\n"
      "ld1w { z16.s }, p2/Z, [x25]\n"
      ".inst 0x648aaa18  // bfcvtnt z24.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #1, MUL VL]\n"
      ".inst 0x648aaa17  // bfcvtnt z23.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x21]\n"
      ".inst 0x648aaa16  // bfcvtnt z22.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x21, #1, MUL VL]\n"
      ".inst 0x648aaa15  // bfcvtnt z21.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #2, MUL VL]\n"
      ".inst 0x648aaa14  // bfcvtnt z20.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      ".inst 0x648aaa13  // bfcvtnt z19.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x21, #2, MUL VL]\n"
      ".inst 0x648aaa12  // bfcvtnt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x21, #3, MUL VL]\n"
      "st1h { z24.h }, p2, [x22]\n"
      "addvl x21, x21, #4\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z23.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z22.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z21.h }, p2, [x22, #3, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z20.h }, p2, [x22]\n"
      "st1h { z19.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z18.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z17.h }, p2, [x22, #3, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x23, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x23\n"
      "whilelt p1.s, XZR, x20\n"
      "ld1w { z16.s }, p1/Z, [x26]\n"
      ".inst 0x658aaa14  // bfcvt z20.h, p2/M, z16.s\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z16.s }, p0/Z, [x26, #1, MUL VL]\n"
      ".inst 0x658aaa13  // bfcvt z19.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x24]\n"
      ".inst 0x658aaa12  // bfcvt z18.h, p2/M, z16.s\n"
      "decw x23, ALL, MUL #2\n"
      "cmp x23, #0x0\n"
      "ld1w { z16.s }, p0/Z, [x24, #1, MUL VL]\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "addvl x26, x26, #2\n"
      "addvl x24, x24, #2\n"
      "ld1w { z16.s }, p1/Z, [x25]\n"
      ".inst 0x648aaa14  // bfcvtnt z20.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x25, #1, MUL VL]\n"
      "addvl x25, x25, #2\n"
      ".inst 0x648aaa13  // bfcvtnt z19.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x21]\n"
      ".inst 0x648aaa12  // bfcvtnt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z20.h }, p2, [x22]\n"
      "st1h { z19.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z18.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z17.h }, p2, [x22, #3, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x4\n"
      "addvl %x[out], %x[out], #4\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x26, %x[in]\n"
      "add x25, x26, %x[in_stride]\n"
      "cmp %x[height], #0x1\n"
      "mov x21, %x[width]\n"
      "cnth x20, ALL, MUL #2\n"
      "add %x[in], x25, %x[in_stride]\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "cmp x21, x20\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x2\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1w { z16.s }, p2/Z, [x26]\n"
      ".inst 0x658aaa14  // bfcvt z20.h, p2/M, z16.s\n"
      "sub x21, x21, x20\n"
      "cmp x21, x20\n"
      "ld1w { z16.s }, p2/Z, [x26, #1, MUL VL]\n"
      ".inst 0x658aaa13  // bfcvt z19.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x26, #2, MUL VL]\n"
      ".inst 0x658aaa12  // bfcvt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x26, #3, MUL VL]\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "addvl x26, x26, #4\n"
      "ld1w { z16.s }, p2/Z, [x25]\n"
      ".inst 0x648aaa14  // bfcvtnt z20.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #1, MUL VL]\n"
      ".inst 0x648aaa13  // bfcvtnt z19.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #2, MUL VL]\n"
      ".inst 0x648aaa12  // bfcvtnt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #3, MUL VL]\n"
      "st1h { z20.h }, p2, [x22]\n"
      "addvl x25, x25, #4\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z19.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z18.h }, p2, [x22]\n"
      "st1h { z17.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "whilelt p1.s, XZR, x20\n"
      "ld1w { z16.s }, p1/Z, [x26]\n"
      ".inst 0x658aaa12  // bfcvt z18.h, p2/M, z16.s\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z16.s }, p0/Z, [x26, #1, MUL VL]\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x25]\n"
      "decw x21, ALL, MUL #2\n"
      "cmp x21, #0x0\n"
      ".inst 0x648aaa12  // bfcvtnt z18.h, p2/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x25, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "addvl x25, x25, #2\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z18.h }, p2, [x22]\n"
      "st1h { z17.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #2\n"
      "bge 7b\n"
      "12:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace
template<>
void Transform<2, 2, true, VLType::SME>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_2VL_2x2_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
