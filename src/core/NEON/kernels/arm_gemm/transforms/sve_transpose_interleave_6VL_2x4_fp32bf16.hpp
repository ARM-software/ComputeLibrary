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

void sve_transpose_interleave_6VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 6 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p3.b\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "mov x23, %x[width]\n"
      "cnth x20, ALL, MUL #3\n"
      "add x22, x24, %x[in_stride]\n"
      "cmp %x[height], #0x3\n"
      "add %x[in], x22, %x[in_stride]\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel x24, x24, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x25, x25, %x[pad_row], GT\n"
      "cmp x23, x20\n"
      "mov x21, %x[out]\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z17.s }, p3/Z, [x26]\n"
      "ld1w { z18.s }, p3/Z, [x26, #1, MUL VL]\n"
      "sub x23, x23, x20\n"
      "cmp x23, x20\n"
      "ld1w { z19.s }, p3/Z, [x26, #2, MUL VL]\n"
      "ld1w { z16.s }, p3/Z, [x24]\n"
      "zip1 z21.s, z17.s, z16.s\n"
      "zip2 z20.s, z17.s, z16.s\n"
      "ld1w { z17.s }, p3/Z, [x24, #1, MUL VL]\n"
      "ld1w { z16.s }, p3/Z, [x24, #2, MUL VL]\n"
      "zip1 z29.s, z18.s, z17.s\n"
      "zip2 z28.s, z18.s, z17.s\n"
      "ld1w { z17.s }, p3/Z, [x26, #3, MUL VL]\n"
      "ld1w { z18.s }, p3/Z, [x26, #4, MUL VL]\n"
      "zip1 z27.s, z19.s, z16.s\n"
      "zip2 z26.s, z19.s, z16.s\n"
      "ld1w { z19.s }, p3/Z, [x26, #5, MUL VL]\n"
      "ld1w { z16.s }, p3/Z, [x24, #3, MUL VL]\n"
      "zip1 z25.s, z17.s, z16.s\n"
      "zip2 z24.s, z17.s, z16.s\n"
      "ld1w { z17.s }, p3/Z, [x24, #4, MUL VL]\n"
      "ld1w { z16.s }, p3/Z, [x24, #5, MUL VL]\n"
      "zip1 z12.s, z18.s, z17.s\n"
      "zip2 z11.s, z18.s, z17.s\n"
      "ld1w { z18.s }, p3/Z, [x25]\n"
      "ld1w { z23.s }, p3/Z, [x25, #1, MUL VL]\n"
      "zip1 z10.s, z19.s, z16.s\n"
      "zip2 z9.s, z19.s, z16.s\n"
      "ld1w { z22.s }, p3/Z, [x25, #2, MUL VL]\n"
      "ld1w { z17.s }, p3/Z, [x22]\n"
      ".inst 0x658aaea8  // bfcvt z8.h, p3/M, z21.s\n"
      "zip1 z7.s, z18.s, z17.s\n"
      "ld1w { z16.s }, p3/Z, [x22, #1, MUL VL]\n"
      "ld1w { z21.s }, p3/Z, [x22, #2, MUL VL]\n"
      ".inst 0x658aae86  // bfcvt z6.h, p3/M, z20.s\n"
      "zip2 z5.s, z18.s, z17.s\n"
      "ld1w { z20.s }, p3/Z, [x25, #3, MUL VL]\n"
      "ld1w { z19.s }, p3/Z, [x25, #4, MUL VL]\n"
      ".inst 0x658aafa4  // bfcvt z4.h, p3/M, z29.s\n"
      "zip1 z3.s, z23.s, z16.s\n"
      "ld1w { z2.s }, p3/Z, [x25, #5, MUL VL]\n"
      "ld1w { z18.s }, p3/Z, [x22, #3, MUL VL]\n"
      ".inst 0x658aaf81  // bfcvt z1.h, p3/M, z28.s\n"
      "zip2 z0.s, z23.s, z16.s\n"
      "ld1w { z17.s }, p3/Z, [x22, #4, MUL VL]\n"
      "ld1w { z16.s }, p3/Z, [x22, #5, MUL VL]\n"
      ".inst 0x658aaf7f  // bfcvt z31.h, p3/M, z27.s\n"
      "zip1 z30.s, z22.s, z21.s\n"
      ".inst 0x658aaf5d  // bfcvt z29.h, p3/M, z26.s\n"
      "zip2 z28.s, z22.s, z21.s\n"
      "addvl x26, x26, #6\n"
      "addvl x25, x25, #6\n"
      ".inst 0x658aaf3b  // bfcvt z27.h, p3/M, z25.s\n"
      "zip1 z26.s, z20.s, z18.s\n"
      "addvl x24, x24, #6\n"
      "addvl x22, x22, #6\n"
      ".inst 0x658aaf19  // bfcvt z25.h, p3/M, z24.s\n"
      "zip2 z24.s, z20.s, z18.s\n"
      ".inst 0x658aad97  // bfcvt z23.h, p3/M, z12.s\n"
      "zip1 z22.s, z19.s, z17.s\n"
      ".inst 0x658aad75  // bfcvt z21.h, p3/M, z11.s\n"
      "zip2 z20.s, z19.s, z17.s\n"
      ".inst 0x658aad53  // bfcvt z19.h, p3/M, z10.s\n"
      "zip1 z18.s, z2.s, z16.s\n"
      ".inst 0x658aad31  // bfcvt z17.h, p3/M, z9.s\n"
      "zip2 z16.s, z2.s, z16.s\n"
      ".inst 0x648aace8  // bfcvtnt z8.h, p3/M, z7.s\n"
      ".inst 0x648aaca6  // bfcvtnt z6.h, p3/M, z5.s\n"
      "st1h { z8.h }, p3, [x21]\n"
      ".inst 0x648aac64  // bfcvtnt z4.h, p3/M, z3.s\n"
      ".inst 0x648aac01  // bfcvtnt z1.h, p3/M, z0.s\n"
      "st1h { z6.h }, p3, [x21, #1, MUL VL]\n"
      ".inst 0x648aafdf  // bfcvtnt z31.h, p3/M, z30.s\n"
      ".inst 0x648aaf9d  // bfcvtnt z29.h, p3/M, z28.s\n"
      "st1h { z4.h }, p3, [x21, #2, MUL VL]\n"
      "st1h { z1.h }, p3, [x21, #3, MUL VL]\n"
      ".inst 0x648aaf5b  // bfcvtnt z27.h, p3/M, z26.s\n"
      ".inst 0x648aaf19  // bfcvtnt z25.h, p3/M, z24.s\n"
      "st1h { z31.h }, p3, [x21, #4, MUL VL]\n"
      ".inst 0x648aaed7  // bfcvtnt z23.h, p3/M, z22.s\n"
      ".inst 0x648aae95  // bfcvtnt z21.h, p3/M, z20.s\n"
      "st1h { z29.h }, p3, [x21, #5, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      ".inst 0x648aae53  // bfcvtnt z19.h, p3/M, z18.s\n"
      ".inst 0x648aae11  // bfcvtnt z17.h, p3/M, z16.s\n"
      "st1h { z27.h }, p3, [x21]\n"
      "st1h { z25.h }, p3, [x21, #1, MUL VL]\n"
      "st1h { z23.h }, p3, [x21, #2, MUL VL]\n"
      "st1h { z21.h }, p3, [x21, #3, MUL VL]\n"
      "st1h { z19.h }, p3, [x21, #4, MUL VL]\n"
      "st1h { z17.h }, p3, [x21, #5, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x23, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x23\n"
      "whilelt p2.s, XZR, x20\n"
      "ld1w { z20.s }, p2/Z, [x26]\n"
      "ld1w { z19.s }, p2/Z, [x24]\n"
      "decw x20\n"
      "whilelt p1.s, XZR, x20\n"
      "ld1w { z18.s }, p1/Z, [x26, #1, MUL VL]\n"
      "ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z25.s }, p0/Z, [x26, #2, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x24, #2, MUL VL]\n"
      "ld1w { z24.s }, p2/Z, [x25]\n"
      "ld1w { z30.s }, p1/Z, [x25, #1, MUL VL]\n"
      "zip1 z23.s, z20.s, z19.s\n"
      "zip2 z22.s, z20.s, z19.s\n"
      "ld1w { z29.s }, p0/Z, [x25, #2, MUL VL]\n"
      "ld1w { z21.s }, p2/Z, [x22]\n"
      "zip1 z20.s, z18.s, z17.s\n"
      "zip2 z19.s, z18.s, z17.s\n"
      "ld1w { z18.s }, p1/Z, [x22, #1, MUL VL]\n"
      "ld1w { z28.s }, p0/Z, [x22, #2, MUL VL]\n"
      "zip1 z17.s, z25.s, z16.s\n"
      "zip2 z16.s, z25.s, z16.s\n"
      "decd x23, ALL, MUL #6\n"
      ".inst 0x658aaefb  // bfcvt z27.h, p3/M, z23.s\n"
      "zip1 z26.s, z24.s, z21.s\n"
      "cmp x23, #0x0\n"
      ".inst 0x658aaed9  // bfcvt z25.h, p3/M, z22.s\n"
      "zip2 z24.s, z24.s, z21.s\n"
      "addvl x26, x26, #3\n"
      "addvl x25, x25, #3\n"
      ".inst 0x658aae97  // bfcvt z23.h, p3/M, z20.s\n"
      "zip1 z22.s, z30.s, z18.s\n"
      "addvl x24, x24, #3\n"
      "addvl x22, x22, #3\n"
      ".inst 0x658aae75  // bfcvt z21.h, p3/M, z19.s\n"
      "zip2 z20.s, z30.s, z18.s\n"
      ".inst 0x658aae33  // bfcvt z19.h, p3/M, z17.s\n"
      "zip1 z18.s, z29.s, z28.s\n"
      ".inst 0x658aae11  // bfcvt z17.h, p3/M, z16.s\n"
      "zip2 z16.s, z29.s, z28.s\n"
      ".inst 0x648aaf5b  // bfcvtnt z27.h, p3/M, z26.s\n"
      ".inst 0x648aaf19  // bfcvtnt z25.h, p3/M, z24.s\n"
      "st1h { z27.h }, p3, [x21]\n"
      ".inst 0x648aaed7  // bfcvtnt z23.h, p3/M, z22.s\n"
      ".inst 0x648aae95  // bfcvtnt z21.h, p3/M, z20.s\n"
      "st1h { z25.h }, p3, [x21, #1, MUL VL]\n"
      ".inst 0x648aae53  // bfcvtnt z19.h, p3/M, z18.s\n"
      ".inst 0x648aae11  // bfcvtnt z17.h, p3/M, z16.s\n"
      "st1h { z23.h }, p3, [x21, #2, MUL VL]\n"
      "st1h { z21.h }, p3, [x21, #3, MUL VL]\n"
      "st1h { z19.h }, p3, [x21, #4, MUL VL]\n"
      "st1h { z17.h }, p3, [x21, #5, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #6\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace
template<>
void Transform<6, 4, true, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_6VL_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
