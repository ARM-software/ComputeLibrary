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

void sve_transpose_interleave_8VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 8 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p2.b\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "cnth x20, ALL, MUL #4\n"
      "cmp %x[height], #0x3\n"
      "mov x24, %x[out]\n"
      "add x23, x26, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "csel x21, x21, %x[pad_row], GT\n"
      "csel x22, x22, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x23, x23, %x[pad_row], GT\n"
      "cmp x25, x20\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z25.s }, p2/Z, [x26]\n"
      "ld1w { z23.s }, p2/Z, [x26, #1, MUL VL]\n"
      "sub x25, x25, x20\n"
      "ld1w { z5.s }, p2/Z, [x26, #2, MUL VL]\n"
      "ld1w { z31.s }, p2/Z, [x26, #3, MUL VL]\n"
      "cmp x25, x20\n"
      "ld1w { z16.s }, p2/Z, [x22]\n"
      "ld1w { z18.s }, p2/Z, [x22, #1, MUL VL]\n"
      "ld1w { z21.s }, p2/Z, [x22, #2, MUL VL]\n"
      "ld1w { z14.s }, p2/Z, [x22, #3, MUL VL]\n"
      "ld1w { z3.s }, p2/Z, [x26, #4, MUL VL]\n"
      "ld1w { z30.s }, p2/Z, [x26, #5, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x26, #6, MUL VL]\n"
      "ld1w { z22.s }, p2/Z, [x26, #7, MUL VL]\n"
      "zip1 z19.s, z25.s, z16.s\n"
      "zip2 z17.s, z25.s, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x22, #4, MUL VL]\n"
      "ld1w { z13.s }, p2/Z, [x22, #5, MUL VL]\n"
      "zip1 z11.s, z23.s, z18.s\n"
      "zip2 z10.s, z23.s, z18.s\n"
      "ld1w { z18.s }, p2/Z, [x22, #6, MUL VL]\n"
      "ld1w { z15.s }, p2/Z, [x22, #7, MUL VL]\n"
      "zip1 z9.s, z5.s, z21.s\n"
      "zip2 z8.s, z5.s, z21.s\n"
      "ld1w { z12.s }, p2/Z, [x23]\n"
      "ld1w { z27.s }, p2/Z, [x23, #1, MUL VL]\n"
      "zip1 z7.s, z31.s, z14.s\n"
      "zip2 z6.s, z31.s, z14.s\n"
      "ld1w { z25.s }, p2/Z, [x23, #2, MUL VL]\n"
      "ld1w { z23.s }, p2/Z, [x23, #3, MUL VL]\n"
      "zip1 z5.s, z3.s, z16.s\n"
      "zip2 z4.s, z3.s, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x21]\n"
      "ld1w { z28.s }, p2/Z, [x21, #1, MUL VL]\n"
      "zip1 z3.s, z30.s, z13.s\n"
      "zip2 z2.s, z30.s, z13.s\n"
      "ld1w { z26.s }, p2/Z, [x21, #2, MUL VL]\n"
      "ld1w { z24.s }, p2/Z, [x21, #3, MUL VL]\n"
      "zip1 z1.s, z20.s, z18.s\n"
      "zip2 z0.s, z20.s, z18.s\n"
      "ld1w { z21.s }, p2/Z, [x23, #4, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x23, #5, MUL VL]\n"
      "zip1 z31.s, z22.s, z15.s\n"
      "zip2 z30.s, z22.s, z15.s\n"
      "ld1w { z14.s }, p2/Z, [x23, #6, MUL VL]\n"
      "ld1w { z15.s }, p2/Z, [x23, #7, MUL VL]\n"
      ".inst 0x658aaa76  // bfcvt z22.h, p2/M, z19.s\n"
      "zip1 z29.s, z12.s, z16.s\n"
      "ld1w { z19.s }, p2/Z, [x21, #4, MUL VL]\n"
      "ld1w { z18.s }, p2/Z, [x21, #5, MUL VL]\n"
      ".inst 0x658aaa2d  // bfcvt z13.h, p2/M, z17.s\n"
      "zip2 z12.s, z12.s, z16.s\n"
      "ld1w { z17.s }, p2/Z, [x21, #6, MUL VL]\n"
      "ld1w { z16.s }, p2/Z, [x21, #7, MUL VL]\n"
      ".inst 0x658aa96b  // bfcvt z11.h, p2/M, z11.s\n"
      ".inst 0x658aa94a  // bfcvt z10.h, p2/M, z10.s\n"
      ".inst 0x658aa929  // bfcvt z9.h, p2/M, z9.s\n"
      ".inst 0x658aa908  // bfcvt z8.h, p2/M, z8.s\n"
      "addvl x26, x26, #8\n"
      "addvl x23, x23, #8\n"
      ".inst 0x658aa8e7  // bfcvt z7.h, p2/M, z7.s\n"
      ".inst 0x658aa8c6  // bfcvt z6.h, p2/M, z6.s\n"
      "addvl x22, x22, #8\n"
      "addvl x21, x21, #8\n"
      ".inst 0x658aa8a5  // bfcvt z5.h, p2/M, z5.s\n"
      ".inst 0x658aa884  // bfcvt z4.h, p2/M, z4.s\n"
      ".inst 0x658aa863  // bfcvt z3.h, p2/M, z3.s\n"
      ".inst 0x658aa842  // bfcvt z2.h, p2/M, z2.s\n"
      ".inst 0x658aa821  // bfcvt z1.h, p2/M, z1.s\n"
      ".inst 0x658aa800  // bfcvt z0.h, p2/M, z0.s\n"
      ".inst 0x658aabff  // bfcvt z31.h, p2/M, z31.s\n"
      ".inst 0x658aabde  // bfcvt z30.h, p2/M, z30.s\n"
      ".inst 0x648aabb6  // bfcvtnt z22.h, p2/M, z29.s\n"
      "zip1 z29.s, z27.s, z28.s\n"
      "zip2 z28.s, z27.s, z28.s\n"
      "zip1 z27.s, z25.s, z26.s\n"
      "zip2 z26.s, z25.s, z26.s\n"
      "zip1 z25.s, z23.s, z24.s\n"
      "zip2 z24.s, z23.s, z24.s\n"
      "zip1 z23.s, z21.s, z19.s\n"
      "st1h { z22.h }, p2, [x24]\n"
      "zip2 z22.s, z21.s, z19.s\n"
      "zip1 z21.s, z20.s, z18.s\n"
      "zip2 z20.s, z20.s, z18.s\n"
      "zip1 z19.s, z14.s, z17.s\n"
      "zip2 z18.s, z14.s, z17.s\n"
      "zip1 z17.s, z15.s, z16.s\n"
      "zip2 z16.s, z15.s, z16.s\n"
      ".inst 0x648aa98d  // bfcvtnt z13.h, p2/M, z12.s\n"
      ".inst 0x648aabab  // bfcvtnt z11.h, p2/M, z29.s\n"
      ".inst 0x648aab8a  // bfcvtnt z10.h, p2/M, z28.s\n"
      ".inst 0x648aab69  // bfcvtnt z9.h, p2/M, z27.s\n"
      ".inst 0x648aab48  // bfcvtnt z8.h, p2/M, z26.s\n"
      ".inst 0x648aab27  // bfcvtnt z7.h, p2/M, z25.s\n"
      ".inst 0x648aab06  // bfcvtnt z6.h, p2/M, z24.s\n"
      "st1h { z13.h }, p2, [x24, #1, MUL VL]\n"
      "st1h { z11.h }, p2, [x24, #2, MUL VL]\n"
      ".inst 0x648aaae5  // bfcvtnt z5.h, p2/M, z23.s\n"
      ".inst 0x648aaac4  // bfcvtnt z4.h, p2/M, z22.s\n"
      "st1h { z10.h }, p2, [x24, #3, MUL VL]\n"
      ".inst 0x648aaaa3  // bfcvtnt z3.h, p2/M, z21.s\n"
      ".inst 0x648aaa82  // bfcvtnt z2.h, p2/M, z20.s\n"
      "st1h { z9.h }, p2, [x24, #4, MUL VL]\n"
      ".inst 0x648aaa61  // bfcvtnt z1.h, p2/M, z19.s\n"
      ".inst 0x648aaa40  // bfcvtnt z0.h, p2/M, z18.s\n"
      "st1h { z8.h }, p2, [x24, #5, MUL VL]\n"
      ".inst 0x648aaa3f  // bfcvtnt z31.h, p2/M, z17.s\n"
      ".inst 0x648aaa1e  // bfcvtnt z30.h, p2/M, z16.s\n"
      "st1h { z7.h }, p2, [x24, #6, MUL VL]\n"
      "st1h { z6.h }, p2, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "st1h { z5.h }, p2, [x24]\n"
      "st1h { z4.h }, p2, [x24, #1, MUL VL]\n"
      "st1h { z3.h }, p2, [x24, #2, MUL VL]\n"
      "st1h { z2.h }, p2, [x24, #3, MUL VL]\n"
      "st1h { z1.h }, p2, [x24, #4, MUL VL]\n"
      "st1h { z0.h }, p2, [x24, #5, MUL VL]\n"
      "st1h { z31.h }, p2, [x24, #6, MUL VL]\n"
      "st1h { z30.h }, p2, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x25\n"
      "decd x25, ALL, MUL #8\n"
      "whilelt p1.s, XZR, x20\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z18.s }, p1/Z, [x26]\n"
      "ld1w { z16.s }, p1/Z, [x22]\n"
      "ld1w { z26.s }, p1/Z, [x23]\n"
      "ld1w { z25.s }, p1/Z, [x21]\n"
      "whilelt p1.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z20.s }, p0/Z, [x26, #1, MUL VL]\n"
      "ld1w { z17.s }, p0/Z, [x22, #1, MUL VL]\n"
      "ld1w { z24.s }, p0/Z, [x23, #1, MUL VL]\n"
      "ld1w { z23.s }, p0/Z, [x21, #1, MUL VL]\n"
      "zip1 z19.s, z18.s, z16.s\n"
      "zip2 z16.s, z18.s, z16.s\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z22.s }, p1/Z, [x26, #2, MUL VL]\n"
      "ld1w { z18.s }, p1/Z, [x22, #2, MUL VL]\n"
      "zip1 z3.s, z26.s, z25.s\n"
      "ld1w { z2.s }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1w { z1.s }, p1/Z, [x21, #2, MUL VL]\n"
      "zip1 z21.s, z20.s, z17.s\n"
      "zip2 z17.s, z20.s, z17.s\n"
      ".inst 0x658aaa60  // bfcvt z0.h, p2/M, z19.s\n"
      ".inst 0x658aaa1f  // bfcvt z31.h, p2/M, z16.s\n"
      "cmp x25, #0x0\n"
      "ld1w { z20.s }, p0/Z, [x26, #3, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x22, #3, MUL VL]\n"
      "ld1w { z30.s }, p0/Z, [x23, #3, MUL VL]\n"
      "zip1 z19.s, z22.s, z18.s\n"
      "zip2 z18.s, z22.s, z18.s\n"
      "ld1w { z29.s }, p0/Z, [x21, #3, MUL VL]\n"
      "zip2 z28.s, z26.s, z25.s\n"
      ".inst 0x658aaabb  // bfcvt z27.h, p2/M, z21.s\n"
      "addvl x26, x26, #4\n"
      "zip1 z26.s, z24.s, z23.s\n"
      ".inst 0x658aaa39  // bfcvt z25.h, p2/M, z17.s\n"
      "addvl x23, x23, #4\n"
      "addvl x22, x22, #4\n"
      "zip1 z17.s, z20.s, z16.s\n"
      "zip2 z16.s, z20.s, z16.s\n"
      "addvl x21, x21, #4\n"
      "zip2 z24.s, z24.s, z23.s\n"
      ".inst 0x658aaa77  // bfcvt z23.h, p2/M, z19.s\n"
      "zip1 z22.s, z2.s, z1.s\n"
      ".inst 0x658aaa55  // bfcvt z21.h, p2/M, z18.s\n"
      "zip2 z20.s, z2.s, z1.s\n"
      ".inst 0x658aaa33  // bfcvt z19.h, p2/M, z17.s\n"
      "zip1 z18.s, z30.s, z29.s\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "zip2 z16.s, z30.s, z29.s\n"
      ".inst 0x648aa860  // bfcvtnt z0.h, p2/M, z3.s\n"
      ".inst 0x648aab9f  // bfcvtnt z31.h, p2/M, z28.s\n"
      ".inst 0x648aab5b  // bfcvtnt z27.h, p2/M, z26.s\n"
      ".inst 0x648aab19  // bfcvtnt z25.h, p2/M, z24.s\n"
      ".inst 0x648aaad7  // bfcvtnt z23.h, p2/M, z22.s\n"
      ".inst 0x648aaa95  // bfcvtnt z21.h, p2/M, z20.s\n"
      ".inst 0x648aaa53  // bfcvtnt z19.h, p2/M, z18.s\n"
      "st1h { z0.h }, p2, [x24]\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z31.h }, p2, [x24, #1, MUL VL]\n"
      "st1h { z27.h }, p2, [x24, #2, MUL VL]\n"
      "st1h { z25.h }, p2, [x24, #3, MUL VL]\n"
      "st1h { z23.h }, p2, [x24, #4, MUL VL]\n"
      "st1h { z21.h }, p2, [x24, #5, MUL VL]\n"
      "st1h { z19.h }, p2, [x24, #6, MUL VL]\n"
      "st1h { z17.h }, p2, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace
template<>
void Transform<8, 4, true, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_8VL_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}


#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
