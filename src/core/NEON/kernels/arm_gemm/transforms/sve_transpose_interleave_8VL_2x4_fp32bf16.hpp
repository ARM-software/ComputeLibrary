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

void sve_transpose_interleave_8VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 8 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p4.b\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "mov x23, %x[width]\n"
      "cnth x20, ALL, MUL #4\n"
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
      "ld1w { z19.s }, p4/Z, [x26]\n"
      "ld1w { z18.s }, p4/Z, [x26, #1, MUL VL]\n"
      "sub x23, x23, x20\n"
      "cmp x23, x20\n"
      "ld1w { z20.s }, p4/Z, [x26, #2, MUL VL]\n"
      "ld1w { z24.s }, p4/Z, [x26, #3, MUL VL]\n"
      "ld1w { z23.s }, p4/Z, [x24]\n"
      "ld1w { z17.s }, p4/Z, [x24, #1, MUL VL]\n"
      "zip1 z22.s, z19.s, z23.s\n"
      "zip2 z21.s, z19.s, z23.s\n"
      "ld1w { z31.s }, p4/Z, [x24, #2, MUL VL]\n"
      "ld1w { z16.s }, p4/Z, [x24, #3, MUL VL]\n"
      "zip1 z9.s, z18.s, z17.s\n"
      "zip2 z7.s, z18.s, z17.s\n"
      "ld1w { z19.s }, p4/Z, [x26, #4, MUL VL]\n"
      "ld1w { z18.s }, p4/Z, [x26, #5, MUL VL]\n"
      "zip1 z6.s, z20.s, z31.s\n"
      "zip2 z5.s, z20.s, z31.s\n"
      "ld1w { z15.s }, p4/Z, [x26, #6, MUL VL]\n"
      "ld1w { z20.s }, p4/Z, [x26, #7, MUL VL]\n"
      "zip1 z3.s, z24.s, z16.s\n"
      "zip2 z2.s, z24.s, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x24, #4, MUL VL]\n"
      "ld1w { z17.s }, p4/Z, [x24, #5, MUL VL]\n"
      "zip1 z1.s, z19.s, z16.s\n"
      "zip2 z0.s, z19.s, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x24, #6, MUL VL]\n"
      "ld1w { z19.s }, p4/Z, [x24, #7, MUL VL]\n"
      "zip1 z31.s, z18.s, z17.s\n"
      "zip2 z30.s, z18.s, z17.s\n"
      "ld1w { z18.s }, p4/Z, [x25]\n"
      "ld1w { z17.s }, p4/Z, [x25, #1, MUL VL]\n"
      "zip1 z29.s, z15.s, z16.s\n"
      "zip2 z28.s, z15.s, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x25, #2, MUL VL]\n"
      "ld1w { z23.s }, p4/Z, [x25, #3, MUL VL]\n"
      "zip1 z27.s, z20.s, z19.s\n"
      "zip2 z26.s, z20.s, z19.s\n"
      "ld1w { z11.s }, p4/Z, [x22]\n"
      "ld1w { z8.s }, p4/Z, [x22, #1, MUL VL]\n"
      ".inst 0x658ab2d8  // bfcvt z24.h, p4/M, z22.s\n"
      "zip1 z25.s, z18.s, z11.s\n"
      "ld1w { z4.s }, p4/Z, [x22, #2, MUL VL]\n"
      "ld1w { z22.s }, p4/Z, [x22, #3, MUL VL]\n"
      ".inst 0x658ab2af  // bfcvt z15.h, p4/M, z21.s\n"
      "zip2 z14.s, z18.s, z11.s\n"
      "ld1w { z21.s }, p4/Z, [x25, #4, MUL VL]\n"
      "ld1w { z20.s }, p4/Z, [x25, #5, MUL VL]\n"
      ".inst 0x658ab12d  // bfcvt z13.h, p4/M, z9.s\n"
      "zip1 z12.s, z17.s, z8.s\n"
      "ld1w { z11.s }, p4/Z, [x25, #6, MUL VL]\n"
      "ld1w { z10.s }, p4/Z, [x25, #7, MUL VL]\n"
      ".inst 0x658ab0e9  // bfcvt z9.h, p4/M, z7.s\n"
      "zip2 z8.s, z17.s, z8.s\n"
      "ld1w { z19.s }, p4/Z, [x22, #4, MUL VL]\n"
      "ld1w { z18.s }, p4/Z, [x22, #5, MUL VL]\n"
      ".inst 0x658ab0c7  // bfcvt z7.h, p4/M, z6.s\n"
      "zip1 z6.s, z16.s, z4.s\n"
      "ld1w { z17.s }, p4/Z, [x22, #6, MUL VL]\n"
      ".inst 0x658ab0a5  // bfcvt z5.h, p4/M, z5.s\n"
      "zip2 z4.s, z16.s, z4.s\n"
      "ld1w { z16.s }, p4/Z, [x22, #7, MUL VL]\n"
      ".inst 0x658ab063  // bfcvt z3.h, p4/M, z3.s\n"
      ".inst 0x658ab042  // bfcvt z2.h, p4/M, z2.s\n"
      "addvl x26, x26, #8\n"
      "addvl x25, x25, #8\n"
      ".inst 0x658ab021  // bfcvt z1.h, p4/M, z1.s\n"
      ".inst 0x658ab000  // bfcvt z0.h, p4/M, z0.s\n"
      "addvl x24, x24, #8\n"
      "addvl x22, x22, #8\n"
      ".inst 0x658ab3ff  // bfcvt z31.h, p4/M, z31.s\n"
      ".inst 0x658ab3de  // bfcvt z30.h, p4/M, z30.s\n"
      ".inst 0x658ab3bd  // bfcvt z29.h, p4/M, z29.s\n"
      ".inst 0x658ab39c  // bfcvt z28.h, p4/M, z28.s\n"
      ".inst 0x658ab37b  // bfcvt z27.h, p4/M, z27.s\n"
      ".inst 0x658ab35a  // bfcvt z26.h, p4/M, z26.s\n"
      ".inst 0x648ab338  // bfcvtnt z24.h, p4/M, z25.s\n"
      "zip1 z25.s, z23.s, z22.s\n"
      "st1h { z24.h }, p4, [x21]\n"
      "zip2 z24.s, z23.s, z22.s\n"
      "zip1 z23.s, z21.s, z19.s\n"
      "zip2 z22.s, z21.s, z19.s\n"
      "zip1 z21.s, z20.s, z18.s\n"
      "zip2 z20.s, z20.s, z18.s\n"
      "zip1 z19.s, z11.s, z17.s\n"
      "zip2 z18.s, z11.s, z17.s\n"
      "zip1 z17.s, z10.s, z16.s\n"
      "zip2 z16.s, z10.s, z16.s\n"
      ".inst 0x648ab1cf  // bfcvtnt z15.h, p4/M, z14.s\n"
      "st1h { z15.h }, p4, [x21, #1, MUL VL]\n"
      ".inst 0x648ab18d  // bfcvtnt z13.h, p4/M, z12.s\n"
      ".inst 0x648ab109  // bfcvtnt z9.h, p4/M, z8.s\n"
      "st1h { z13.h }, p4, [x21, #2, MUL VL]\n"
      ".inst 0x648ab0c7  // bfcvtnt z7.h, p4/M, z6.s\n"
      ".inst 0x648ab085  // bfcvtnt z5.h, p4/M, z4.s\n"
      "st1h { z9.h }, p4, [x21, #3, MUL VL]\n"
      ".inst 0x648ab323  // bfcvtnt z3.h, p4/M, z25.s\n"
      ".inst 0x648ab302  // bfcvtnt z2.h, p4/M, z24.s\n"
      "st1h { z7.h }, p4, [x21, #4, MUL VL]\n"
      "st1h { z5.h }, p4, [x21, #5, MUL VL]\n"
      ".inst 0x648ab2e1  // bfcvtnt z1.h, p4/M, z23.s\n"
      ".inst 0x648ab2c0  // bfcvtnt z0.h, p4/M, z22.s\n"
      "st1h { z3.h }, p4, [x21, #6, MUL VL]\n"
      ".inst 0x648ab2bf  // bfcvtnt z31.h, p4/M, z21.s\n"
      ".inst 0x648ab29e  // bfcvtnt z30.h, p4/M, z20.s\n"
      "st1h { z2.h }, p4, [x21, #7, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      ".inst 0x648ab27d  // bfcvtnt z29.h, p4/M, z19.s\n"
      ".inst 0x648ab25c  // bfcvtnt z28.h, p4/M, z18.s\n"
      ".inst 0x648ab23b  // bfcvtnt z27.h, p4/M, z17.s\n"
      ".inst 0x648ab21a  // bfcvtnt z26.h, p4/M, z16.s\n"
      "st1h { z1.h }, p4, [x21]\n"
      "st1h { z0.h }, p4, [x21, #1, MUL VL]\n"
      "st1h { z31.h }, p4, [x21, #2, MUL VL]\n"
      "st1h { z30.h }, p4, [x21, #3, MUL VL]\n"
      "st1h { z29.h }, p4, [x21, #4, MUL VL]\n"
      "st1h { z28.h }, p4, [x21, #5, MUL VL]\n"
      "st1h { z27.h }, p4, [x21, #6, MUL VL]\n"
      "st1h { z26.h }, p4, [x21, #7, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x23, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x23\n"
      "whilelt p3.s, XZR, x20\n"
      "ld1w { z22.s }, p3/Z, [x26]\n"
      "ld1w { z21.s }, p3/Z, [x24]\n"
      "decw x20\n"
      "whilelt p2.s, XZR, x20\n"
      "ld1w { z20.s }, p2/Z, [x26, #1, MUL VL]\n"
      "ld1w { z19.s }, p2/Z, [x24, #1, MUL VL]\n"
      "decw x20\n"
      "whilelt p1.s, XZR, x20\n"
      "ld1w { z18.s }, p1/Z, [x26, #2, MUL VL]\n"
      "ld1w { z17.s }, p1/Z, [x24, #2, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z28.s }, p0/Z, [x26, #3, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x24, #3, MUL VL]\n"
      "ld1w { z27.s }, p3/Z, [x25]\n"
      "ld1w { z3.s }, p2/Z, [x25, #1, MUL VL]\n"
      "zip1 z26.s, z22.s, z21.s\n"
      "zip2 z25.s, z22.s, z21.s\n"
      "ld1w { z2.s }, p1/Z, [x25, #2, MUL VL]\n"
      "ld1w { z1.s }, p0/Z, [x25, #3, MUL VL]\n"
      "zip1 z24.s, z20.s, z19.s\n"
      "zip2 z23.s, z20.s, z19.s\n"
      "ld1w { z22.s }, p3/Z, [x22]\n"
      "ld1w { z21.s }, p2/Z, [x22, #1, MUL VL]\n"
      "zip1 z20.s, z18.s, z17.s\n"
      "zip2 z19.s, z18.s, z17.s\n"
      "ld1w { z18.s }, p1/Z, [x22, #2, MUL VL]\n"
      "ld1w { z0.s }, p0/Z, [x22, #3, MUL VL]\n"
      "zip1 z17.s, z28.s, z16.s\n"
      "zip2 z16.s, z28.s, z16.s\n"
      "decd x23, ALL, MUL #8\n"
      ".inst 0x658ab35f  // bfcvt z31.h, p4/M, z26.s\n"
      "zip1 z30.s, z27.s, z22.s\n"
      "cmp x23, #0x0\n"
      ".inst 0x658ab33d  // bfcvt z29.h, p4/M, z25.s\n"
      "zip2 z28.s, z27.s, z22.s\n"
      "addvl x26, x26, #4\n"
      "addvl x25, x25, #4\n"
      ".inst 0x658ab31b  // bfcvt z27.h, p4/M, z24.s\n"
      "zip1 z26.s, z3.s, z21.s\n"
      "addvl x24, x24, #4\n"
      "addvl x22, x22, #4\n"
      ".inst 0x658ab2f9  // bfcvt z25.h, p4/M, z23.s\n"
      "zip2 z24.s, z3.s, z21.s\n"
      ".inst 0x658ab297  // bfcvt z23.h, p4/M, z20.s\n"
      "zip1 z22.s, z2.s, z18.s\n"
      ".inst 0x658ab275  // bfcvt z21.h, p4/M, z19.s\n"
      "zip2 z20.s, z2.s, z18.s\n"
      ".inst 0x658ab233  // bfcvt z19.h, p4/M, z17.s\n"
      "zip1 z18.s, z1.s, z0.s\n"
      ".inst 0x658ab211  // bfcvt z17.h, p4/M, z16.s\n"
      "zip2 z16.s, z1.s, z0.s\n"
      ".inst 0x648ab3df  // bfcvtnt z31.h, p4/M, z30.s\n"
      ".inst 0x648ab39d  // bfcvtnt z29.h, p4/M, z28.s\n"
      "st1h { z31.h }, p4, [x21]\n"
      ".inst 0x648ab35b  // bfcvtnt z27.h, p4/M, z26.s\n"
      ".inst 0x648ab319  // bfcvtnt z25.h, p4/M, z24.s\n"
      "st1h { z29.h }, p4, [x21, #1, MUL VL]\n"
      ".inst 0x648ab2d7  // bfcvtnt z23.h, p4/M, z22.s\n"
      ".inst 0x648ab295  // bfcvtnt z21.h, p4/M, z20.s\n"
      "st1h { z27.h }, p4, [x21, #2, MUL VL]\n"
      ".inst 0x648ab253  // bfcvtnt z19.h, p4/M, z18.s\n"
      ".inst 0x648ab211  // bfcvtnt z17.h, p4/M, z16.s\n"
      "st1h { z25.h }, p4, [x21, #3, MUL VL]\n"
      "st1h { z23.h }, p4, [x21, #4, MUL VL]\n"
      "st1h { z21.h }, p4, [x21, #5, MUL VL]\n"
      "st1h { z19.h }, p4, [x21, #6, MUL VL]\n"
      "st1h { z17.h }, p4, [x21, #7, MUL VL]\n"
      "add x21, x21, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
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

#endif
