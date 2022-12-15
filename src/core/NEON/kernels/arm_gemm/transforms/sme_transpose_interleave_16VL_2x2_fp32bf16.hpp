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

void sme_transpose_interleave_16VL_2x2_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 16 * roundup<size_t>(height, 2) * sme::get_vector_length<uint16_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "ptrue p7.b\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "add x24, x25, %x[in_stride]\n"
      "cmp %x[height], #0x1\n"
      "add %x[in], x24, %x[in_stride]\n"
      "mov x23, %x[out]\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "sub %x[height], %x[height], #0x2\n"
      "mov x22, %x[width]\n"
      "2:"  // Main row loop: Column loop
      "mov x21, x22\n"
      "whilelt p1.s, XZR, x21\n"
      "ld1w { z16.s }, p1/Z, [x25]\n"
      ".inst 0x658abe00  // bfcvt z0.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p0.s, XZR, x21\n"
      "ld1w { z16.s }, p0/Z, [x25, #1, MUL VL]\n"
      ".inst 0x658abe1f  // bfcvt z31.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p6.s, XZR, x21\n"
      "ld1w { z16.s }, p6/Z, [x25, #2, MUL VL]\n"
      ".inst 0x658abe1e  // bfcvt z30.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p5.s, XZR, x21\n"
      "ld1w { z16.s }, p5/Z, [x25, #3, MUL VL]\n"
      ".inst 0x658abe1d  // bfcvt z29.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p4.s, XZR, x21\n"
      "ld1w { z16.s }, p4/Z, [x25, #4, MUL VL]\n"
      ".inst 0x658abe1c  // bfcvt z28.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p3.s, XZR, x21\n"
      "ld1w { z16.s }, p3/Z, [x25, #5, MUL VL]\n"
      ".inst 0x658abe1b  // bfcvt z27.h, p7/M, z16.s\n"
      "decw x21\n"
      "whilelt p2.s, XZR, x21\n"
      "ld1w { z16.s }, p2/Z, [x25, #6, MUL VL]\n"
      ".inst 0x658abe1a  // bfcvt z26.h, p7/M, z16.s\n"
      "decw x21\n"
      "ld1w { z16.s }, p1/Z, [x24]\n"
      "whilelt p1.s, XZR, x21\n"
      ".inst 0x648abe00  // bfcvtnt z0.h, p7/M, z16.s\n"
      "decw x21\n"
      "ld1w { z16.s }, p1/Z, [x25, #7, MUL VL]\n"
      "addvl x25, x25, #16\n"
      ".inst 0x658abe19  // bfcvt z25.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x24, #1, MUL VL]\n"
      "whilelt p0.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1f  // bfcvtnt z31.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x25, #-8, MUL VL]\n"
      ".inst 0x658abe18  // bfcvt z24.h, p7/M, z16.s\n"
      "mov x20, x23\n"
      "decw x22, ALL, MUL #16\n"
      "ld1w { z16.s }, p6/Z, [x24, #2, MUL VL]\n"
      "whilelt p6.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1e  // bfcvtnt z30.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p6/Z, [x25, #-7, MUL VL]\n"
      ".inst 0x658abe17  // bfcvt z23.h, p7/M, z16.s\n"
      "add x23, x23, %x[out_stride]\n"
      "ld1w { z16.s }, p5/Z, [x24, #3, MUL VL]\n"
      "whilelt p5.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1d  // bfcvtnt z29.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p5/Z, [x25, #-6, MUL VL]\n"
      ".inst 0x658abe16  // bfcvt z22.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x24, #4, MUL VL]\n"
      "whilelt p4.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1c  // bfcvtnt z28.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x25, #-5, MUL VL]\n"
      ".inst 0x658abe15  // bfcvt z21.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p3/Z, [x24, #5, MUL VL]\n"
      "whilelt p3.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1b  // bfcvtnt z27.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p3/Z, [x25, #-4, MUL VL]\n"
      ".inst 0x658abe14  // bfcvt z20.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x24, #6, MUL VL]\n"
      "whilelt p2.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe1a  // bfcvtnt z26.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x25, #-3, MUL VL]\n"
      ".inst 0x658abe13  // bfcvt z19.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x24, #7, MUL VL]\n"
      "whilelt p1.s, XZR, x21\n"
      "decw x21\n"
      ".inst 0x648abe19  // bfcvtnt z25.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x25, #-2, MUL VL]\n"
      "addvl x24, x24, #16\n"
      ".inst 0x658abe12  // bfcvt z18.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x24, #-8, MUL VL]\n"
      "whilelt p0.s, XZR, x21\n"
      "cmp x22, #0x0\n"
      ".inst 0x648abe18  // bfcvtnt z24.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x25, #-1, MUL VL]\n"
      ".inst 0x658abe11  // bfcvt z17.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p6/Z, [x24, #-7, MUL VL]\n"
      ".inst 0x648abe17  // bfcvtnt z23.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p5/Z, [x24, #-6, MUL VL]\n"
      ".inst 0x648abe16  // bfcvtnt z22.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x24, #-5, MUL VL]\n"
      ".inst 0x648abe15  // bfcvtnt z21.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p3/Z, [x24, #-4, MUL VL]\n"
      ".inst 0x648abe14  // bfcvtnt z20.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p2/Z, [x24, #-3, MUL VL]\n"
      ".inst 0x648abe13  // bfcvtnt z19.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x24, #-2, MUL VL]\n"
      ".inst 0x648abe12  // bfcvtnt z18.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x24, #-1, MUL VL]\n"
      "st1h { z0.h }, p7, [x20]\n"
      ".inst 0x648abe11  // bfcvtnt z17.h, p7/M, z16.s\n"
      "st1h { z31.h }, p7, [x20, #1, MUL VL]\n"
      "st1h { z30.h }, p7, [x20, #2, MUL VL]\n"
      "st1h { z29.h }, p7, [x20, #3, MUL VL]\n"
      "st1h { z28.h }, p7, [x20, #4, MUL VL]\n"
      "st1h { z27.h }, p7, [x20, #5, MUL VL]\n"
      "st1h { z26.h }, p7, [x20, #6, MUL VL]\n"
      "st1h { z25.h }, p7, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1h { z24.h }, p7, [x20, #-8, MUL VL]\n"
      "st1h { z23.h }, p7, [x20, #-7, MUL VL]\n"
      "st1h { z22.h }, p7, [x20, #-6, MUL VL]\n"
      "st1h { z21.h }, p7, [x20, #-5, MUL VL]\n"
      "st1h { z20.h }, p7, [x20, #-4, MUL VL]\n"
      "st1h { z19.h }, p7, [x20, #-3, MUL VL]\n"
      "st1h { z18.h }, p7, [x20, #-2, MUL VL]\n"
      "st1h { z17.h }, p7, [x20, #-1, MUL VL]\n"
      "bgt 2b\n"
      "3:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace
template<>
void Transform<16, 2, true, VLType::SME>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_16VL_2x2_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
