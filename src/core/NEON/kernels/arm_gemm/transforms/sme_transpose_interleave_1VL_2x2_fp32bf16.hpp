/*
 * Copyright (c) 2022-2024 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SME)

namespace {

void sme_transpose_interleave_1VL_2x2_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 1 * roundup<size_t>(height, 2) * sme::get_vector_length<uint16_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cmp %x[height], #0x4\n"
      "ptrue p1.b\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "add x24, x26, %x[in_stride]\n"
      "cnth x23, ALL, MUL #2\n"
      "add x21, x24, %x[in_stride]\n"
      "cmp x25, x23\n"
      "add x20, x21, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z19.s }, p1/Z, [x26]\n"
      "sub x25, x25, x23\n"
      "ld1w { z18.s }, p1/Z, [x21]\n"
      "cmp x25, x23\n"
      "ld1w { z17.s }, p1/Z, [x26, #1, MUL VL]\n"
      "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
      ".inst 0x658aa67b  // bfcvt z27.h, p1/M, z19.s\n"
      "ld1w { z19.s }, p1/Z, [x26, #2, MUL VL]\n"
      ".inst 0x658aa65a  // bfcvt z26.h, p1/M, z18.s\n"
      "ld1w { z18.s }, p1/Z, [x21, #2, MUL VL]\n"
      ".inst 0x658aa639  // bfcvt z25.h, p1/M, z17.s\n"
      "ld1w { z17.s }, p1/Z, [x26, #3, MUL VL]\n"
      ".inst 0x658aa618  // bfcvt z24.h, p1/M, z16.s\n"
      "addvl x26, x26, #4\n"
      "ld1w { z16.s }, p1/Z, [x21, #3, MUL VL]\n"
      ".inst 0x658aa677  // bfcvt z23.h, p1/M, z19.s\n"
      "addvl x21, x21, #4\n"
      "ld1w { z19.s }, p1/Z, [x24]\n"
      ".inst 0x658aa656  // bfcvt z22.h, p1/M, z18.s\n"
      "ld1w { z18.s }, p1/Z, [x20]\n"
      ".inst 0x658aa635  // bfcvt z21.h, p1/M, z17.s\n"
      "ld1w { z17.s }, p1/Z, [x24, #1, MUL VL]\n"
      ".inst 0x658aa614  // bfcvt z20.h, p1/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x20, #1, MUL VL]\n"
      ".inst 0x648aa67b  // bfcvtnt z27.h, p1/M, z19.s\n"
      "ld1w { z19.s }, p1/Z, [x24, #2, MUL VL]\n"
      ".inst 0x648aa65a  // bfcvtnt z26.h, p1/M, z18.s\n"
      "ld1w { z18.s }, p1/Z, [x20, #2, MUL VL]\n"
      ".inst 0x648aa639  // bfcvtnt z25.h, p1/M, z17.s\n"
      "ld1w { z17.s }, p1/Z, [x24, #3, MUL VL]\n"
      ".inst 0x648aa618  // bfcvtnt z24.h, p1/M, z16.s\n"
      "addvl x24, x24, #4\n"
      "ld1w { z16.s }, p1/Z, [x20, #3, MUL VL]\n"
      "st1h { z27.h }, p1, [x22]\n"
      ".inst 0x648aa677  // bfcvtnt z23.h, p1/M, z19.s\n"
      "addvl x20, x20, #4\n"
      "st1h { z26.h }, p1, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      ".inst 0x648aa656  // bfcvtnt z22.h, p1/M, z18.s\n"
      "st1h { z25.h }, p1, [x22]\n"
      ".inst 0x648aa635  // bfcvtnt z21.h, p1/M, z17.s\n"
      "st1h { z24.h }, p1, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      ".inst 0x648aa614  // bfcvtnt z20.h, p1/M, z16.s\n"
      "st1h { z23.h }, p1, [x22]\n"
      "st1h { z22.h }, p1, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z21.h }, p1, [x22]\n"
      "st1h { z20.h }, p1, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.s, XZR, x25\n"
      "decw x25\n"
      "ld1w { z17.s }, p0/Z, [x26]\n"
      "cmp x25, #0x0\n"
      "addvl x26, x26, #1\n"
      "ld1w { z16.s }, p0/Z, [x21]\n"
      "addvl x21, x21, #1\n"
      "ld1w { z19.s }, p0/Z, [x24]\n"
      "addvl x24, x24, #1\n"
      ".inst 0x658aa632  // bfcvt z18.h, p1/M, z17.s\n"
      "ld1w { z17.s }, p0/Z, [x20]\n"
      "addvl x20, x20, #1\n"
      ".inst 0x658aa610  // bfcvt z16.h, p1/M, z16.s\n"
      ".inst 0x648aa672  // bfcvtnt z18.h, p1/M, z19.s\n"
      ".inst 0x648aa630  // bfcvtnt z16.h, p1/M, z17.s\n"
      "st1h { z18.h }, p1, [x22]\n"
      "st1h { z16.h }, p1, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x4\n"
      "addvl %x[out], %x[out], #2\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x26, %x[in]\n"
      "cmp %x[height], #0x1\n"
      "add x24, x26, %x[in_stride]\n"
      "mov x21, %x[width]\n"
      "cnth x20, ALL, MUL #2\n"
      "add %x[in], x24, %x[in_stride]\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "cmp x21, x20\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x2\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1w { z19.s }, p1/Z, [x26]\n"
      "sub x21, x21, x20\n"
      "ld1w { z18.s }, p1/Z, [x26, #1, MUL VL]\n"
      "cmp x21, x20\n"
      "ld1w { z17.s }, p1/Z, [x26, #2, MUL VL]\n"
      "ld1w { z16.s }, p1/Z, [x26, #3, MUL VL]\n"
      ".inst 0x658aa677  // bfcvt z23.h, p1/M, z19.s\n"
      "addvl x26, x26, #4\n"
      "ld1w { z22.s }, p1/Z, [x24]\n"
      ".inst 0x658aa655  // bfcvt z21.h, p1/M, z18.s\n"
      "ld1w { z20.s }, p1/Z, [x24, #1, MUL VL]\n"
      ".inst 0x658aa633  // bfcvt z19.h, p1/M, z17.s\n"
      "ld1w { z18.s }, p1/Z, [x24, #2, MUL VL]\n"
      ".inst 0x658aa611  // bfcvt z17.h, p1/M, z16.s\n"
      "ld1w { z16.s }, p1/Z, [x24, #3, MUL VL]\n"
      ".inst 0x648aa6d7  // bfcvtnt z23.h, p1/M, z22.s\n"
      "addvl x24, x24, #4\n"
      ".inst 0x648aa695  // bfcvtnt z21.h, p1/M, z20.s\n"
      ".inst 0x648aa653  // bfcvtnt z19.h, p1/M, z18.s\n"
      ".inst 0x648aa611  // bfcvtnt z17.h, p1/M, z16.s\n"
      "st1h { z23.h }, p1, [x22]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z21.h }, p1, [x22]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z19.h }, p1, [x22]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z17.h }, p1, [x22]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "whilelt p0.s, XZR, x21\n"
      "decw x21\n"
      "ld1w { z16.s }, p0/Z, [x26]\n"
      "cmp x21, #0x0\n"
      "addvl x26, x26, #1\n"
      "ld1w { z17.s }, p0/Z, [x24]\n"
      "addvl x24, x24, #1\n"
      ".inst 0x658aa610  // bfcvt z16.h, p1/M, z16.s\n"
      ".inst 0x648aa630  // bfcvtnt z16.h, p1/M, z17.s\n"
      "st1h { z16.h }, p1, [x22]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #1\n"
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
void Transform<1, 2, true, VLType::SME>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_1VL_2x2_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}


#endif  // defined(ARM_COMPUTE_ENABLE_SME)
