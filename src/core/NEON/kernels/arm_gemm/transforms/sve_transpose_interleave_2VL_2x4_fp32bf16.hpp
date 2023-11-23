/*
 * Copyright (c) 2024 Arm Limited.
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

void sve_transpose_interleave_2VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 2 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p1.b\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "cnth x24\n"
      "cmp %x[height], #0x3\n"
      "mov x23, %x[out]\n"
      "add x22, x26, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "csel x20, x20, %x[pad_row], GT\n"
      "csel x21, x21, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "cmp x25, x24\n"
      "sub %x[height], %x[height], #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z18.s }, p1/Z, [x26]\n"
      "ld1w { z17.s }, p1/Z, [x21]\n"
      "sub x25, x25, x24\n"
      "ld1w { z21.s }, p1/Z, [x26, #1, MUL VL]\n"
      "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
      "cmp x25, x24\n"
      "addvl x26, x26, #2\n"
      "ld1w { z26.s }, p1/Z, [x22]\n"
      "ld1w { z20.s }, p1/Z, [x20]\n"
      "addvl x21, x21, #2\n"
      "zip1 z19.s, z18.s, z17.s\n"
      "zip2 z18.s, z18.s, z17.s\n"
      "ld1w { z25.s }, p1/Z, [x22, #1, MUL VL]\n"
      "ld1w { z24.s }, p1/Z, [x20, #1, MUL VL]\n"
      "addvl x22, x22, #2\n"
      "zip1 z17.s, z21.s, z16.s\n"
      "zip2 z16.s, z21.s, z16.s\n"
      "addvl x20, x20, #2\n"
      ".inst 0x658aa677  // bfcvt z23.h, p1/M, z19.s\n"
      "zip1 z22.s, z26.s, z20.s\n"
      ".inst 0x658aa655  // bfcvt z21.h, p1/M, z18.s\n"
      "zip2 z20.s, z26.s, z20.s\n"
      ".inst 0x658aa633  // bfcvt z19.h, p1/M, z17.s\n"
      "zip1 z18.s, z25.s, z24.s\n"
      ".inst 0x658aa611  // bfcvt z17.h, p1/M, z16.s\n"
      "zip2 z16.s, z25.s, z24.s\n"
      ".inst 0x648aa6d7  // bfcvtnt z23.h, p1/M, z22.s\n"
      ".inst 0x648aa695  // bfcvtnt z21.h, p1/M, z20.s\n"
      ".inst 0x648aa653  // bfcvtnt z19.h, p1/M, z18.s\n"
      ".inst 0x648aa611  // bfcvtnt z17.h, p1/M, z16.s\n"
      "st1h { z23.h }, p1, [x23]\n"
      "st1h { z21.h }, p1, [x23, #1, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "st1h { z19.h }, p1, [x23]\n"
      "st1h { z17.h }, p1, [x23, #1, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.s, XZR, x25\n"
      "decd x25, ALL, MUL #2\n"
      "ld1w { z19.s }, p0/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      "ld1w { z16.s }, p0/Z, [x21]\n"
      "addvl x21, x21, #1\n"
      "ld1w { z20.s }, p0/Z, [x22]\n"
      "addvl x22, x22, #1\n"
      "ld1w { z18.s }, p0/Z, [x20]\n"
      "addvl x20, x20, #1\n"
      "cmp x25, #0x0\n"
      "zip1 z17.s, z19.s, z16.s\n"
      "zip2 z16.s, z19.s, z16.s\n"
      "zip1 z19.s, z20.s, z18.s\n"
      "zip2 z18.s, z20.s, z18.s\n"
      ".inst 0x658aa631  // bfcvt z17.h, p1/M, z17.s\n"
      ".inst 0x658aa610  // bfcvt z16.h, p1/M, z16.s\n"
      ".inst 0x648aa671  // bfcvtnt z17.h, p1/M, z19.s\n"
      ".inst 0x648aa650  // bfcvtnt z16.h, p1/M, z18.s\n"
      "st1h { z17.h }, p1, [x23]\n"
      "st1h { z16.h }, p1, [x23, #1, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #2\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26"
    );
}

} // anonymous namespace
template<>
void Transform<2, 4, true, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_2VL_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}


#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
