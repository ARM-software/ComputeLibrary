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

void sve_transpose_interleave_8VL(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 8 * height * get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "cmp %x[height], #0x2\n"
      "ptrue p1.b\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "cntw x24, ALL, MUL #16\n"
      "add x23, x26, %x[in_stride]\n"
      "cmp x25, x24\n"
      "add %x[in], x23, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x2\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z15.s }, p1/Z, [x26]\n"
      "ld1w { z14.s }, p1/Z, [x26, #1, MUL VL]\n"
      "mov x21, x22\n"
      "add x22, x22, %x[out_stride]\n"
      "ld1w { z13.s }, p1/Z, [x26, #2, MUL VL]\n"
      "ld1w { z12.s }, p1/Z, [x26, #3, MUL VL]\n"
      "mov x20, x22\n"
      "sub x25, x25, x24\n"
      "ld1w { z11.s }, p1/Z, [x26, #4, MUL VL]\n"
      "ld1w { z10.s }, p1/Z, [x26, #5, MUL VL]\n"
      "cmp x25, x24\n"
      "add x22, x22, %x[out_stride]\n"
      "ld1w { z9.s }, p1/Z, [x26, #6, MUL VL]\n"
      "ld1w { z8.s }, p1/Z, [x26, #7, MUL VL]\n"
      "addvl x26, x26, #16\n"
      "ld1w { z7.s }, p1/Z, [x23]\n"
      "ld1w { z6.s }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1w { z5.s }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1w { z4.s }, p1/Z, [x23, #3, MUL VL]\n"
      "ld1w { z3.s }, p1/Z, [x23, #4, MUL VL]\n"
      "ld1w { z2.s }, p1/Z, [x23, #5, MUL VL]\n"
      "ld1w { z1.s }, p1/Z, [x23, #6, MUL VL]\n"
      "ld1w { z0.s }, p1/Z, [x23, #7, MUL VL]\n"
      "addvl x23, x23, #16\n"
      "ld1w { z31.s }, p1/Z, [x26, #-8, MUL VL]\n"
      "ld1w { z30.s }, p1/Z, [x26, #-7, MUL VL]\n"
      "ld1w { z29.s }, p1/Z, [x26, #-6, MUL VL]\n"
      "ld1w { z28.s }, p1/Z, [x26, #-5, MUL VL]\n"
      "ld1w { z27.s }, p1/Z, [x26, #-4, MUL VL]\n"
      "ld1w { z26.s }, p1/Z, [x26, #-3, MUL VL]\n"
      "ld1w { z25.s }, p1/Z, [x26, #-2, MUL VL]\n"
      "ld1w { z24.s }, p1/Z, [x26, #-1, MUL VL]\n"
      "ld1w { z23.s }, p1/Z, [x23, #-8, MUL VL]\n"
      "ld1w { z22.s }, p1/Z, [x23, #-7, MUL VL]\n"
      "ld1w { z21.s }, p1/Z, [x23, #-6, MUL VL]\n"
      "ld1w { z20.s }, p1/Z, [x23, #-5, MUL VL]\n"
      "ld1w { z19.s }, p1/Z, [x23, #-4, MUL VL]\n"
      "ld1w { z18.s }, p1/Z, [x23, #-3, MUL VL]\n"
      "ld1w { z17.s }, p1/Z, [x23, #-2, MUL VL]\n"
      "ld1w { z16.s }, p1/Z, [x23, #-1, MUL VL]\n"
      "st1w { z15.s }, p1, [x21]\n"
      "st1w { z14.s }, p1, [x21, #1, MUL VL]\n"
      "st1w { z13.s }, p1, [x21, #2, MUL VL]\n"
      "st1w { z12.s }, p1, [x21, #3, MUL VL]\n"
      "st1w { z11.s }, p1, [x21, #4, MUL VL]\n"
      "st1w { z10.s }, p1, [x21, #5, MUL VL]\n"
      "st1w { z9.s }, p1, [x21, #6, MUL VL]\n"
      "st1w { z8.s }, p1, [x21, #7, MUL VL]\n"
      "addvl x21, x21, #16\n"
      "st1w { z7.s }, p1, [x21, #-8, MUL VL]\n"
      "st1w { z6.s }, p1, [x21, #-7, MUL VL]\n"
      "st1w { z5.s }, p1, [x21, #-6, MUL VL]\n"
      "st1w { z4.s }, p1, [x21, #-5, MUL VL]\n"
      "st1w { z3.s }, p1, [x21, #-4, MUL VL]\n"
      "st1w { z2.s }, p1, [x21, #-3, MUL VL]\n"
      "st1w { z1.s }, p1, [x21, #-2, MUL VL]\n"
      "st1w { z0.s }, p1, [x21, #-1, MUL VL]\n"
      "st1w { z31.s }, p1, [x20]\n"
      "st1w { z30.s }, p1, [x20, #1, MUL VL]\n"
      "st1w { z29.s }, p1, [x20, #2, MUL VL]\n"
      "st1w { z28.s }, p1, [x20, #3, MUL VL]\n"
      "st1w { z27.s }, p1, [x20, #4, MUL VL]\n"
      "st1w { z26.s }, p1, [x20, #5, MUL VL]\n"
      "st1w { z25.s }, p1, [x20, #6, MUL VL]\n"
      "st1w { z24.s }, p1, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1w { z23.s }, p1, [x20, #-8, MUL VL]\n"
      "st1w { z22.s }, p1, [x20, #-7, MUL VL]\n"
      "st1w { z21.s }, p1, [x20, #-6, MUL VL]\n"
      "st1w { z20.s }, p1, [x20, #-5, MUL VL]\n"
      "st1w { z19.s }, p1, [x20, #-4, MUL VL]\n"
      "st1w { z18.s }, p1, [x20, #-3, MUL VL]\n"
      "st1w { z17.s }, p1, [x20, #-2, MUL VL]\n"
      "st1w { z16.s }, p1, [x20, #-1, MUL VL]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x25\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z31.s }, p0/Z, [x26]\n"
      "ld1w { z30.s }, p0/Z, [x23]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z29.s }, p0/Z, [x26, #1, MUL VL]\n"
      "ld1w { z28.s }, p0/Z, [x23, #1, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z27.s }, p0/Z, [x26, #2, MUL VL]\n"
      "ld1w { z26.s }, p0/Z, [x23, #2, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z25.s }, p0/Z, [x26, #3, MUL VL]\n"
      "ld1w { z24.s }, p0/Z, [x23, #3, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z23.s }, p0/Z, [x26, #4, MUL VL]\n"
      "ld1w { z22.s }, p0/Z, [x23, #4, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z21.s }, p0/Z, [x26, #5, MUL VL]\n"
      "ld1w { z20.s }, p0/Z, [x23, #5, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z19.s }, p0/Z, [x26, #6, MUL VL]\n"
      "ld1w { z18.s }, p0/Z, [x23, #6, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z17.s }, p0/Z, [x26, #7, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x23, #7, MUL VL]\n"
      "mov x20, x22\n"
      "decw x25, ALL, MUL #8\n"
      "st1w { z31.s }, p1, [x20]\n"
      "st1w { z29.s }, p1, [x20, #1, MUL VL]\n"
      "cmp x25, #0x0\n"
      "addvl x26, x26, #8\n"
      "st1w { z27.s }, p1, [x20, #2, MUL VL]\n"
      "addvl x23, x23, #8\n"
      "add x22, x22, %x[out_stride]\n"
      "st1w { z25.s }, p1, [x20, #3, MUL VL]\n"
      "st1w { z23.s }, p1, [x20, #4, MUL VL]\n"
      "st1w { z21.s }, p1, [x20, #5, MUL VL]\n"
      "st1w { z19.s }, p1, [x20, #6, MUL VL]\n"
      "st1w { z17.s }, p1, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1w { z30.s }, p1, [x20, #-8, MUL VL]\n"
      "st1w { z28.s }, p1, [x20, #-7, MUL VL]\n"
      "st1w { z26.s }, p1, [x20, #-6, MUL VL]\n"
      "st1w { z24.s }, p1, [x20, #-5, MUL VL]\n"
      "st1w { z22.s }, p1, [x20, #-4, MUL VL]\n"
      "st1w { z20.s }, p1, [x20, #-3, MUL VL]\n"
      "st1w { z18.s }, p1, [x20, #-2, MUL VL]\n"
      "st1w { z16.s }, p1, [x20, #-1, MUL VL]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp %x[height], #0x2\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      "cbz %x[height], 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x21, %x[width]\n"
      "cntw x20, ALL, MUL #16\n"
      "mov x26, %x[in]\n"
      "cmp x21, x20\n"
      "add %x[in], x26, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "sub %x[height], %x[height], #0x1\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1w { z31.s }, p1/Z, [x26]\n"
      "ld1w { z30.s }, p1/Z, [x26, #1, MUL VL]\n"
      "sub x21, x21, x20\n"
      "cmp x21, x20\n"
      "ld1w { z29.s }, p1/Z, [x26, #2, MUL VL]\n"
      "ld1w { z28.s }, p1/Z, [x26, #3, MUL VL]\n"
      "ld1w { z27.s }, p1/Z, [x26, #4, MUL VL]\n"
      "ld1w { z26.s }, p1/Z, [x26, #5, MUL VL]\n"
      "ld1w { z25.s }, p1/Z, [x26, #6, MUL VL]\n"
      "ld1w { z24.s }, p1/Z, [x26, #7, MUL VL]\n"
      "addvl x26, x26, #16\n"
      "ld1w { z23.s }, p1/Z, [x26, #-8, MUL VL]\n"
      "ld1w { z22.s }, p1/Z, [x26, #-7, MUL VL]\n"
      "ld1w { z21.s }, p1/Z, [x26, #-6, MUL VL]\n"
      "ld1w { z20.s }, p1/Z, [x26, #-5, MUL VL]\n"
      "ld1w { z19.s }, p1/Z, [x26, #-4, MUL VL]\n"
      "ld1w { z18.s }, p1/Z, [x26, #-3, MUL VL]\n"
      "ld1w { z17.s }, p1/Z, [x26, #-2, MUL VL]\n"
      "ld1w { z16.s }, p1/Z, [x26, #-1, MUL VL]\n"
      "st1w { z31.s }, p1, [x22]\n"
      "st1w { z30.s }, p1, [x22, #1, MUL VL]\n"
      "st1w { z29.s }, p1, [x22, #2, MUL VL]\n"
      "st1w { z28.s }, p1, [x22, #3, MUL VL]\n"
      "st1w { z27.s }, p1, [x22, #4, MUL VL]\n"
      "st1w { z26.s }, p1, [x22, #5, MUL VL]\n"
      "st1w { z25.s }, p1, [x22, #6, MUL VL]\n"
      "st1w { z24.s }, p1, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1w { z23.s }, p1, [x22]\n"
      "st1w { z22.s }, p1, [x22, #1, MUL VL]\n"
      "st1w { z21.s }, p1, [x22, #2, MUL VL]\n"
      "st1w { z20.s }, p1, [x22, #3, MUL VL]\n"
      "st1w { z19.s }, p1, [x22, #4, MUL VL]\n"
      "st1w { z18.s }, p1, [x22, #5, MUL VL]\n"
      "st1w { z17.s }, p1, [x22, #6, MUL VL]\n"
      "st1w { z16.s }, p1, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z23.s }, p0/Z, [x26]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z22.s }, p0/Z, [x26, #1, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z21.s }, p0/Z, [x26, #2, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z20.s }, p0/Z, [x26, #3, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z19.s }, p0/Z, [x26, #4, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z18.s }, p0/Z, [x26, #5, MUL VL]\n"
      "decw x20\n"
      "whilelt p0.s, XZR, x20\n"
      "ld1w { z17.s }, p0/Z, [x26, #6, MUL VL]\n"
      "decw x20\n"
      "decw x21, ALL, MUL #8\n"
      "whilelt p0.s, XZR, x20\n"
      "cmp x21, #0x0\n"
      "ld1w { z16.s }, p0/Z, [x26, #7, MUL VL]\n"
      "st1w { z23.s }, p1, [x22]\n"
      "addvl x26, x26, #8\n"
      "st1w { z22.s }, p1, [x22, #1, MUL VL]\n"
      "st1w { z21.s }, p1, [x22, #2, MUL VL]\n"
      "st1w { z20.s }, p1, [x22, #3, MUL VL]\n"
      "st1w { z19.s }, p1, [x22, #4, MUL VL]\n"
      "st1w { z18.s }, p1, [x22, #5, MUL VL]\n"
      "st1w { z17.s }, p1, [x22, #6, MUL VL]\n"
      "st1w { z16.s }, p1, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp %x[height], #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 7b\n"
      "12:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<8, 1, true, VLType::SVE>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_8VL(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif
