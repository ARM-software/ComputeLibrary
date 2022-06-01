/*
 * Copyright (c) 2022 Arm Limited.
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

#if defined(__ARM_FEATURE_SVE)

template <>
void interleave_block<2, 1, VLType::SME, false>(
  __fp16 * &out, const __fp16 * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  ARM_COMPUTE_UNUSED(first);

  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cnth x27\n"
      "cmp %x[height], x27\n"
      "cnth x26\n"
      "csel x27, %x[height], x27, LT\n"
      "mov x25, #0x0\n"
      "ptrue p13.s\n"
      "sub x27, x27, #0x1\n"
      "whilelt p12.h, XZR, %x[height]\n"
      "whilelt p11.h, x26, %x[height]\n"
      "mov x24, %x[row_offset]\n"
      "mov x23, %x[out]\n"
      "whilelt p10.h, x25, %x[width]\n"
      "whilelt p9.h, x25, %x[width]\n"
      "whilelt p8.h, x25, %x[width]\n"
      "1:"  // Width loop
      "add x22, %x[in], XZR, LSL #3\n"
      "add x19, %x[in], x26, LSL #3\n"
      "ldr x21, [x22], #0x8\n"
      "mov x12, #0x0\n"
      "ldr x20, [x19], #0x8\n"
      "cbz x27, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25286581  // psel p1.h, p9.h/Z, p12.h[w12]\n"
      ".inst 0x25286160  // psel p0.h, p8.h/Z, p11.h[w12]\n"
      ".inst 0xe05806a0  // ld1h { za0h.h[x12] }, p1/Z, [x21, x24, LSL #1]\n"
      "ldr x21, [x22], #0x8\n"
      ".inst 0xe0580288  // ld1h { za1h.h[x12] }, p0/Z, [x20, x24, LSL #1]\n"
      "add x12, x12, #0x2\n"
      "cmp x12, x27, LSL #1\n"
      "ldr x20, [x19], #0x8\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      "sub x19, %x[width], x25\n"
      ".inst 0x25286580  // psel p0.h, p9.h/Z, p12.h[w12]\n"
      ".inst 0xe05802a0  // ld1h { za0h.h[x12] }, p0/Z, [x21, x24, LSL #1]\n"
      ".inst 0x25286160  // psel p0.h, p8.h/Z, p11.h[w12]\n"
      "cmp x19, x26\n"
      ".inst 0xe0580288  // ld1h { za1h.h[x12] }, p0/Z, [x20, x24, LSL #1]\n"
      "mov x12, #0x0\n"
      "csel x19, x19, x26, LT\n"
      "4:"  // Stores: Loop
      ".inst 0x25287540  // psel p0.h, p13.h/Z, p10.h[w12]\n"
      ".inst 0xe07f82e0  // st1h { za0v.h[x12] }, p0/Z, [x23, XZR, LSL #1]\n"
      ".inst 0x25287540  // psel p0.h, p13.h/Z, p10.h[w12]\n"
      ".inst 0xe07a82e8  // st1h { za1v.h[x12] }, p0/Z, [x23, x26, LSL #1]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x19\n"
      "addvl x23, x23, #4\n"
      "blt 4b\n"
      "inch x25\n"
      "whilelt p10.h, x25, %x[width]\n"
      "whilelt p9.h, x25, %x[width]\n"
      "whilelt p8.h, x25, %x[width]\n"
      "inch x24\n"
      "b.any 1b\n"
      "mov %x[out], x23\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x12", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
