/*
 * Copyright (c) 2023 Arm Limited.
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
void interleave_block<4, 2, VLType::SME, false>(
  __fp16 * &out, const __fp16 * const *in,
  size_t width, size_t height, size_t row_offset, bool
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x17, #0x0\n"
      "mov x16, %x[row_offset]\n"
      "cntw x15\n"
      "cntw x14\n"
      "cntw x11, ALL, MUL #2\n"
      "cntw x10, ALL, MUL #3\n"
      "cmp %x[height], x15\n"
      "cnth x9\n"
      "csel x15, %x[height], x15, LT\n"
      "whilelt p11.h, XZR, %x[height]\n"
      "whilelt p10.h, x14, %x[height]\n"
      "whilelt p9.h, x11, %x[height]\n"
      "whilelt p8.h, x10, %x[height]\n"
      "ptrue p13.s\n"
      "sub x15, x15, #0x1\n"
      "zip1 p12.h, p11.h, p9.h\n"
      "zip1 p11.h, p10.h, p8.h\n"
      "mov x28, %x[out]\n"
      "whilelt p10.h, x17, %x[width]\n"
      "whilelt p9.h, x17, %x[width]\n"
      "whilelt p8.h, x17, %x[width]\n"
      "1:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x14, LSL #3\n"
      "add x25, %x[in], x11, LSL #3\n"
      "add x20, %x[in], x10, LSL #3\n"
      "ldr x24, [x27], #0x8\n"
      "mov x13, #0x0\n"
      "ldr x23, [x26], #0x8\n"
      "ldr x22, [x25], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x15, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25296582  // psel p2.h, p9.h/Z, p12.h[w13]\n"
      ".inst 0x25296161  // psel p1.h, p8.h/Z, p11.h[w13]\n"
      ".inst 0x25396580  // psel p0.h, p9.h/Z, p12.h[w13, #1]\n"
      ".inst 0xe0502b00  // ld1h { za0h.h[x13] }, p2/Z, [x24, x16, LSL #1]\n"
      ".inst 0x25396162  // psel p2.h, p8.h/Z, p11.h[w13, #1]\n"
      "ldr x24, [x27], #0x8\n"
      ".inst 0xe05026e8  // ld1h { za1h.h[x13] }, p1/Z, [x23, x16, LSL #1]\n"
      "ldr x23, [x26], #0x8\n"
      ".inst 0xe05022c1  // ld1h { za0h.h[x13, #1] }, p0/Z, [x22, x16, LSL #1]\n"
      "ldr x22, [x25], #0x8\n"
      ".inst 0xe0502aa9  // ld1h { za1h.h[x13, #1] }, p2/Z, [x21, x16, LSL #1]\n"
      "add x13, x13, #0x2\n"
      "ldr x21, [x20], #0x8\n"
      "cmp x13, x15, LSL #1\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      ".inst 0x25296581  // psel p1.h, p9.h/Z, p12.h[w13]\n"
      ".inst 0x25296160  // psel p0.h, p8.h/Z, p11.h[w13]\n"
      "sub x20, %x[width], x17\n"
      ".inst 0x25396582  // psel p2.h, p9.h/Z, p12.h[w13, #1]\n"
      "cmp x20, x9\n"
      "mov x12, #0x0\n"
      ".inst 0xe0502700  // ld1h { za0h.h[x13] }, p1/Z, [x24, x16, LSL #1]\n"
      ".inst 0xe05022e8  // ld1h { za1h.h[x13] }, p0/Z, [x23, x16, LSL #1]\n"
      ".inst 0x25396161  // psel p1.h, p8.h/Z, p11.h[w13, #1]\n"
      "csel x20, x20, x9, LT\n"
      "add x20, x20, #0x1\n"
      ".inst 0xe0502ac1  // ld1h { za0h.h[x13, #1] }, p2/Z, [x22, x16, LSL #1]\n"
      "lsr x20, x20, #0x1\n"
      ".inst 0xe05026a9  // ld1h { za1h.h[x13, #1] }, p1/Z, [x21, x16, LSL #1]\n"
      "4:"  // Stores: Loop
      ".inst 0x25307540  // psel p0.s, p13.s/Z, p10.s[w12]\n"
      ".inst 0x25307542  // psel p2.s, p13.s/Z, p10.s[w12]\n"
      ".inst 0x25307541  // psel p1.s, p13.s/Z, p10.s[w12]\n"
      ".inst 0xe0bf8380  // st1w { za0v.s[x12] }, p0/Z, [x28, XZR, LSL #2]\n"
      ".inst 0x25307540  // psel p0.s, p13.s/Z, p10.s[w12]\n"
      ".inst 0xe0ae8b84  // st1w { za1v.s[x12] }, p2/Z, [x28, x14, LSL #2]\n"
      ".inst 0xe0ab8788  // st1w { za2v.s[x12] }, p1/Z, [x28, x11, LSL #2]\n"
      ".inst 0xe0aa838c  // st1w { za3v.s[x12] }, p0/Z, [x28, x10, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "addvl x28, x28, #4\n"
      "cmp x12, x20\n"
      "blt 4b\n"
      "inch x17\n"
      "inch x16\n"
      "whilelt p10.h, x17, %x[width]\n"
      "whilelt p9.h, x17, %x[width]\n"
      "whilelt p8.h, x17, %x[width]\n"
      "b.any 1b\n"
      "mov %x[out], x28\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
