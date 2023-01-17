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
void interleave_block<4, 4, VLType::SME, true>(
  int8_t * &out, const int8_t * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov z24.b, #0x1\n"
      "mov z23.s, #0x0\n"
      "ptrue p2.b\n"
      "mov z22.s, #0x0\n"
      "cntw x16\n"
      "mov z21.s, #0x0\n"
      "cntw x15, ALL, MUL #2\n"
      "mov z20.s, #0x0\n"
      "cntw x14, ALL, MUL #3\n"
      "cntb x11\n"
      "ptrue p11.s\n"
      "cntw x10\n"
      "cmp %x[height], x10\n"
      "csel x10, %x[height], x10, LT\n"
      "sub x10, x10, #0x1\n"
      "whilelt p10.b, XZR, %x[height]\n"
      "whilelt p9.b, x16, %x[height]\n"
      "whilelt p8.b, x15, %x[height]\n"
      "zip1 p10.b, p10.b, p8.b\n"
      "whilelt p8.b, x14, %x[height]\n"
      "zip1 p9.b, p9.b, p8.b\n"
      "mov x9, %x[row_offset]\n"
      "mov x28, %x[out]\n"
      "zip1 p10.b, p10.b, p9.b\n"
      "cbnz %x[first], 1f\n"
      "addvl x28, x28, #-4\n"
      "ld1w { z23.s }, p2/Z, [x28]\n"
      "ld1w { z22.s }, p2/Z, [x28, #1, MUL VL]\n"
      "ld1w { z21.s }, p2/Z, [x28, #2, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x28, #3, MUL VL]\n"
      "1:"  // Initialise row sums: End
      "mov x27, #0x0\n"
      "whilelt p9.b, x27, %x[width]\n"
      "whilelt p8.b, x27, %x[width]\n"
      "2:"  // Width loop
      "mov x13, #0x0\n"
      "add x26, %x[in], XZR, LSL #3\n"
      "add x25, %x[in], x16, LSL #3\n"
      "add x24, %x[in], x15, LSL #3\n"
      "add x23, %x[in], x14, LSL #3\n"
      "ldr x22, [x26], #0x8\n"
      "ldr x21, [x25], #0x8\n"
      "ldr x19, [x24], #0x8\n"
      "ldr x20, [x23], #0x8\n"
      "cbz x10, 4f\n"
      "3:"  // Loads: Loop
      ".inst 0x25256140  // dup p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0xe00922c0  // ld1b { za0h.b[x13] }, p0/Z, [x22, x9]\n"
      ".inst 0x252d6140  // dup p0.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // dup p1.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe00922a1  // ld1b { za0h.b[x13, #1] }, p0/Z, [x21, x9]\n"
      ".inst 0x253d6140  // dup p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "ldr x22, [x26], #0x8\n"
      ".inst 0xe0092662  // ld1b { za0h.b[x13, #2] }, p1/Z, [x19, x9]\n"
      "ldr x21, [x25], #0x8\n"
      "ldr x19, [x24], #0x8\n"
      ".inst 0xe0092283  // ld1b { za0h.b[x13, #3] }, p0/Z, [x20, x9]\n"
      "ldr x20, [x23], #0x8\n"
      "add x13, x13, #0x4\n"
      "cmp x13, x10, LSL #2\n"
      "blt 3b\n"
      "4:"  // Loads: Tail
      ".inst 0x25256140  // dup p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0xe00922c0  // ld1b { za0h.b[x13] }, p0/Z, [x22, x9]\n"
      ".inst 0x252d6140  // dup p0.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // dup p1.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe00922a1  // ld1b { za0h.b[x13, #1] }, p0/Z, [x21, x9]\n"
      ".inst 0x253d6140  // dup p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "mov x12, #0x0\n"
      ".inst 0xe0092662  // ld1b { za0h.b[x13, #2] }, p1/Z, [x19, x9]\n"
      "sub x19, %x[width], x27\n"
      "cmp x19, x11\n"
      ".inst 0xe0092283  // ld1b { za0h.b[x13, #3] }, p0/Z, [x20, x9]\n"
      "csel x19, x19, x11, LT\n"
      "add x19, x19, #0x3\n"
      "lsr x19, x19, #0x2\n"
      "5:"  // Stores: Loop
      ".inst 0xc0828813  // mova z19.s, p2/M, za0v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8380  // st1w { za0v.s[x12] }, p0/Z, [x28, XZR, LSL #2]\n"
      ".inst 0xc0828892  // mova z18.s, p2/M, za1v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828911  // mova z17.s, p2/M, za2v.s[x12]\n"
      ".inst 0xe0b08384  // st1w { za1v.s[x12] }, p0/Z, [x28, x16, LSL #2]\n"
      ".inst 0x25306d21  // dup p1.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828990  // mova z16.s, p2/M, za3v.s[x12]\n"
      "sdot z23.s, z19.b, z24.b\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      "sdot z22.s, z18.b, z24.b\n"
      ".inst 0xe0af8788  // st1w { za2v.s[x12] }, p1/Z, [x28, x15, LSL #2]\n"
      "sdot z21.s, z17.b, z24.b\n"
      "sdot z20.s, z16.b, z24.b\n"
      ".inst 0xe0ae838c  // st1w { za3v.s[x12] }, p0/Z, [x28, x14, LSL #2]\n"
      "addvl x28, x28, #4\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x19\n"
      "blt 5b\n"
      "incb x9\n"
      "incb x27\n"
      "whilelt p9.b, x27, %x[width]\n"
      "whilelt p8.b, x27, %x[width]\n"
      "b.any 2b\n"
      "st1w { z23.s }, p2, [x28]\n"
      "st1w { z22.s }, p2, [x28, #1, MUL VL]\n"
      "st1w { z21.s }, p2, [x28, #2, MUL VL]\n"
      "st1w { z20.s }, p2, [x28, #3, MUL VL]\n"
      "addvl x28, x28, #4\n"
      "mov %x[out], x28\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p8", "p9", "p10", "p11", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
