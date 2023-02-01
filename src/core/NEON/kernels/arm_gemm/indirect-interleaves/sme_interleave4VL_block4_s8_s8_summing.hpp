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

#if defined(__ARM_FEATURE_SVE)

template <>
void interleave_block<4, 4, VLType::SME, true>(
  int8_t * &out, const int8_t * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cntw x16\n"
      "cntw x15\n"
      "mov z24.b, #0x1\n"
      "cntw x14, ALL, MUL #2\n"
      "cntw x13, ALL, MUL #3\n"
      "mov z23.s, #0x0\n"
      "mov z22.s, #0x0\n"
      "cmp %x[height], x16\n"
      "csel x16, %x[height], x16, LT\n"
      "mov z21.s, #0x0\n"
      "mov z20.s, #0x0\n"
      "whilelt p12.b, XZR, %x[height]\n"
      "whilelt p10.b, x15, %x[height]\n"
      "whilelt p9.b, x14, %x[height]\n"
      "whilelt p8.b, x13, %x[height]\n"
      "zip1 p12.b, p12.b, p9.b\n"
      "zip1 p10.b, p10.b, p8.b\n"
      "ptrue p2.b\n"
      "cntb x11\n"
      "ptrue p11.s\n"
      "sub x16, x16, #0x1\n"
      "zip1 p10.b, p12.b, p10.b\n"
      "mov x10, %x[row_offset]\n"
      "mov x9, %x[out]\n"
      "cbnz %x[first], 1f\n"
      "addvl x9, x9, #-4\n"
      "ld1w { z23.s }, p2/Z, [x9]\n"
      "ld1w { z22.s }, p2/Z, [x9, #1, MUL VL]\n"
      "ld1w { z21.s }, p2/Z, [x9, #2, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x9, #3, MUL VL]\n"
      "1:"  // Initialise row sums: End
      "mov x28, #0x0\n"
      "whilelt p9.b, x28, %x[width]\n"
      "whilelt p8.b, x28, %x[width]\n"
      "2:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x15, LSL #3\n"
      "ldr x25, [x27], #0x8\n"
      "add x24, %x[in], x14, LSL #3\n"
      "add x23, %x[in], x13, LSL #3\n"
      "ldr x20, [x26], #0x8\n"
      "mov x12, #0x0\n"
      "ldr x22, [x24], #0x8\n"
      "ldr x21, [x23], #0x8\n"
      "cbz x16, 4f\n"
      "3:"  // Loads: Loop
      ".inst 0x25246140  // psel p0.b, p8.b/Z, p10.b[w12]\n"
      ".inst 0xe00a0320  // ld1b { za0h.b[x12] }, p0/Z, [x25, x10]\n"
      ".inst 0x252c6140  // psel p0.b, p8.b/Z, p10.b[w12, #1]\n"
      "ldr x25, [x27], #0x8\n"
      ".inst 0xe00a0281  // ld1b { za0h.b[x12, #1] }, p0/Z, [x20, x10]\n"
      ".inst 0x25346141  // psel p1.b, p8.b/Z, p10.b[w12, #2]\n"
      ".inst 0x253c6140  // psel p0.b, p8.b/Z, p10.b[w12, #3]\n"
      "ldr x20, [x26], #0x8\n"
      ".inst 0xe00a06c2  // ld1b { za0h.b[x12, #2] }, p1/Z, [x22, x10]\n"
      "ldr x22, [x24], #0x8\n"
      ".inst 0xe00a02a3  // ld1b { za0h.b[x12, #3] }, p0/Z, [x21, x10]\n"
      "add x12, x12, #0x4\n"
      "cmp x12, x16, LSL #2\n"
      "ldr x21, [x23], #0x8\n"
      "blt 3b\n"
      "4:"  // Loads: Tail
      ".inst 0x25246140  // psel p0.b, p8.b/Z, p10.b[w12]\n"
      ".inst 0xe00a0320  // ld1b { za0h.b[x12] }, p0/Z, [x25, x10]\n"
      ".inst 0x252c6140  // psel p0.b, p8.b/Z, p10.b[w12, #1]\n"
      ".inst 0xe00a0281  // ld1b { za0h.b[x12, #1] }, p0/Z, [x20, x10]\n"
      ".inst 0x25346140  // psel p0.b, p8.b/Z, p10.b[w12, #2]\n"
      "sub x20, %x[width], x28\n"
      ".inst 0xe00a02c2  // ld1b { za0h.b[x12, #2] }, p0/Z, [x22, x10]\n"
      "cmp x20, x11\n"
      "csel x20, x20, x11, LT\n"
      ".inst 0x253c6140  // psel p0.b, p8.b/Z, p10.b[w12, #3]\n"
      "add x20, x20, #0x3\n"
      ".inst 0xe00a02a3  // ld1b { za0h.b[x12, #3] }, p0/Z, [x21, x10]\n"
      "mov x12, #0x0\n"
      "lsr x20, x20, #0x2\n"
      "5:"  // Stores: Loop
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8120  // st1w { za0v.s[x12] }, p0/Z, [x9, XZR, LSL #2]\n"
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828812  // mova z18.s, p2/M, za0v.s[x12]\n"
      ".inst 0xe0af8124  // st1w { za1v.s[x12] }, p0/Z, [x9, x15, LSL #2]\n"
      ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828891  // mova z17.s, p2/M, za1v.s[x12]\n"
      ".inst 0xe0ae8528  // st1w { za2v.s[x12] }, p1/Z, [x9, x14, LSL #2]\n"
      ".inst 0xc0828910  // mova z16.s, p2/M, za2v.s[x12]\n"
      "sdot z23.s, z18.b, z24.b\n"
      ".inst 0xe0ad812c  // st1w { za3v.s[x12] }, p0/Z, [x9, x13, LSL #2]\n"
      ".inst 0xc0828993  // mova z19.s, p2/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x20\n"
      "sdot z22.s, z17.b, z24.b\n"
      "sdot z21.s, z16.b, z24.b\n"
      "addvl x9, x9, #4\n"
      "sdot z20.s, z19.b, z24.b\n"
      "blt 5b\n"
      "incb x28\n"
      "whilelt p9.b, x28, %x[width]\n"
      "whilelt p8.b, x28, %x[width]\n"
      "incb x10\n"
      "b.any 2b\n"
      "st1w { z23.s }, p2, [x9]\n"
      "st1w { z22.s }, p2, [x9, #1, MUL VL]\n"
      "st1w { z21.s }, p2, [x9, #2, MUL VL]\n"
      "st1w { z20.s }, p2, [x9, #3, MUL VL]\n"
      "addvl x9, x9, #4\n"
      "mov %x[out], x9\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
