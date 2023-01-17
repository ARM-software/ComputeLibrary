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
void interleave_block<1, 4, VLType::SME, true>(
  uint8_t * &out, const uint8_t * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov z18.b, #0x1\n"
      "mov z17.s, #0x0\n"
      "cntb x20\n"
      "cntw x10\n"
      "ptrue p1.b\n"
      "mov x19, %x[width]\n"
      "incb x19\n"
      "sub x19, x19, #0x1\n"
      "udiv x19, x19, x20\n" // n_passes = ceildiv(width, VL<T>)
      "sub x9, x19, #0x1\n"
      "lsr x9, x9, #0x1\n" // n_loops = (n_passes - 1) / 2
      "and x28, x19, #0x1\n" // odd_tail = bool(n_passes & 0x1)
      "mov x19, %x[width]\n"
      "sub x27, x20, #0x1\n"
      "ands x27, x19, x27\n"
      "csel x27, x27, x20, NE\n"
      "add x27, x27, #0x3\n"
      "lsr x27, x27, #0x2\n"
      "sub x26, x10, #0x2\n"
      "ptrue p11.s\n"
      "lsl x20, %x[height], #0x1\n" // height * 2
      "lsl x19, x10, #0x1\n"
      "whilelt p9.b, XZR, x20\n"
      "whilelt p8.b, x19, x20\n"
      "zip1 p10.b, p9.b, p8.b\n"
      "mov x25, %x[row_offset]\n"
      "mov x24, %x[out]\n"
      "mov x23, #0x0\n"
      "whilelt p9.b, x23, %x[width]\n"
      "whilelt p8.b, x23, %x[width]\n"
      "cbnz %x[first], 1f\n"
      "addvl x24, x24, #-1\n"
      "ld1w { z17.s }, p1/Z, [x24]\n"
      "1:"  // K loop: Load row sums: End
      "mov x22, %x[in]\n"
      "ldr x21, [x22, #0x0]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      "mov x12, #0x0\n"
      "cbz x26, 3f\n"
      "2:"  // K loop: Charge: Loop
      ".inst 0x25246140  // dup p0.b, p8.b/Z, p10.b[w12]\n"
      ".inst 0xe01902a0  // ld1b { za0h.b[x12] }, p0/Z, [x21, x25]\n"
      ".inst 0x25646140  // dup p0.b, p8.b/Z, p10.b[w12, #4]\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe0190284  // ld1b { za0h.b[x12, #4] }, p0/Z, [x20, x25]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      "add x12, x12, #0x8\n"
      "cmp x12, x26, LSL #2\n"
      "blt 2b\n"
      "3:"  // K loop: Charge: End
      ".inst 0x25246140  // dup p0.b, p8.b/Z, p10.b[w12]\n"
      ".inst 0xe01902a0  // ld1b { za0h.b[x12] }, p0/Z, [x21, x25]\n"
      ".inst 0x25646140  // dup p0.b, p8.b/Z, p10.b[w12, #4]\n"
      "mov x22, %x[in]\n"
      ".inst 0xe0190284  // ld1b { za0h.b[x12, #4] }, p0/Z, [x20, x25]\n"
      "ldr x21, [x22, #0x0]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      "incb x25\n"
      "incb x23\n"
      "cbz x9, 9f\n"
      "mov x19, x9\n"
      "4:"  // K loop: Main loop
      "whilelt p8.b, x23, %x[width]\n"
      "mov x13, #0x0\n"
      "mov x12, #0x0\n"
      "cbz x26, 6f\n"
      "5:"  // K loop: Main loop: First: Loop
      ".inst 0x25356140  // dup p0.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe01922a2  // ld1b { za0h.b[x13, #2] }, p0/Z, [x21, x25]\n"
      ".inst 0x25756140  // dup p0.b, p8.b/Z, p10.b[w13, #6]\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe0192286  // ld1b { za0h.b[x13, #6] }, p0/Z, [x20, x25]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      ".inst 0xc0828410  // mova z16.s, p1/M, za0v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8300  // st1w { za0v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      ".inst 0x25706d20  // dup p0.s, p11.s/Z, p9.s[w12, #1]\n"
      "add x13, x13, #0x8\n"
      ".inst 0xe0aa8301  // st1w { za0v.s[x12, #1] }, p0/Z, [x24, x10, LSL #2]\n"
      "udot z17.s, z16.b, z18.b\n"
      ".inst 0xc0828430  // mova z16.s, p1/M, za0v.s[x12, #1]\n"
      "addvl x24, x24, #2\n"
      "add x12, x12, #0x2\n"
      "cmp x12, x26\n"
      "udot z17.s, z16.b, z18.b\n"
      "blt 5b\n"
      "6:"  // K loop: Main loop: First: Tail
      "mov x22, %x[in]\n"
      ".inst 0x25356140  // dup p0.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe01922a2  // ld1b { za0h.b[x13, #2] }, p0/Z, [x21, x25]\n"
      ".inst 0x25756140  // dup p0.b, p8.b/Z, p10.b[w13, #6]\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe0192286  // ld1b { za0h.b[x13, #6] }, p0/Z, [x20, x25]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      ".inst 0xc0828410  // mova z16.s, p1/M, za0v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8300  // st1w { za0v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      ".inst 0x25706d20  // dup p0.s, p11.s/Z, p9.s[w12, #1]\n"
      "whilelt p9.b, x23, %x[width]\n"
      ".inst 0xe0aa8301  // st1w { za0v.s[x12, #1] }, p0/Z, [x24, x10, LSL #2]\n"
      "udot z17.s, z16.b, z18.b\n"
      ".inst 0xc0828430  // mova z16.s, p1/M, za0v.s[x12, #1]\n"
      "addvl x24, x24, #2\n"
      "incb x23\n"
      "incb x25\n"
      "udot z17.s, z16.b, z18.b\n"
      "whilelt p8.b, x23, %x[width]\n"
      "mov x13, #0x0\n"
      "mov x12, #0x0\n"
      "cbz x26, 8f\n"
      "7:"  // K loop: Main loop: Second: Loop
      ".inst 0x25256140  // dup p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0xe01922a0  // ld1b { za0h.b[x13] }, p0/Z, [x21, x25]\n"
      ".inst 0x25656140  // dup p0.b, p8.b/Z, p10.b[w13, #4]\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe0192284  // ld1b { za0h.b[x13, #4] }, p0/Z, [x20, x25]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      ".inst 0xc0828510  // mova z16.s, p1/M, za2v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8308  // st1w { za2v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      ".inst 0x25706d20  // dup p0.s, p11.s/Z, p9.s[w12, #1]\n"
      "add x13, x13, #0x8\n"
      ".inst 0xe0aa8309  // st1w { za2v.s[x12, #1] }, p0/Z, [x24, x10, LSL #2]\n"
      "udot z17.s, z16.b, z18.b\n"
      ".inst 0xc0828530  // mova z16.s, p1/M, za2v.s[x12, #1]\n"
      "addvl x24, x24, #2\n"
      "add x12, x12, #0x2\n"
      "cmp x12, x26\n"
      "udot z17.s, z16.b, z18.b\n"
      "blt 7b\n"
      "8:"  // K loop: Main loop: Second: Tail
      "mov x22, %x[in]\n"
      ".inst 0x25256140  // dup p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0xe01922a0  // ld1b { za0h.b[x13] }, p0/Z, [x21, x25]\n"
      ".inst 0x25656140  // dup p0.b, p8.b/Z, p10.b[w13, #4]\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe0192284  // ld1b { za0h.b[x13, #4] }, p0/Z, [x20, x25]\n"
      "ldr x20, [x22, #0x8]\n"
      "add x22, x22, #0x10\n"
      ".inst 0xc0828510  // mova z16.s, p1/M, za2v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8308  // st1w { za2v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      ".inst 0x25706d20  // dup p0.s, p11.s/Z, p9.s[w12, #1]\n"
      "whilelt p9.b, x23, %x[width]\n"
      ".inst 0xe0aa8309  // st1w { za2v.s[x12, #1] }, p0/Z, [x24, x10, LSL #2]\n"
      "udot z17.s, z16.b, z18.b\n"
      ".inst 0xc0828530  // mova z16.s, p1/M, za2v.s[x12, #1]\n"
      "addvl x24, x24, #2\n"
      "incb x23\n"
      "incb x25\n"
      "udot z17.s, z16.b, z18.b\n"
      "subs x19, x19, #0x1\n"
      "bgt 4b\n"
      "9:"  // K loop: Tails
      "cbnz x28, 12f\n"
      "mov x22, %x[in]\n"
      "whilelt p8.b, x23, %x[width]\n"
      "mov x13, #0x0\n"
      "mov x12, #0x0\n"
      "10:"  // K loop: Tails: Even: First
      ".inst 0xc0828410  // mova z16.s, p1/M, za0v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8300  // st1w { za0v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      ".inst 0x25356140  // dup p0.b, p8.b/Z, p10.b[w13, #2]\n"
      "addvl x24, x24, #1\n"
      "ldr x21, [x22, #0x0]\n"
      ".inst 0xe01922a2  // ld1b { za0h.b[x13, #2] }, p0/Z, [x21, x25]\n"
      "udot z17.s, z16.b, z18.b\n"
      "add x22, x22, #0x8\n"
      "add x13, x13, #0x4\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x10\n"
      "blt 10b\n"
      "whilelt p9.b, x23, %x[width]\n"
      "whilelt p8.b, x23, %x[width]\n"
      "mov x19, #0x0\n"
      "mov x12, #0x0\n"
      "11:"  // K loop: Tails: Even: Second
      ".inst 0xc0828510  // mova z16.s, p1/M, za2v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8308  // st1w { za2v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      "addvl x24, x24, #1\n"
      "add x19, x19, #0x4\n"
      "add x12, x12, #0x1\n"
      "udot z17.s, z16.b, z18.b\n"
      "cmp x12, x27\n"
      "blt 11b\n"
      "whilelt p9.b, x23, %x[width]\n"
      "b 14f\n"
      "12:"  // K loop: Tails: Odd
      "mov x12, #0x0\n"
      "13:"  // K loop: Tails: Odd: Loop
      ".inst 0xc0828410  // mova z16.s, p1/M, za0v.s[x12]\n"
      ".inst 0x25306d20  // dup p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8300  // st1w { za0v.s[x12] }, p0/Z, [x24, XZR, LSL #2]\n"
      "addvl x24, x24, #1\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x27\n"
      "udot z17.s, z16.b, z18.b\n"
      "blt 13b\n"
      "14:"  // K loop: End
      "st1w { z17.s }, p1, [x24]\n"
      "addvl x24, x24, #1\n"
      "mov %x[out], x24\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p8", "p9", "p10", "p11", "x9", "x10", "x12", "x13", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
