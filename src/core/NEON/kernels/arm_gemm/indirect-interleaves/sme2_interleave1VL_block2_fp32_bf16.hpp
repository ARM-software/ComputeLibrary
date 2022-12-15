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
void interleave_block<1, 2, VLType::SME, false>(
  bfloat16 * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  ARM_COMPUTE_UNUSED(first);

  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cntw x22, ALL, MUL #2\n"
      "sub x28, %x[width], #0x1\n"
      "cntw x21, ALL, MUL #2\n"
      "sub x20, x22, #0x1\n"
      "whilelt p10.s, XZR, %x[height]\n"
      "add x28, x28, x21\n"
      "ands x27, %x[width], x20\n"
      "udiv x28, x28, x21\n"
      "csel x27, x27, x22, NE\n"
      "mov x26, #0x0\n"
      "and x25, x28, #0x1\n"
      "sub x28, x28, #0x1\n"
      "add x27, x27, #0x1\n"
      "mov x20, %x[width]\n"
      "ptrue p0.b\n"
      "mov x24, %x[outptr_raw]\n"
      "mov x23, %x[row_offset]\n"
      "cntw x22\n"
      "lsr x28, x28, #0x1\n"
      "lsr x27, x27, #0x1\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44751  // whilelt pn9.s, x26, x20, VLx2\n"
      "mov x21, %x[in]\n"
      "1:"  // Width loop: Preamble: Loop
      "ldr x20, [x21], #0x8\n"
      ".inst 0x25306548  // psel p8.s, p9.s/Z, p10.s[w12]\n"
      ".inst 0xa0174286  // ld1w { z6.s-z7.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xc160e0c6  // bfcvt z6.h, { z6.s-z7.s }\n"
      ".inst 0xc08000c0  // mova za0h.s[x12], p0/M, z6.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x22\n"
      "blt 1b\n"
      "incw x23, ALL, MUL #2\n"
      "incw x26, ALL, MUL #2\n"
      "cbz x28, 5f\n"
      "2:"  // Width loop
      "mov x20, %x[width]\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44751  // whilelt pn9.s, x26, x20, VLx2\n"
      "mov x21, %x[in]\n"
      "3:"  // Width loop: Odd: Loop
      "ldr x20, [x21], #0x8\n"
      ".inst 0x25306548  // psel p8.s, p9.s/Z, p10.s[w12]\n"
      ".inst 0xa017429e  // ld1w { z30.s-z31.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xc160e3de  // bfcvt z30.h, { z30.s-z31.s }\n"
      ".inst 0xc08003c8  // mova za2h.s[x12], p0/M, z30.s\n"
      ".inst 0xc082800f  // mova z15.s, p0/M, za0v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x22\n"
      "st1w { z15.s }, p0, [x24]\n"
      "addvl x24, x24, #1\n"
      "blt 3b\n"
      "incw x26, ALL, MUL #2\n"
      "mov x20, %x[width]\n"
      "incw x23, ALL, MUL #2\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44751  // whilelt pn9.s, x26, x20, VLx2\n"
      "mov x21, %x[in]\n"
      "4:"  // Width loop: Even: Loop
      "ldr x20, [x21], #0x8\n"
      ".inst 0x25306548  // psel p8.s, p9.s/Z, p10.s[w12]\n"
      ".inst 0xa0174298  // ld1w { z24.s-z25.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xc160e318  // bfcvt z24.h, { z24.s-z25.s }\n"
      ".inst 0xc0800300  // mova za0h.s[x12], p0/M, z24.s\n"
      ".inst 0xc0828110  // mova z16.s, p0/M, za2v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x22\n"
      "st1w { z16.s }, p0, [x24]\n"
      "addvl x24, x24, #1\n"
      "blt 4b\n"
      "subs x28, x28, #0x1\n"
      "incw x23, ALL, MUL #2\n"
      "incw x26, ALL, MUL #2\n"
      "bgt 2b\n"
      "5:"  // Width loop: Tails
      "cbnz x25, 8f\n"
      "mov x20, %x[width]\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44751  // whilelt pn9.s, x26, x20, VLx2\n"
      "mov x21, %x[in]\n"
      "6:"  // Width loop: Tails: Even: Odd: Loop
      "ldr x20, [x21], #0x8\n"
      ".inst 0x25306548  // psel p8.s, p9.s/Z, p10.s[w12]\n"
      ".inst 0xa017428e  // ld1w { z14.s-z15.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xc160e1ce  // bfcvt z14.h, { z14.s-z15.s }\n"
      ".inst 0xc08001c8  // mova za2h.s[x12], p0/M, z14.s\n"
      ".inst 0xc0828010  // mova z16.s, p0/M, za0v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x22\n"
      "st1w { z16.s }, p0, [x24]\n"
      "addvl x24, x24, #1\n"
      "blt 6b\n"
      "mov x12, #0x0\n"
      "7:"  // Width loop: Tails: Even: Even: Loop
      ".inst 0xc0828110  // mova z16.s, p0/M, za2v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x27\n"
      "st1w { z16.s }, p0, [x24]\n"
      "addvl x24, x24, #1\n"
      "blt 7b\n"
      "b 10f\n"
      "8:"  // Width loop: Tails: Odd
      "mov x12, #0x0\n"
      "9:"  // Width loop: Tails: Odd: Loop
      ".inst 0xc0828010  // mova z16.s, p0/M, za0v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x27\n"
      "st1w { z16.s }, p0, [x24]\n"
      "addvl x24, x24, #1\n"
      "blt 9b\n"
      "10:"  // End
      "mov %x[outptr_raw], x24\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [outptr_raw] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
