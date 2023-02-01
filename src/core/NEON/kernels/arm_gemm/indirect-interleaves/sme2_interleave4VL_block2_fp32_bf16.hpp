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
void interleave_block<4, 2, VLType::SME, false>(
  bfloat16 * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  ARM_COMPUTE_UNUSED(first);

  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cntw x23, ALL, MUL #2\n"
      "cntw x10\n"
      "cntw x22, ALL, MUL #2\n"
      "cntw x20, ALL, MUL #3\n"
      "sub x21, x23, #0x1\n"
      ".inst 0x25207817  // ptrue pn15.b\n"
      "whilelt p1.s, XZR, %x[height]\n"
      "whilelt p14.s, x10, %x[height]\n"
      "whilelt p13.s, x22, %x[height]\n"
      "whilelt p12.s, x20, %x[height]\n"
      "sub x9, %x[width], #0x1\n"
      "cntw x20, ALL, MUL #2\n"
      "ands x28, %x[width], x21\n"
      "mov x27, %x[in]\n"
      "add x9, x9, x20\n"
      "csel x28, x28, x23, NE\n"
      "add x26, x27, x10, LSL #3\n"
      "mov x25, #0x0\n"
      "udiv x9, x9, x20\n"
      "add x28, x28, #0x1\n"
      "mov x20, %x[width]\n"
      "add x24, x26, x10, LSL #3\n"
      "ptrue p0.b\n"
      "mov x23, %x[outptr_raw]\n"
      "mov x22, %x[row_offset]\n"
      "sub x9, x9, #0x1\n"
      "lsr x28, x28, #0x1\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44733  // whilelt pn11.s, x25, x20, VLx2\n"
      "add x21, x24, x10, LSL #3\n"
      "1:"  // Width loop: Preamble: Loop
      "ldr x20, [x27], #0x8\n"
      ".inst 0x25306c28  // psel p8.s, p11.s/Z, p1.s[w12]\n"
      ".inst 0x25306dca  // psel p10.s, p11.s/Z, p14.s[w12]\n"
      ".inst 0xa0164298  // ld1w { z24.s-z25.s }, pn8.s/Z, [x20, x22, LSL #2]\n"
      "ldr x20, [x26], #0x8\n"
      ".inst 0x25306da9  // psel p9.s, p11.s/Z, p13.s[w12]\n"
      ".inst 0x25306d88  // psel p8.s, p11.s/Z, p12.s[w12]\n"
      ".inst 0xa0164a82  // ld1w { z2.s-z3.s }, pn10.s/Z, [x20, x22, LSL #2]\n"
      "ldr x20, [x24], #0x8\n"
      ".inst 0xa016468a  // ld1w { z10.s-z11.s }, pn9.s/Z, [x20, x22, LSL #2]\n"
      ".inst 0xc160e318  // bfcvt z24.h, { z24.s-z25.s }\n"
      ".inst 0xc160e042  // bfcvt z2.h, { z2.s-z3.s }\n"
      "ldr x20, [x21], #0x8\n"
      ".inst 0xa016428c  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x22, LSL #2]\n"
      ".inst 0xc160e14a  // bfcvt z10.h, { z10.s-z11.s }\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc0800300  // mova za0h.s[x12], p0/M, z24.s\n"
      ".inst 0xc0800044  // mova za1h.s[x12], p0/M, z2.s\n"
      ".inst 0xc0800148  // mova za2h.s[x12], p0/M, z10.s\n"
      ".inst 0xc080018c  // mova za3h.s[x12], p0/M, z12.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x10\n"
      "blt 1b\n"
      "incw x22, ALL, MUL #2\n"
      "incw x25, ALL, MUL #2\n"
      "cbz x9, 5f\n"
      "2:"  // Width loop
      "mov x12, #0x0\n"
      "3:"  // Width loop: Store: Loop
      ".inst 0xc0828011  // mova z17.s, p0/M, za0v.s[x12]\n"
      ".inst 0xc0828095  // mova z21.s, p0/M, za1v.s[x12]\n"
      ".inst 0xc0828119  // mova z25.s, p0/M, za2v.s[x12]\n"
      ".inst 0xc082819d  // mova z29.s, p0/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x10\n"
      ".inst 0xa160def1  // st1w { z17.s, z21.s, z25.s, z29.s }, pn15.b, [x23]\n"
      "addvl x23, x23, #4\n"
      "blt 3b\n"
      "mov x27, %x[in]\n"
      "add x26, x27, x10, LSL #3\n"
      "mov x20, %x[width]\n"
      "add x24, x26, x10, LSL #3\n"
      "mov x12, #0x0\n"
      ".inst 0x25b44733  // whilelt pn11.s, x25, x20, VLx2\n"
      "add x21, x24, x10, LSL #3\n"
      "4:"  // Width loop: Load: Loop
      "ldr x20, [x27], #0x8\n"
      ".inst 0x25306c28  // psel p8.s, p11.s/Z, p1.s[w12]\n"
      ".inst 0x25306dca  // psel p10.s, p11.s/Z, p14.s[w12]\n"
      ".inst 0xa016428c  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x22, LSL #2]\n"
      "ldr x20, [x26], #0x8\n"
      ".inst 0x25306da9  // psel p9.s, p11.s/Z, p13.s[w12]\n"
      ".inst 0x25306d88  // psel p8.s, p11.s/Z, p12.s[w12]\n"
      ".inst 0xa0164a8e  // ld1w { z14.s-z15.s }, pn10.s/Z, [x20, x22, LSL #2]\n"
      "ldr x20, [x24], #0x8\n"
      ".inst 0xa0164692  // ld1w { z18.s-z19.s }, pn9.s/Z, [x20, x22, LSL #2]\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc160e1ce  // bfcvt z14.h, { z14.s-z15.s }\n"
      "ldr x20, [x21], #0x8\n"
      ".inst 0xa016429e  // ld1w { z30.s-z31.s }, pn8.s/Z, [x20, x22, LSL #2]\n"
      ".inst 0xc160e252  // bfcvt z18.h, { z18.s-z19.s }\n"
      ".inst 0xc160e3de  // bfcvt z30.h, { z30.s-z31.s }\n"
      ".inst 0xc0800180  // mova za0h.s[x12], p0/M, z12.s\n"
      ".inst 0xc08001c4  // mova za1h.s[x12], p0/M, z14.s\n"
      ".inst 0xc0800248  // mova za2h.s[x12], p0/M, z18.s\n"
      ".inst 0xc08003cc  // mova za3h.s[x12], p0/M, z30.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x10\n"
      "blt 4b\n"
      "subs x9, x9, #0x1\n"
      "incw x22, ALL, MUL #2\n"
      "incw x25, ALL, MUL #2\n"
      "bgt 2b\n"
      "5:"  // Width loop: Tails
      "mov x12, #0x0\n"
      "6:"  // Width loop: Tails: Loop
      ".inst 0xc0828011  // mova z17.s, p0/M, za0v.s[x12]\n"
      ".inst 0xc0828095  // mova z21.s, p0/M, za1v.s[x12]\n"
      ".inst 0xc0828119  // mova z25.s, p0/M, za2v.s[x12]\n"
      ".inst 0xc082819d  // mova z29.s, p0/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x28\n"
      ".inst 0xa160def1  // st1w { z17.s, z21.s, z25.s, z29.s }, pn15.b, [x23]\n"
      "addvl x23, x23, #4\n"
      "blt 6b\n"
      "7:"  // End
      "mov %x[outptr_raw], x23\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [outptr_raw] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
