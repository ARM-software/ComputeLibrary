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
void interleave_block<4, 1, VLType::SME, false>(
  float * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool first
)
{
  ARM_COMPUTE_UNUSED(first);

  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cntw x15\n"
      "cmp %x[height], x15\n"
      "cntw x14\n"
      "cntw x13, ALL, MUL #2\n"
      "cntw x11, ALL, MUL #3\n"
      "csel x15, %x[height], x15, LT\n"
      "mov x10, #0x0\n"
      "ptrue p4.s\n"
      "sub x15, x15, #0x1\n"
      "whilelt p3.s, XZR, %x[height]\n"
      "whilelt p15.s, x14, %x[height]\n"
      "whilelt p14.s, x13, %x[height]\n"
      "whilelt p13.s, x11, %x[height]\n"
      "mov x9, %x[row_offset]\n"
      "mov x28, %x[out]\n"
      "whilelt p12.s, x10, %x[width]\n"
      "whilelt p11.s, x10, %x[width]\n"
      "whilelt p10.s, x10, %x[width]\n"
      "whilelt p9.s, x10, %x[width]\n"
      "whilelt p8.s, x10, %x[width]\n"
      "1:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x14, LSL #3\n"
      "ldr x25, [x27], #0x8\n"
      "add x24, %x[in], x13, LSL #3\n"
      "add x20, %x[in], x11, LSL #3\n"
      "ldr x23, [x26], #0x8\n"
      "mov x12, #0x0\n"
      "ldr x22, [x24], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x15, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25306c60  // psel p0.s, p11.s/Z, p3.s[w12]\n"
      ".inst 0x253069e2  // psel p2.s, p10.s/Z, p15.s[w12]\n"
      ".inst 0xe0890320  // ld1w { za0h.s[x12] }, p0/Z, [x25, x9, LSL #2]\n"
      "ldr x25, [x27], #0x8\n"
      ".inst 0x253065c1  // psel p1.s, p9.s/Z, p14.s[w12]\n"
      ".inst 0x253061a0  // psel p0.s, p8.s/Z, p13.s[w12]\n"
      ".inst 0xe0890ae4  // ld1w { za1h.s[x12] }, p2/Z, [x23, x9, LSL #2]\n"
      "ldr x23, [x26], #0x8\n"
      ".inst 0xe08906c8  // ld1w { za2h.s[x12] }, p1/Z, [x22, x9, LSL #2]\n"
      "ldr x22, [x24], #0x8\n"
      ".inst 0xe08902ac  // ld1w { za3h.s[x12] }, p0/Z, [x21, x9, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x15\n"
      "ldr x21, [x20], #0x8\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      "sub x20, %x[width], x10\n"
      ".inst 0x25306c60  // psel p0.s, p11.s/Z, p3.s[w12]\n"
      ".inst 0xe0890320  // ld1w { za0h.s[x12] }, p0/Z, [x25, x9, LSL #2]\n"
      ".inst 0x253069e0  // psel p0.s, p10.s/Z, p15.s[w12]\n"
      ".inst 0x253065c1  // psel p1.s, p9.s/Z, p14.s[w12]\n"
      ".inst 0xe08902e4  // ld1w { za1h.s[x12] }, p0/Z, [x23, x9, LSL #2]\n"
      ".inst 0x253061a0  // psel p0.s, p8.s/Z, p13.s[w12]\n"
      "cmp x20, x14\n"
      ".inst 0xe08906c8  // ld1w { za2h.s[x12] }, p1/Z, [x22, x9, LSL #2]\n"
      ".inst 0xe08902ac  // ld1w { za3h.s[x12] }, p0/Z, [x21, x9, LSL #2]\n"
      "mov x12, #0x0\n"
      "csel x20, x20, x14, LT\n"
      "4:"  // Stores: Loop
      ".inst 0x25305180  // psel p0.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0xe0bf8380  // st1w { za0v.s[x12] }, p0/Z, [x28, XZR, LSL #2]\n"
      ".inst 0x25305180  // psel p0.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0xe0ae8384  // st1w { za1v.s[x12] }, p0/Z, [x28, x14, LSL #2]\n"
      ".inst 0x25305181  // psel p1.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0x25305180  // psel p0.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0xe0ad8788  // st1w { za2v.s[x12] }, p1/Z, [x28, x13, LSL #2]\n"
      ".inst 0xe0ab838c  // st1w { za3v.s[x12] }, p0/Z, [x28, x11, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x20\n"
      "addvl x28, x28, #4\n"
      "blt 4b\n"
      "incw x10\n"
      "whilelt p12.s, x10, %x[width]\n"
      "whilelt p11.s, x10, %x[width]\n"
      "whilelt p10.s, x10, %x[width]\n"
      "whilelt p9.s, x10, %x[width]\n"
      "whilelt p8.s, x10, %x[width]\n"
      "incw x9\n"
      "b.any 1b\n"
      "mov %x[out], x28\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif  // defined(__ARM_FEATURE_SVE)
