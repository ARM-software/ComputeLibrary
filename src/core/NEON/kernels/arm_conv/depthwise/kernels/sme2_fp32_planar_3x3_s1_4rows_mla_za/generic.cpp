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

#if defined(ARM_COMPUTE_ENABLE_SME2)

#include <algorithm>
#include <cstddef>

namespace arm_conv {
namespace depthwise {

void sme2_fp32_planar_3x3_s1_4rows_mla_za_impl(
  const float *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const float *weights,
  const float *bias,
  float **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  float act_min,
  float act_max
)
{
  struct Args
  {
    const float *inptr;
    size_t ld_in_vl;
    long unsigned int pad_top, pad_bottom, pad_left;
    const float *weights;
    const float *bias;
    long unsigned int input_cols, output_cols;
    float **outptrs;
    const size_t *ld_out_cols;
    const size_t *ld_out_vls;
    long unsigned int current_channel, n_channels;
    float clamp_min, clamp_max;
  };

  Args args = { inptr, ld_in_vl, pad_top, 6u - std::min(6u, pad_top + valid_input_rows), pad_left, weights, bias, valid_input_cols, output_cols, outptrs, outlds, outvllds, start_channel, valid_channels, act_min, act_max };

  __asm__ __volatile__(
    "ldr x7, [%x[args], %[offsetof_Args_pad_bottom]]\n"
    "mov x20, #0x6\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "sub x20, x20, x7\n"
    "ldr x17, [%x[args], %[offsetof_Args_pad_top]]\n"
    "ptrue p2.b\n"
    ".inst 0x25207812  // ptrue pn10.b\n"
    "ld1rw { z2.s }, p2/Z, [%x[args], %[offsetof_Args_clamp_min]]\n"
    "ldr x16, [%x[args], %[offsetof_Args_n_channels]]\n"
    "whilelt p1.s, XZR, x16\n"
    "whilelt p9.s, XZR, x20\n"
    "ld1rw { z24.s }, p2/Z, [%x[args], %[offsetof_Args_clamp_max]]\n"
    "whilelt p8.s, XZR, x17\n"
    "eor p8.b, p2/Z, p8.b, p9.b\n"
    "ldr x15, [%x[args], %[offsetof_Args_current_channel]]\n"
    "1:"  // Channel loop
    "ldr x20, [%x[args], %[offsetof_Args_bias]]\n"
    "fmov z20.s, #0x0\n"
    "cbz x20, 2f\n"
    "ld1w { z20.s }, p1/Z, [x20, x15, LSL #2]\n"
    "2:"  // Load bias: Done
    "ldr x14, [%x[args], %[offsetof_Args_input_cols]]\n"
    "sub x20, x14, #0x1\n"
    "orr x24, x20, %x[ld_in_col], LSL #18\n"
    "mov z21.d, z20.d\n"
    "ldr x23, [%x[args], %[offsetof_Args_weights]]\n"
    ".inst 0xa0404ae6  // ld1w { z6.s-z7.s }, pn10.b/Z, [x23]\n"
    "orr x24, x16, x24, LSL #20\n"
    "mov x22, #0x6\n"
    "ldr x13, [%x[args], %[offsetof_Args_inptr]]\n"
    "ld1w { z10.s }, p2/Z, [x23, #2, MUL VL]\n"
    "addvl x23, x23, #3\n"
    "add x21, x17, x7\n"
    ".inst 0xa1404ae0  // ld1w { z0.s, z8.s }, pn10.b/Z, [x23]\n"
    "lsl x20, %x[ld_in_row], #0x2\n"
    "mov z22.d, z20.d\n"
    "mov z23.d, z20.d\n"
    "ld1w { z9.s }, p2/Z, [x23, #2, MUL VL]\n"
    "addvl x23, x23, #3\n"
    "mov x8, #0x0\n"
    "ldr x11, [%x[args], %[offsetof_Args_output_cols]]\n"
    ".inst 0xa0404ae4  // ld1w { z4.s-z5.s }, pn10.b/Z, [x23]\n"
    "lsl x24, x24, #0x2\n"
    "sub x22, x22, x21\n"
    "ld1w { z1.s }, p2/Z, [x23, #2, MUL VL]\n"
    "madd x20, x20, x17, x13\n"
    "3:"  // Issue prefetches
    "subs x22, x22, #0x1\n"
    ".inst 0xf8b84a9c  // rprfm pldstrm, x24, [x20]\n"
    "add x20, x20, %x[ld_in_col], LSL #2\n"
    "bgt 3b\n"
    "ldr x22, [%x[args], %[offsetof_Args_outptrs]]\n"
    "lsl x20, %x[ld_in_row], #0x2\n"
    "msub x13, x17, x20, x13\n"
    ".inst 0xc0040e80  // mova za.d[x8, #0], { z20.d-z23.d }\n"
    "ldr x20, [%x[args], %[offsetof_Args_ld_out_cols]]\n"
    ".inst 0xc0040e81  // mova za.d[x8, #1], { z20.d-z23.d }\n"
    "mov x10, #0x2\n"
    "ldp x9, x28, [x22], #0x10\n"
    ".inst 0xc0040e82  // mova za.d[x8, #2], { z20.d-z23.d }\n"
    "ldp x27, x26, [x20], #0x10\n"
    "ldr x21, [%x[args], %[offsetof_Args_pad_left]]\n"
    "ldp x25, x24, [x22], #0x10\n"
    "ldp x23, x22, [x20], #0x10\n"
    "cbz x21, 5f\n"
    "cmp x21, x10\n"
    "csel x20, x21, x10, LT\n"
    "sub x21, x21, x20\n"
    "sub x10, x10, x20\n"
    "cbz x21, 5f\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    "sub x11, x11, x21\n"
    ".inst 0xc1b8c84c  // fclamp { z12.s-z15.s }, z2.s, z24.s\n"
    "4:"  // Left padding
    "subs x21, x21, #0x1\n"
    "st1w { z12.s }, p1, [x9]\n"
    "add x9, x9, x27, LSL #2\n"
    "st1w { z13.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    "st1w { z14.s }, p1, [x25]\n"
    "add x25, x25, x23, LSL #2\n"
    "st1w { z15.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "bgt 4b\n"
    "5:"  // Left padding: End
    "adds XZR, x17, x7\n"
    "bne 10f\n"
    "cbz x10, 8f\n"
    "cmp x10, #0x1\n"
    "sub x14, x14, x10\n"
    "beq 7f\n"
    "6:"  // Unpadded: 2 priming loads
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z14.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z15.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "ld1w { z16.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "ld1w { z17.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc13619c0  // fmla za.s[x8, 0], { z14.s-z17.s }, z6.s\n"
    "ld1w { z18.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc13019e0  // fmla za.s[x8, 0], { z15.s-z18.s }, z0.s\n"
    "ld1w { z19.s }, p1/Z, [x20]\n"
    ".inst 0xc1341a00  // fmla za.s[x8, 0], { z16.s-z19.s }, z4.s\n"
    "7:"  // Unpadded: 1 priming loads
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z13.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z14.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "ld1w { z15.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "ld1w { z16.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc13719a0  // fmla za.s[x8, 0], { z13.s-z16.s }, z7.s\n"
    ".inst 0xc13619a1  // fmla za.s[x8, 1], { z13.s-z16.s }, z6.s\n"
    "ld1w { z17.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc13819c0  // fmla za.s[x8, 0], { z14.s-z17.s }, z8.s\n"
    "ld1w { z18.s }, p1/Z, [x20]\n"
    ".inst 0xc13019c1  // fmla za.s[x8, 1], { z14.s-z17.s }, z0.s\n"
    ".inst 0xc13519e0  // fmla za.s[x8, 0], { z15.s-z18.s }, z5.s\n"
    ".inst 0xc13419e1  // fmla za.s[x8, 1], { z15.s-z18.s }, z4.s\n"
    "8:"  // Unpadded: 0 priming loads
    "cbz x14, 16f\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z25.s }, p1/Z, [x13]\n"
    "sub x14, x14, #0x1\n"
    "ld1w { z26.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "sub x11, x11, #0x1\n"
    "ld1w { z27.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "cmp x14, x11\n"
    "ld1w { z28.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "csel x21, x14, x11, LT\n"
    "ld1w { z29.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z30.s }, p1/Z, [x20]\n"
    "sub x11, x11, x21\n"
    "cbz x21, 15f\n"
    "9:"  // Unpadded: Main loop
    ".inst 0xc13a1b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z10.s\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    "subs x21, x21, #0x1\n"
    ".inst 0xc1391b40  // fmla za.s[x8, 0], { z26.s-z29.s }, z9.s\n"
    ".inst 0xc1371b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z7.s\n"
    ".inst 0xc1361b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z6.s\n"
    "ld1w { z25.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1311b60  // fmla za.s[x8, 0], { z27.s-z30.s }, z1.s\n"
    ".inst 0xc1381b41  // fmla za.s[x8, 1], { z26.s-z29.s }, z8.s\n"
    ".inst 0xc1301b42  // fmla za.s[x8, 2], { z26.s-z29.s }, z0.s\n"
    "ld1w { z26.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    ".inst 0xc1b8c84c  // fclamp { z12.s-z15.s }, z2.s, z24.s\n"
    "st1w { z12.s }, p1, [x9]\n"
    "add x9, x9, x27, LSL #2\n"
    ".inst 0xc1351b61  // fmla za.s[x8, 1], { z27.s-z30.s }, z5.s\n"
    "st1w { z13.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc1341b62  // fmla za.s[x8, 2], { z27.s-z30.s }, z4.s\n"
    "ld1w { z27.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "add x8, x8, #0x1\n"
    "ld1w { z28.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "st1w { z14.s }, p1, [x25]\n"
    "add x25, x25, x23, LSL #2\n"
    "ld1w { z29.s }, p1/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "st1w { z15.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    ".inst 0xc0040e82  // mova za.d[x8, #2], { z20.d-z23.d }\n"
    "ld1w { z30.s }, p1/Z, [x20]\n"
    "bgt 9b\n"
    "b 15f\n"
    "10:"  // Padded
    "cbz x10, 13f\n"
    "cmp x10, #0x1\n"
    "sub x14, x14, x10\n"
    "beq 12f\n"
    "11:"  // Padded: 2 priming loads
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z11.s }, p0/Z, [x13]\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z12.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z13.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z14.s }, p0/Z, [x20]\n"
    "mov x12, #0x4\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1361960  // fmla za.s[x8, 0], { z11.s-z14.s }, z6.s\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z15.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1301980  // fmla za.s[x8, 0], { z12.s-z15.s }, z0.s\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z16.s }, p0/Z, [x20]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc13419a0  // fmla za.s[x8, 0], { z13.s-z16.s }, z4.s\n"
    "12:"  // Padded: 1 priming loads
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z11.s }, p0/Z, [x13]\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z12.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z13.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z14.s }, p0/Z, [x20]\n"
    "mov x12, #0x4\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1371960  // fmla za.s[x8, 0], { z11.s-z14.s }, z7.s\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    ".inst 0xc1361961  // fmla za.s[x8, 1], { z11.s-z14.s }, z6.s\n"
    "ld1w { z15.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    ".inst 0xc1381980  // fmla za.s[x8, 0], { z12.s-z15.s }, z8.s\n"
    "ld1w { z16.s }, p0/Z, [x20]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1301981  // fmla za.s[x8, 1], { z12.s-z15.s }, z0.s\n"
    ".inst 0xc13519a0  // fmla za.s[x8, 0], { z13.s-z16.s }, z5.s\n"
    ".inst 0xc13419a1  // fmla za.s[x8, 1], { z13.s-z16.s }, z4.s\n"
    "13:"  // Padded: 0 priming loads
    "cbz x14, 16f\n"
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z25.s }, p0/Z, [x13]\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z26.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z27.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z28.s }, p0/Z, [x20]\n"
    "mov x12, #0x4\n"
    "sub x14, x14, #0x1\n"
    "sub x11, x11, #0x1\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "cmp x14, x11\n"
    "ld1w { z29.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z30.s }, p0/Z, [x20]\n"
    "csel x21, x14, x11, LT\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "sub x11, x11, x21\n"
    "cbz x21, 15f\n"
    "14:"  // Padded: Main loop
    ".inst 0xc13a1b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z10.s\n"
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    ".inst 0xc1391b40  // fmla za.s[x8, 0], { z26.s-z29.s }, z9.s\n"
    "add x20, x13, %x[ld_in_row], LSL #2\n"
    "subs x21, x21, #0x1\n"
    ".inst 0xc1371b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z7.s\n"
    ".inst 0xc1361b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z6.s\n"
    "ld1w { z25.s }, p0/Z, [x13]\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1311b60  // fmla za.s[x8, 0], { z27.s-z30.s }, z1.s\n"
    ".inst 0xc1381b41  // fmla za.s[x8, 1], { z26.s-z29.s }, z8.s\n"
    ".inst 0xc1301b42  // fmla za.s[x8, 2], { z26.s-z29.s }, z0.s\n"
    "ld1w { z26.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    ".inst 0xc0060c10  // mova { z16.d-z19.d }, za.d[x8, #0]\n"
    ".inst 0xc1b8c850  // fclamp { z16.s-z19.s }, z2.s, z24.s\n"
    "st1w { z16.s }, p1, [x9]\n"
    "add x9, x9, x27, LSL #2\n"
    ".inst 0xc1351b61  // fmla za.s[x8, 1], { z27.s-z30.s }, z5.s\n"
    "st1w { z17.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc1341b62  // fmla za.s[x8, 2], { z27.s-z30.s }, z4.s\n"
    "ld1w { z27.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "mov x12, #0x4\n"
    "ld1w { z28.s }, p0/Z, [x20]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    "st1w { z18.s }, p1, [x25]\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "add x8, x8, #0x1\n"
    "ld1w { z29.s }, p0/Z, [x20]\n"
    "st1w { z19.s }, p1, [x24]\n"
    "add x20, x20, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    ".inst 0xc0040e82  // mova za.d[x8, #2], { z20.d-z23.d }\n"
    "ld1w { z30.s }, p0/Z, [x20]\n"
    "add x25, x25, x23, LSL #2\n"
    "add x24, x24, x22, LSL #2\n"
    "bgt 14b\n"
    "15:"  // Main loop tail
    ".inst 0xc13a1b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z10.s\n"
    ".inst 0xc1391b40  // fmla za.s[x8, 0], { z26.s-z29.s }, z9.s\n"
    ".inst 0xc1371b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z7.s\n"
    ".inst 0xc1361b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z6.s\n"
    ".inst 0xc1311b60  // fmla za.s[x8, 0], { z27.s-z30.s }, z1.s\n"
    ".inst 0xc1381b41  // fmla za.s[x8, 1], { z26.s-z29.s }, z8.s\n"
    ".inst 0xc1301b42  // fmla za.s[x8, 2], { z26.s-z29.s }, z0.s\n"
    ".inst 0xc0060c10  // mova { z16.d-z19.d }, za.d[x8, #0]\n"
    ".inst 0xc1b8c850  // fclamp { z16.s-z19.s }, z2.s, z24.s\n"
    "st1w { z16.s }, p1, [x9]\n"
    "add x9, x9, x27, LSL #2\n"
    ".inst 0xc1351b61  // fmla za.s[x8, 1], { z27.s-z30.s }, z5.s\n"
    "st1w { z17.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc1341b62  // fmla za.s[x8, 2], { z27.s-z30.s }, z4.s\n"
    "add x8, x8, #0x1\n"
    "st1w { z18.s }, p1, [x25]\n"
    "add x25, x25, x23, LSL #2\n"
    "st1w { z19.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    ".inst 0xc0040e82  // mova za.d[x8, #2], { z20.d-z23.d }\n"
    "16:"  // Main loop skip tail
    "cbz x11, 18f\n"
    "17:"  // Right padding loop
    ".inst 0xc0060c08  // mova { z8.d-z11.d }, za.d[x8, #0]\n"
    "add x8, x8, #0x1\n"
    "subs x11, x11, #0x1\n"
    ".inst 0xc1b8c848  // fclamp { z8.s-z11.s }, z2.s, z24.s\n"
    "st1w { z8.s }, p1, [x9]\n"
    "add x9, x9, x27, LSL #2\n"
    ".inst 0xc0040e82  // mova za.d[x8, #2], { z20.d-z23.d }\n"
    "st1w { z9.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    "st1w { z10.s }, p1, [x25]\n"
    "add x25, x25, x23, LSL #2\n"
    "st1w { z11.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "bgt 17b\n"
    "18:"  // End
    "ldr x20, [%x[args], %[offsetof_Args_weights]]\n"
    "incb x20, ALL, MUL #9\n"
    "str x20, [%x[args], %[offsetof_Args_weights]]\n"
    "incw x15\n"
    "ldr x21, [%x[args], %[offsetof_Args_ld_in_vl]]\n"
    "whilelt p1.s, x15, x16\n"
    "ldr x20, [%x[args], %[offsetof_Args_inptr]]\n"
    "add x20, x20, x21, LSL #2\n"
    "str x20, [%x[args], %[offsetof_Args_inptr]]\n"
    "ldr x25, [%x[args], %[offsetof_Args_outptrs]]\n"
    "ldr x24, [%x[args], %[offsetof_Args_ld_out_vls]]\n"
    "ldp x23, x22, [x25, #0x0]\n"
    "ldp x21, x20, [x24, #0x0]\n"
    "add x23, x23, x21, LSL #2\n"
    "add x22, x22, x20, LSL #2\n"
    "stp x23, x22, [x25, #0x0]\n"
    "ldp x23, x22, [x25, #0x10]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "add x23, x23, x21, LSL #2\n"
    "add x22, x22, x20, LSL #2\n"
    "stp x23, x22, [x25, #0x10]\n"
    "b.any 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [args] "r" (&args), [ld_in_col] "r" (ld_in_col), [ld_in_row] "r" (ld_in_row), [offsetof_Args_bias] "I" (offsetof(Args, bias)), [offsetof_Args_clamp_max] "I" (offsetof(Args, clamp_max)), [offsetof_Args_clamp_min] "I" (offsetof(Args, clamp_min)), [offsetof_Args_current_channel] "I" (offsetof(Args, current_channel)), [offsetof_Args_inptr] "I" (offsetof(Args, inptr)), [offsetof_Args_input_cols] "I" (offsetof(Args, input_cols)), [offsetof_Args_ld_in_vl] "I" (offsetof(Args, ld_in_vl)), [offsetof_Args_ld_out_cols] "I" (offsetof(Args, ld_out_cols)), [offsetof_Args_ld_out_vls] "I" (offsetof(Args, ld_out_vls)), [offsetof_Args_n_channels] "I" (offsetof(Args, n_channels)), [offsetof_Args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_Args_output_cols] "I" (offsetof(Args, output_cols)), [offsetof_Args_pad_bottom] "I" (offsetof(Args, pad_bottom)), [offsetof_Args_pad_left] "I" (offsetof(Args, pad_left)), [offsetof_Args_pad_top] "I" (offsetof(Args, pad_top)), [offsetof_Args_weights] "I" (offsetof(Args, weights))
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
