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
    "mov x19, #0x6\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "sub x19, x19, x7\n"
    "ldr x17, [%x[args], %[offsetof_Args_pad_top]]\n"
    "ptrue p2.b\n"
    ".inst 0x25207812  // ptrue pn10.b\n"
    "ld1rw { z5.s }, p2/Z, [%x[args], %[offsetof_Args_clamp_min]]\n"
    "ldr x16, [%x[args], %[offsetof_Args_n_channels]]\n"
    "whilelt p1.s, XZR, x16\n"
    "whilelt p9.s, XZR, x19\n"
    "ld1rw { z11.s }, p2/Z, [%x[args], %[offsetof_Args_clamp_max]]\n"
    "whilelt p8.s, XZR, x17\n"
    "eor p8.b, p2/Z, p8.b, p9.b\n"
    "ldr x15, [%x[args], %[offsetof_Args_current_channel]]\n"
    "1:"  // Channel loop
    "ldr x19, [%x[args], %[offsetof_Args_bias]]\n"
    "fmov z16.s, #0x0\n"
    "cbz x19, 2f\n"
    "ld1w { z16.s }, p1/Z, [x19, x15, LSL #2]\n"
    "2:"  // Load bias: Done
    "ldr x14, [%x[args], %[offsetof_Args_input_cols]]\n"
    "sub x19, x14, #0x1\n"
    "orr x23, x19, %x[ld_in_col], LSL #18\n"
    "mov z17.d, z16.d\n"
    "ldr x22, [%x[args], %[offsetof_Args_weights]]\n"
    ".inst 0xa1404ac0  // ld1w { z0.s, z8.s }, pn10.b/Z, [x22]\n"
    "orr x23, x16, x23, LSL #20\n"
    "mov x21, #0x6\n"
    "ldr x13, [%x[args], %[offsetof_Args_inptr]]\n"
    "ld1w { z3.s }, p2/Z, [x22, #2, MUL VL]\n"
    "addvl x22, x22, #3\n"
    "add x20, x17, x7\n"
    ".inst 0xa0404ac6  // ld1w { z6.s-z7.s }, pn10.b/Z, [x22]\n"
    "lsl x19, %x[ld_in_row], #0x2\n"
    "mov z18.d, z16.d\n"
    "mov z19.d, z16.d\n"
    "ld1w { z9.s }, p2/Z, [x22, #2, MUL VL]\n"
    "addvl x22, x22, #3\n"
    "mov x8, #0x0\n"
    "ldr x11, [%x[args], %[offsetof_Args_output_cols]]\n"
    ".inst 0xa1404ac2  // ld1w { z2.s, z10.s }, pn10.b/Z, [x22]\n"
    "lsl x23, x23, #0x2\n"
    "sub x21, x21, x20\n"
    "ld1w { z1.s }, p2/Z, [x22, #2, MUL VL]\n"
    "madd x19, x19, x17, x13\n"
    "3:"  // Issue prefetches
    "subs x21, x21, #0x1\n"
    ".inst 0xf8b74a7c  // rprfm pldstrm, x23, [x19]\n"
    "add x19, x19, %x[ld_in_col], LSL #2\n"
    "bgt 3b\n"
    "ldr x10, [%x[args], %[offsetof_Args_outptrs]]\n"
    "lsl x19, %x[ld_in_row], #0x2\n"
    "msub x13, x17, x19, x13\n"
    ".inst 0xc0040e00  // mova za.d[x8, #0], { z16.d-z19.d }\n"
    "ldr x19, [%x[args], %[offsetof_Args_ld_out_cols]]\n"
    ".inst 0xc0040e01  // mova za.d[x8, #1], { z16.d-z19.d }\n"
    "mov x9, #0x2\n"
    "ldp x28, x27, [x10], #0x10\n"
    ".inst 0xc0040e02  // mova za.d[x8, #2], { z16.d-z19.d }\n"
    "ldp x26, x25, [x19], #0x10\n"
    "ldr x20, [%x[args], %[offsetof_Args_pad_left]]\n"
    "ldp x24, x23, [x10], #0x10\n"
    "ldp x22, x21, [x19], #0x10\n"
    "cbz x20, 5f\n"
    "cmp x20, x9\n"
    "csel x19, x20, x9, LT\n"
    "sub x20, x20, x19\n"
    "sub x9, x9, x19\n"
    "cbz x20, 5f\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    "sub x11, x11, x20\n"
    ".inst 0xc1abc8ac  // fclamp { z12.s-z15.s }, z5.s, z11.s\n"
    "4:"  // Left padding
    "subs x20, x20, #0x1\n"
    "st1w { z12.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    "st1w { z13.s }, p1, [x27]\n"
    "add x27, x27, x25, LSL #2\n"
    "st1w { z14.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "st1w { z15.s }, p1, [x23]\n"
    "add x23, x23, x21, LSL #2\n"
    "bgt 4b\n"
    "5:"  // Left padding: End
    "adds XZR, x17, x7\n"
    "bne 10f\n"
    "cbz x9, 8f\n"
    "cmp x9, #0x1\n"
    "sub x14, x14, x9\n"
    "beq 7f\n"
    "6:"  // Unpadded: 2 priming loads
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z23.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z24.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "ld1w { z25.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "ld1w { z26.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1301ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z0.s\n"
    "ld1w { z27.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1361b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z6.s\n"
    "ld1w { z28.s }, p1/Z, [x19]\n"
    ".inst 0xc1321b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z2.s\n"
    "7:"  // Unpadded: 1 priming loads
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z23.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z24.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "ld1w { z25.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "ld1w { z26.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1381ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z8.s\n"
    ".inst 0xc1301ae1  // fmla za.s[x8, 1], { z23.s-z26.s }, z0.s\n"
    "ld1w { z27.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1371b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z7.s\n"
    "ld1w { z28.s }, p1/Z, [x19]\n"
    ".inst 0xc1361b01  // fmla za.s[x8, 1], { z24.s-z27.s }, z6.s\n"
    ".inst 0xc13a1b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z10.s\n"
    ".inst 0xc1321b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z2.s\n"
    "8:"  // Unpadded: 0 priming loads
    "cbz x14, 16f\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    "ld1w { z23.s }, p1/Z, [x13]\n"
    "sub x14, x14, #0x1\n"
    "ld1w { z24.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "sub x11, x11, #0x1\n"
    "ld1w { z25.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "cmp x14, x11\n"
    "ld1w { z26.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "csel x20, x14, x11, LT\n"
    "ld1w { z27.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "ld1w { z28.s }, p1/Z, [x19]\n"
    "sub x11, x11, x20\n"
    "cbz x20, 15f\n"
    "9:"  // Unpadded: Main loop
    ".inst 0xc1331ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z3.s\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    "subs x20, x20, #0x1\n"
    ".inst 0xc1391b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z9.s\n"
    ".inst 0xc1381ae1  // fmla za.s[x8, 1], { z23.s-z26.s }, z8.s\n"
    ".inst 0xc1301ae2  // fmla za.s[x8, 2], { z23.s-z26.s }, z0.s\n"
    "ld1w { z23.s }, p1/Z, [x13]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1311b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z1.s\n"
    ".inst 0xc1371b01  // fmla za.s[x8, 1], { z24.s-z27.s }, z7.s\n"
    ".inst 0xc1361b02  // fmla za.s[x8, 2], { z24.s-z27.s }, z6.s\n"
    "ld1w { z24.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    ".inst 0xc1abc8ac  // fclamp { z12.s-z15.s }, z5.s, z11.s\n"
    "st1w { z12.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc13a1b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z10.s\n"
    "st1w { z13.s }, p1, [x27]\n"
    "add x27, x27, x25, LSL #2\n"
    ".inst 0xc1321b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z2.s\n"
    "ld1w { z25.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "add x8, x8, #0x1\n"
    "ld1w { z26.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "st1w { z14.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "ld1w { z27.s }, p1/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "st1w { z15.s }, p1, [x23]\n"
    "add x23, x23, x21, LSL #2\n"
    ".inst 0xc0040e02  // mova za.d[x8, #2], { z16.d-z19.d }\n"
    "ld1w { z28.s }, p1/Z, [x19]\n"
    "bgt 9b\n"
    "b 15f\n"
    "10:"  // Padded
    "cbz x9, 13f\n"
    "cmp x9, #0x1\n"
    "sub x14, x14, x9\n"
    "beq 12f\n"
    "11:"  // Padded: 2 priming loads
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z23.s }, p0/Z, [x13]\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z24.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z25.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z26.s }, p0/Z, [x19]\n"
    "mov x12, #0x4\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1301ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z0.s\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z27.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1361b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z6.s\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z28.s }, p0/Z, [x19]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1321b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z2.s\n"
    "12:"  // Padded: 1 priming loads
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z23.s }, p0/Z, [x13]\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z24.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z25.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z26.s }, p0/Z, [x19]\n"
    "mov x12, #0x4\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0xc1381ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z8.s\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    ".inst 0xc1301ae1  // fmla za.s[x8, 1], { z23.s-z26.s }, z0.s\n"
    "ld1w { z27.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    ".inst 0xc1371b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z7.s\n"
    "ld1w { z28.s }, p0/Z, [x19]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1361b01  // fmla za.s[x8, 1], { z24.s-z27.s }, z6.s\n"
    ".inst 0xc13a1b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z10.s\n"
    ".inst 0xc1321b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z2.s\n"
    "13:"  // Padded: 0 priming loads
    "cbz x14, 16f\n"
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "ld1w { z23.s }, p0/Z, [x13]\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z24.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    "ld1w { z25.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "ld1w { z26.s }, p0/Z, [x19]\n"
    "mov x12, #0x4\n"
    "sub x14, x14, #0x1\n"
    "sub x11, x11, #0x1\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "cmp x14, x11\n"
    "ld1w { z27.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "ld1w { z28.s }, p0/Z, [x19]\n"
    "csel x20, x14, x11, LT\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    "sub x11, x11, x20\n"
    "cbz x20, 15f\n"
    "14:"  // Padded: Main loop
    ".inst 0xc1331ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z3.s\n"
    "mov x12, #0x0\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    ".inst 0xc1391b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z9.s\n"
    "add x19, x13, %x[ld_in_row], LSL #2\n"
    "subs x20, x20, #0x1\n"
    ".inst 0xc1381ae1  // fmla za.s[x8, 1], { z23.s-z26.s }, z8.s\n"
    ".inst 0xc1301ae2  // fmla za.s[x8, 2], { z23.s-z26.s }, z0.s\n"
    "ld1w { z23.s }, p0/Z, [x13]\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    "add x13, x13, %x[ld_in_col], LSL #2\n"
    ".inst 0xc1311b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z1.s\n"
    ".inst 0xc1371b01  // fmla za.s[x8, 1], { z24.s-z27.s }, z7.s\n"
    ".inst 0xc1361b02  // fmla za.s[x8, 2], { z24.s-z27.s }, z6.s\n"
    "ld1w { z24.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25b04500  // psel p0.s, p1.s/Z, p8.s[w12, #2]\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    ".inst 0xc1abc8ac  // fclamp { z12.s-z15.s }, z5.s, z11.s\n"
    "st1w { z12.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc13a1b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z10.s\n"
    "st1w { z13.s }, p1, [x27]\n"
    "add x27, x27, x25, LSL #2\n"
    ".inst 0xc1321b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z2.s\n"
    "ld1w { z25.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25f04500  // psel p0.s, p1.s/Z, p8.s[w12, #3]\n"
    "mov x12, #0x4\n"
    "ld1w { z26.s }, p0/Z, [x19]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    "st1w { z14.s }, p1, [x24]\n"
    ".inst 0x25304500  // psel p0.s, p1.s/Z, p8.s[w12]\n"
    "add x8, x8, #0x1\n"
    "ld1w { z27.s }, p0/Z, [x19]\n"
    "st1w { z15.s }, p1, [x23]\n"
    "add x19, x19, %x[ld_in_row], LSL #2\n"
    ".inst 0x25704500  // psel p0.s, p1.s/Z, p8.s[w12, #1]\n"
    ".inst 0xc0040e02  // mova za.d[x8, #2], { z16.d-z19.d }\n"
    "ld1w { z28.s }, p0/Z, [x19]\n"
    "add x24, x24, x22, LSL #2\n"
    "add x23, x23, x21, LSL #2\n"
    "bgt 14b\n"
    "15:"  // Main loop tail
    ".inst 0xc1331ae0  // fmla za.s[x8, 0], { z23.s-z26.s }, z3.s\n"
    ".inst 0xc1391b00  // fmla za.s[x8, 0], { z24.s-z27.s }, z9.s\n"
    ".inst 0xc1381ae1  // fmla za.s[x8, 1], { z23.s-z26.s }, z8.s\n"
    ".inst 0xc1301ae2  // fmla za.s[x8, 2], { z23.s-z26.s }, z0.s\n"
    ".inst 0xc1311b20  // fmla za.s[x8, 0], { z25.s-z28.s }, z1.s\n"
    ".inst 0xc1371b01  // fmla za.s[x8, 1], { z24.s-z27.s }, z7.s\n"
    ".inst 0xc1361b02  // fmla za.s[x8, 2], { z24.s-z27.s }, z6.s\n"
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    ".inst 0xc1abc8ac  // fclamp { z12.s-z15.s }, z5.s, z11.s\n"
    "st1w { z12.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc13a1b21  // fmla za.s[x8, 1], { z25.s-z28.s }, z10.s\n"
    "st1w { z13.s }, p1, [x27]\n"
    "add x27, x27, x25, LSL #2\n"
    ".inst 0xc1321b22  // fmla za.s[x8, 2], { z25.s-z28.s }, z2.s\n"
    "add x8, x8, #0x1\n"
    "st1w { z14.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "st1w { z15.s }, p1, [x23]\n"
    "add x23, x23, x21, LSL #2\n"
    ".inst 0xc0040e02  // mova za.d[x8, #2], { z16.d-z19.d }\n"
    "16:"  // Main loop skip tail
    "cbz x11, 18f\n"
    "17:"  // Right padding loop
    ".inst 0xc0060c0c  // mova { z12.d-z15.d }, za.d[x8, #0]\n"
    "add x8, x8, #0x1\n"
    "subs x11, x11, #0x1\n"
    ".inst 0xc1abc8ac  // fclamp { z12.s-z15.s }, z5.s, z11.s\n"
    "st1w { z12.s }, p1, [x28]\n"
    "add x28, x28, x26, LSL #2\n"
    ".inst 0xc0040e02  // mova za.d[x8, #2], { z16.d-z19.d }\n"
    "st1w { z13.s }, p1, [x27]\n"
    "add x27, x27, x25, LSL #2\n"
    "st1w { z14.s }, p1, [x24]\n"
    "add x24, x24, x22, LSL #2\n"
    "st1w { z15.s }, p1, [x23]\n"
    "add x23, x23, x21, LSL #2\n"
    "bgt 17b\n"
    "18:"  // End
    "ldr x22, [%x[args], %[offsetof_Args_weights]]\n"
    "incb x22, ALL, MUL #9\n"
    "str x22, [%x[args], %[offsetof_Args_weights]]\n"
    "incw x15\n"
    "ldr x19, [%x[args], %[offsetof_Args_ld_in_vl]]\n"
    "whilelt p1.s, x15, x16\n"
    "ldr x13, [%x[args], %[offsetof_Args_inptr]]\n"
    "add x13, x13, x19, LSL #2\n"
    "str x13, [%x[args], %[offsetof_Args_inptr]]\n"
    "ldr x10, [%x[args], %[offsetof_Args_outptrs]]\n"
    "ldr x23, [%x[args], %[offsetof_Args_ld_out_vls]]\n"
    "ldp x22, x21, [x10, #0x0]\n"
    "ldp x20, x19, [x23, #0x0]\n"
    "add x22, x22, x20, LSL #2\n"
    "add x21, x21, x19, LSL #2\n"
    "stp x22, x21, [x10, #0x0]\n"
    "ldp x22, x21, [x10, #0x10]\n"
    "ldp x20, x19, [x23, #0x10]\n"
    "add x22, x22, x20, LSL #2\n"
    "add x21, x21, x19, LSL #2\n"
    "stp x22, x21, [x10, #0x10]\n"
    "b.any 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [args] "r" (&args), [ld_in_col] "r" (ld_in_col), [ld_in_row] "r" (ld_in_row), [offsetof_Args_bias] "I" (offsetof(Args, bias)), [offsetof_Args_clamp_max] "I" (offsetof(Args, clamp_max)), [offsetof_Args_clamp_min] "I" (offsetof(Args, clamp_min)), [offsetof_Args_current_channel] "I" (offsetof(Args, current_channel)), [offsetof_Args_inptr] "I" (offsetof(Args, inptr)), [offsetof_Args_input_cols] "I" (offsetof(Args, input_cols)), [offsetof_Args_ld_in_vl] "I" (offsetof(Args, ld_in_vl)), [offsetof_Args_ld_out_cols] "I" (offsetof(Args, ld_out_cols)), [offsetof_Args_ld_out_vls] "I" (offsetof(Args, ld_out_vls)), [offsetof_Args_n_channels] "I" (offsetof(Args, n_channels)), [offsetof_Args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_Args_output_cols] "I" (offsetof(Args, output_cols)), [offsetof_Args_pad_bottom] "I" (offsetof(Args, pad_bottom)), [offsetof_Args_pad_left] "I" (offsetof(Args, pad_left)), [offsetof_Args_pad_top] "I" (offsetof(Args, pad_top)), [offsetof_Args_weights] "I" (offsetof(Args, weights))
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
