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

#include <cstddef>
#include <cstdint>

#if defined(ARM_COMPUTE_ENABLE_SME2)

namespace arm_conv {
namespace depthwise {

void sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
  const unsigned int n_tile_rows,
  const unsigned int n_tile_cols,
  const __fp16 *inptr,
  int64_t ld_input_row,
  int64_t ld_input_col,
  __fp16 *outptr,
  int64_t ld_output_row,
  int64_t ld_output_col,
  const void *params,
  unsigned int n_channels,
  const __fp16 activation_min,
  const __fp16 activation_max
)
{
  struct Args
  {
    const uint64_t n_tile_rows, n_tile_cols;
    const __fp16 *inptr;
    const uint64_t ld_input_row;
    const uint64_t ld_input_col;
    __fp16 *outptr;
    const uint64_t ld_output_row;
    const uint64_t ld_output_col;
    const void *params;
    const __fp16 min, max;

    uint64_t tile_i = 0, tile_j = 0;

    Args(
      const unsigned int n_tile_rows,
      const unsigned int n_tile_cols,
      const __fp16 *inptr,
      int64_t ld_input_row,
      int64_t ld_input_col,
      __fp16 *outptr,
      int64_t ld_output_row,
      int64_t ld_output_col,
      const void *params,
      const float activation_min,
      const float activation_max
    ) : n_tile_rows(n_tile_rows), n_tile_cols(n_tile_cols), inptr(inptr),
        ld_input_row(ld_input_row), ld_input_col(ld_input_col), outptr(outptr),
        ld_output_row(ld_output_row), ld_output_col(ld_output_col),
        params(params), min(activation_min), max(activation_max)
    {
    }
  };

  Args params_struct(
    n_tile_rows, n_tile_cols,
    inptr, ld_input_row, ld_input_col,
    outptr, ld_output_row, ld_output_col,
    params, activation_min, activation_max
  );

  __asm__ __volatile__(
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x2, #0x0\n"
    "mov x3, #0x0\n"
    "ptrue p3.b\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "1:"  // Tile loop
    "str x2, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x22, #0x4\n"
    "str x3, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x4, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x20, x2, x21\n"  // offset = tile_i * ld_input_row
    "ldr x6, [%x[params_struct], %[offsetof_args_params]]\n"
    "madd x20, x3, x4, x20\n"  // offset += tile_j * ld_input_col
    "mul x20, x20, x22\n"  // offset *= kernel_stride * output_size
    "add x7, x4, x4\n"
    "add x5, x5, x20, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x8, x5, x21, LSL #1\n"
    "add x17, x7, x4\n"
    "add x16, x8, x21, LSL #1\n"
    "add x15, x17, x4\n"
    "add x14, x16, x21, LSL #1\n"
    "add x13, x14, x21, LSL #1\n"
    "cbnz x3, 2f\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "lsl x12, %x[n_channels], #0x1\n"
    "mov x28, #0x8\n"
    "mul x28, x28, x4\n"
    "add x27, x16, x7, LSL #1\n"
    "add x26, x5, x4, LSL #1\n"
    "add x25, x5, x17, LSL #1\n"
    "sub x20, x24, x3\n"
    "add x24, x5, x15, LSL #1\n"
    "sub x20, x20, #0x1\n"
    "add x23, x8, x4, LSL #1\n"
    "and x20, x20, #0x3fffff\n"
    "add x22, x5, x7, LSL #1\n"
    "orr x12, x12, x20, LSL #22\n"
    "add x21, x8, x17, LSL #1\n"
    "orr x12, x12, x28, LSL #38\n"
    "add x20, x8, x15, LSL #1\n"
    "add x11, x8, x7, LSL #1\n"
    "add x10, x14, x4, LSL #1\n"
    "add x9, x16, x4, LSL #1\n"
    "add x28, x14, x17, LSL #1\n"
    ".inst 0xf8ac4b7a  // rprfm pldonce, x12, [x27]\n"
    "add x27, x16, x17, LSL #1\n"
    ".inst 0xf8ac48ba  // rprfm pldonce, x12, [x5]\n"
    ".inst 0xf8ac4b5a  // rprfm pldonce, x12, [x26]\n"
    "add x26, x14, x15, LSL #1\n"
    ".inst 0xf8ac4b3a  // rprfm pldonce, x12, [x25]\n"
    "add x25, x16, x15, LSL #1\n"
    ".inst 0xf8ac4b1a  // rprfm pldonce, x12, [x24]\n"
    "add x24, x13, x4, LSL #1\n"
    ".inst 0xf8ac491a  // rprfm pldonce, x12, [x8]\n"
    ".inst 0xf8ac4afa  // rprfm pldonce, x12, [x23]\n"
    "add x23, x14, x7, LSL #1\n"
    ".inst 0xf8ac4ada  // rprfm pldonce, x12, [x22]\n"
    "add x22, x13, x17, LSL #1\n"
    ".inst 0xf8ac4aba  // rprfm pldonce, x12, [x21]\n"
    "add x21, x13, x7, LSL #1\n"
    ".inst 0xf8ac4a9a  // rprfm pldonce, x12, [x20]\n"
    "add x20, x13, x15, LSL #1\n"
    ".inst 0xf8ac497a  // rprfm pldonce, x12, [x11]\n"
    ".inst 0xf8ac49da  // rprfm pldonce, x12, [x14]\n"
    ".inst 0xf8ac4a1a  // rprfm pldonce, x12, [x16]\n"
    ".inst 0xf8ac495a  // rprfm pldonce, x12, [x10]\n"
    ".inst 0xf8ac493a  // rprfm pldonce, x12, [x9]\n"
    ".inst 0xf8ac4b9a  // rprfm pldonce, x12, [x28]\n"
    ".inst 0xf8ac4b7a  // rprfm pldonce, x12, [x27]\n"
    ".inst 0xf8ac4b5a  // rprfm pldonce, x12, [x26]\n"
    ".inst 0xf8ac49ba  // rprfm pldonce, x12, [x13]\n"
    ".inst 0xf8ac4b3a  // rprfm pldonce, x12, [x25]\n"
    ".inst 0xf8ac4b1a  // rprfm pldonce, x12, [x24]\n"
    ".inst 0xf8ac4afa  // rprfm pldonce, x12, [x23]\n"
    ".inst 0xf8ac4ada  // rprfm pldonce, x12, [x22]\n"
    ".inst 0xf8ac4aba  // rprfm pldonce, x12, [x21]\n"
    ".inst 0xf8ac4a9a  // rprfm pldonce, x12, [x20]\n"
    "2:"  // Tile loop: Prefetch input rows: End
    "ldr x26, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "mov x20, #0x2\n"
    "ld1h { z19.h }, p3/Z, [x6]\n"
    "addvl x6, x6, #1\n"
    "ldr x25, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "cnth x24\n"
    ".inst 0xa040a0c0  // ld1h { z0.h-z3.h }, pn8.b/Z, [x6]\n"
    "addvl x6, x6, #4\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    ".inst 0xa040a0c4  // ld1h { z4.h-z7.h }, pn8.b/Z, [x6]\n"
    "addvl x6, x6, #4\n"
    "mul x22, x2, x26\n"  // offset = tile_i * ld_output_row
    "cmp x24, %x[n_channels]\n"
    "ld1rh { z18.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "madd x22, x3, x25, x22\n"  // offset += tile_j * ld_output_col
    "ld1rh { z17.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "mov x21, #0x0\n"
    "mul x22, x22, x20\n"  // offset *= output_tile_size
    "sub x20, XZR, x24\n"
    "ld1h { z8.h }, p3/Z, [x6]\n"
    "add x23, x23, x22, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "ld1h { z9.h }, p2/Z, [x16, x7, LSL #1]\n"
    "addvl x6, x6, #1\n"
    "add x22, x23, x26, LSL #1\n"
    "ld1h { z10.h }, p2/Z, [x5]\n"
    "ld1h { z11.h }, p2/Z, [x5, x4, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x5, x17, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x5, x15, LSL #1]\n"
    "ld1h { z14.h }, p2/Z, [x8]\n"
    "ld1h { z15.h }, p2/Z, [x8, x4, LSL #1]\n"
    "ld1h { z16.h }, p2/Z, [x5, x7, LSL #1]\n"
    "bge 4f\n"
    "3:"  // Tile loop: Channel loop
    "movprfx z28, z19\n fmla z28.h, p3/M, z8.h, z9.h\n"
    "movprfx z29, z19\n fmla z29.h, p3/M, z6.h, z9.h\n"
    "whilelt p1.h, x24, %x[n_channels]\n"
    "inch x21\n"
    "movprfx z30, z19\n fmla z30.h, p3/M, z2.h, z9.h\n"
    "movprfx z31, z19\n fmla z31.h, p3/M, z0.h, z9.h\n"
    "ld1h { z19.h }, p3/Z, [x6]\n"
    "addvl x6, x6, #1\n"
    "inch x24\n"
    "mov p0.b, p2.b\n"
    "addvl x5, x5, #1\n"
    "inch x20\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x15, LSL #1]\n"
    "ld1h { z10.h }, p1/Z, [x5]\n"
    "fmla z28.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x8, x17, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x8, x7, LSL #1]\n"
    "addvl x8, x8, #1\n"
    "fmla z28.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x14]\n"
    "fmla z29.h, p3/M, z0.h, z16.h\n"
    "fmla z28.h, p3/M, z4.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x16]\n"
    "fmla z30.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x14, x15, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x16, x4, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z15.h\n"
    "fmla z29.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x16, x17, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z13.h\n"
    "fmla z29.h, p3/M, z3.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x14, x17, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x16, x15, LSL #1]\n"
    "addvl x16, x16, #1\n"
    "ld1h { z9.h }, p1/Z, [x16, x7, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z13.h\n"
    "fmla z28.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x13]\n"
    "fmla z29.h, p3/M, z7.h, z12.h\n"
    "fmla z30.h, p3/M, z1.h, z16.h\n"
    "ld1h { z13.h }, p2/Z, [x13, x4, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x5, x17, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x7, LSL #1]\n"
    "addvl x14, x14, #1\n"
    "fmla z30.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x13, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z8.h, z11.h\n"
    "fmla z31.h, p3/M, z5.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x13, x17, LSL #1]\n"
    "fmla z30.h, p3/M, z7.h, z13.h\n"
    "ld1h { z13.h }, p1/Z, [x5, x15, LSL #1]\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x15, LSL #1]\n"
    "whilelt p2.h, x21, %x[n_channels]\n"
    "cmp x24, %x[n_channels]\n"
    "addvl x13, x13, #1\n"
    "fmla z30.h, p3/M, z5.h, z16.h\n"
    "fmla z31.h, p3/M, z3.h, z16.h\n"
    ".inst 0xa040a0c0  // ld1h { z0.h-z3.h }, pn8.b/Z, [x6]\n"
    "addvl x6, x6, #4\n"
    "ld1h { z16.h }, p1/Z, [x5, x7, LSL #1]\n"
    "fmla z31.h, p3/M, z7.h, z14.h\n"
    "ld1h { z14.h }, p1/Z, [x8]\n"
    "fmla z30.h, p3/M, z8.h, z15.h\n"
    "fmla z31.h, p3/M, z6.h, z15.h\n"
    ".inst 0xa040a0c4  // ld1h { z4.h-z7.h }, pn8.b/Z, [x6]\n"
    "addvl x6, x6, #4\n"
    "ld1h { z15.h }, p1/Z, [x8, x4, LSL #1]\n"
    "fmla z31.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x5, x4, LSL #1]\n"
    "ld1h { z8.h }, p3/Z, [x6]\n"
    "addvl x6, x6, #1\n"
    ".inst 0xc171ca5c  // fclamp { z28.h-z31.h }, z18.h, z17.h\n"
    "st1h { z28.h }, p0, [x23]\n"
    "st1h { z29.h }, p0, [x23, x25, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "st1h { z30.h }, p0, [x22]\n"
    "st1h { z31.h }, p0, [x22, x25, LSL #1]\n"
    "addvl x22, x22, #1\n"
    "blt 3b\n"
    "4:"  // Tile loop: Channel tail
    "movprfx z28, z19\n fmla z28.h, p3/M, z8.h, z9.h\n"
    "movprfx z29, z19\n fmla z29.h, p3/M, z6.h, z9.h\n"
    "ldr x3, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z30, z19\n fmla z30.h, p3/M, z2.h, z9.h\n"
    "movprfx z31, z19\n fmla z31.h, p3/M, z0.h, z9.h\n"
    "ldr x2, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x3, x3, #0x1\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x15, LSL #1]\n"
    "add x20, x2, #0x1\n"
    "cmp x3, x24\n"
    "csel x2, x2, x20, LT\n"
    "csel x3, x3, XZR, LT\n"
    "cmp x2, x21\n"
    "fmla z28.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x8, x17, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x8, x7, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x14]\n"
    "fmla z29.h, p3/M, z0.h, z16.h\n"
    "fmla z28.h, p3/M, z4.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x16]\n"
    "fmla z30.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x14, x15, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x16, x4, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z15.h\n"
    "fmla z29.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x16, x17, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z13.h\n"
    "fmla z29.h, p3/M, z3.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x14, x17, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x16, x15, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x13, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x13]\n"
    "fmla z29.h, p3/M, z7.h, z12.h\n"
    "fmla z30.h, p3/M, z1.h, z16.h\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "fmla z28.h, p3/M, z7.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x13, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z8.h, z11.h\n"
    "fmla z31.h, p3/M, z5.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x13, x17, LSL #1]\n"
    "fmla z30.h, p3/M, z7.h, z13.h\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x15, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z16.h\n"
    "fmla z31.h, p3/M, z3.h, z16.h\n"
    "fmla z30.h, p3/M, z8.h, z15.h\n"
    "fmla z31.h, p3/M, z7.h, z14.h\n"
    "fmla z31.h, p3/M, z6.h, z15.h\n"
    "fmla z31.h, p3/M, z8.h, z11.h\n"
    ".inst 0xc171ca5c  // fclamp { z28.h-z31.h }, z18.h, z17.h\n"
    "st1h { z28.h }, p0, [x23]\n"
    "st1h { z29.h }, p0, [x23, x25, LSL #1]\n"
    "st1h { z30.h }, p0, [x22]\n"
    "st1h { z31.h }, p0, [x22, x25, LSL #1]\n"
    "blt 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif // defined(ARM_COMPUTE_ENABLE_SME2)
