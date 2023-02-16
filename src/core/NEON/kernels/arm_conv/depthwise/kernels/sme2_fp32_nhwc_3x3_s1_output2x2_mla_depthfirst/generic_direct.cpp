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

#include <cstddef>
#include <cstdint>

namespace arm_conv {
namespace depthwise {

void sme2_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
  const unsigned int n_tile_rows,
  const unsigned int n_tile_cols,
  const float *inptr,
  int64_t ld_input_row,
  int64_t ld_input_col,
  float *outptr,
  int64_t ld_output_row,
  int64_t ld_output_col,
  const void *params,
  unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  struct Args
  {
    const uint64_t n_tile_rows, n_tile_cols;
    const float *inptr;
    const uint64_t ld_input_row;
    const uint64_t ld_input_col;
    float *outptr;
    const uint64_t ld_output_row;
    const uint64_t ld_output_col;
    const void *params;
    const float min, max;

    uint64_t tile_i = 0, tile_j = 0;

    Args(
      const unsigned int n_tile_rows,
      const unsigned int n_tile_cols,
      const float *inptr,
      int64_t ld_input_row,
      int64_t ld_input_col,
      float *outptr,
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
    "ptrue p3.b\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "mov x5, #0x0\n"
    "mov x6, #0x0\n"
    "1:"  // Tile loop
    "str x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x21, #0x2\n"
    "str x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "mul x19, x5, x20\n"  // offset = tile_i * ld_input_row
    "ldr x7, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "madd x19, x6, x7, x19\n"  // offset += tile_j * ld_input_col
    "mul x19, x19, x21\n"  // offset *= kernel_stride * output_size
    "ldr x8, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "add x8, x8, x19, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x17, x8, x20, LSL #2\n"
    "add x16, x17, x20, LSL #2\n"
    "add x15, x7, x7\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x13, x16, x20, LSL #2\n"
    "add x12, x15, x7\n"
    "cbnz x6, 2f\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "sub x20, x19, x6\n"
    "sub x20, x20, #0x1\n"
    "lsl x11, %x[n_channels], #0x2\n"
    "mov x19, #0x8\n"
    "and x20, x20, #0x3fffff\n"
    "mul x19, x19, x7\n"
    "orr x11, x11, x20, LSL #22\n"
    "orr x11, x11, x19, LSL #38\n"
    "add x10, x17, x7, LSL #2\n"
    "add x9, x8, x12, LSL #2\n"
    "add x28, x17, x15, LSL #2\n"
    "add x27, x16, x7, LSL #2\n"
    "add x26, x13, x12, LSL #2\n"
    "add x25, x8, x7, LSL #2\n"
    "add x24, x8, x15, LSL #2\n"
    "add x23, x16, x15, LSL #2\n"
    "add x22, x17, x12, LSL #2\n"
    "add x21, x16, x12, LSL #2\n"
    "add x20, x13, x7, LSL #2\n"
    "add x19, x13, x15, LSL #2\n"
    ".inst 0xf8ab495a  // rprfm pldonce, x10, [x11]\n"
    ".inst 0xf8ab491a  // rprfm pldonce, x8, [x11]\n"
    ".inst 0xf8ab493a  // rprfm pldonce, x9, [x11]\n"
    ".inst 0xf8ab4b9a  // rprfm pldonce, x28, [x11]\n"
    ".inst 0xf8ab4b7a  // rprfm pldonce, x27, [x11]\n"
    ".inst 0xf8ab49ba  // rprfm pldonce, x13, [x11]\n"
    ".inst 0xf8ab4b5a  // rprfm pldonce, x26, [x11]\n"
    ".inst 0xf8ab4b3a  // rprfm pldonce, x25, [x11]\n"
    ".inst 0xf8ab4b1a  // rprfm pldonce, x24, [x11]\n"
    ".inst 0xf8ab4afa  // rprfm pldonce, x23, [x11]\n"
    ".inst 0xf8ab4a3a  // rprfm pldonce, x17, [x11]\n"
    ".inst 0xf8ab4ada  // rprfm pldonce, x22, [x11]\n"
    ".inst 0xf8ab4a1a  // rprfm pldonce, x16, [x11]\n"
    ".inst 0xf8ab4aba  // rprfm pldonce, x21, [x11]\n"
    ".inst 0xf8ab4a9a  // rprfm pldonce, x20, [x11]\n"
    ".inst 0xf8ab4a7a  // rprfm pldonce, x19, [x11]\n"
    "2:"  // Tile loop: Prefetch input rows: End
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "mul x20, x5, x21\n"  // offset = tile_i * ld_output_row
    "mov x19, #0x2\n"
    "ld1w { z18.s }, p3/Z, [x14]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "madd x20, x6, x24, x20\n"  // offset += tile_j * ld_output_col
    "addvl x14, x14, #1\n"
    ".inst 0xa040c1c0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x14]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "mul x20, x20, x19\n"  // offset *= output_tile_size
    "cntw x22\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "addvl x14, x14, #4\n"
    "add x23, x23, x20, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    ".inst 0xa040c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "addvl x14, x14, #4\n"
    "ld1rw { z16.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "cmp x22, %x[n_channels]\n"
    "add x21, x23, x21, LSL #2\n"
    "ld1w { z8.s }, p3/Z, [x14]\n"
    "mov x20, #0x0\n"
    "sub x19, XZR, x22\n"
    "ld1w { z9.s }, p2/Z, [x17, x7, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x8]\n"
    "addvl x14, x14, #1\n"
    "ld1w { z11.s }, p2/Z, [x8, x12, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x17, x15, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x16, x7, LSL #2]\n"
    "bge 4f\n"
    "3:"  // Tile loop: Channel loop
    "movprfx z28, z18\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "whilelt p1.s, x22, %x[n_channels]\n"
    "incw x20\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x13]\n"
    "incw x22\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x12, LSL #2]\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x16, x15, LSL #2]\n"
    "incw x19\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x8, x7, LSL #2]\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x8, x15, LSL #2]\n"
    "addvl x8, x8, #1\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "ld1w { z18.s }, p3/Z, [x14]\n"
    "addvl x14, x14, #1\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x17]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x17, x12, LSL #2]\n"
    "addvl x17, x17, #1\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x16]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x16, x12, LSL #2]\n"
    "addvl x16, x16, #1\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "ld1w { z13.s }, p1/Z, [x16, x7, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x13, x15, LSL #2]\n"
    "whilelt p2.s, x20, %x[n_channels]\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    ".inst 0xa040c1c0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x14]\n"
    "addvl x14, x14, #4\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    ".inst 0xa040c1c4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x14]\n"
    "addvl x14, x14, #4\n"
    "cmp x22, %x[n_channels]\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "addvl x13, x13, #1\n"
    "ld1w { z9.s }, p1/Z, [x17, x7, LSL #2]\n"
    "ld1w { z10.s }, p1/Z, [x8]\n"
    "st1w { z28.s }, p0, [x23]\n"
    "ld1w { z11.s }, p1/Z, [x8, x12, LSL #2]\n"
    "st1w { z29.s }, p0, [x23, x24, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "ld1w { z12.s }, p1/Z, [x17, x15, LSL #2]\n"
    "st1w { z30.s }, p0, [x21]\n"
    "st1w { z31.s }, p0, [x21, x24, LSL #2]\n"
    "addvl x21, x21, #1\n"
    "ld1w { z8.s }, p3/Z, [x14]\n"
    "addvl x14, x14, #1\n"
    "blt 3b\n"
    "4:"  // Tile loop: Channel tail
    "movprfx z28, z18\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "add x6, x6, #0x1\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x13]\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x12, LSL #2]\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x16, x15, LSL #2]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x8, x7, LSL #2]\n"
    "cmp x6, x19\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x8, x15, LSL #2]\n"
    "add x19, x5, #0x1\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "csel x5, x5, x19, LT\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x17]\n"
    "csel x6, x6, XZR, LT\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x17, x12, LSL #2]\n"
    "cmp x5, x20\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x16]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x16, x12, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x13, x15, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "st1w { z28.s }, p0, [x23]\n"
    "st1w { z29.s }, p0, [x23, x24, LSL #2]\n"
    "st1w { z30.s }, p0, [x21]\n"
    "st1w { z31.s }, p0, [x21, x24, LSL #2]\n"
    "blt 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
