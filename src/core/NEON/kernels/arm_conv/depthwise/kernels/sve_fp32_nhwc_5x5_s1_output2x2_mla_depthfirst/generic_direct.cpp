/*
 * Copyright (c) 2021 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl(
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
    "ptrue p3.b\n"
    "mov x5, #0x0\n"
    "mov x6, #0x0\n"
    "1:"  // Tile loop
    "str x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x20, #0x2\n"
    "str x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x7, #0x2\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_params]]\n"
    "mov x17, #0x0\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cntw x16\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "sub x14, XZR, x16\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x5, x22\n" // offset = tile_i * ld_input_row
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x6, x15, x19\n" // offset += tile_j * ld_input_col
    "ldr x12, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x20\n" // offset *= kernel_stride * output_size
    "ldr x11, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x13, x13, x19, LSL #2\n" // inptr[0] += offset * sizeof(float)
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x20, x13, x22, LSL #2\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "add x10, x20, x22, LSL #2\n"
    "ld1w { z16.s }, p3/Z, [x8]\n"
    "mov z31.d, z16.d\n"
    "ld1w { z0.s }, p3/Z, [x8, #1, MUL VL]\n"
    "add x9, x10, x22, LSL #2\n"
    "mov z30.d, z16.d\n"
    "ld1w { z1.s }, p3/Z, [x8, #2, MUL VL]\n"
    "add x28, x9, x22, LSL #2\n"
    "mov z29.d, z16.d\n"
    "ld1w { z2.s }, p3/Z, [x8, #3, MUL VL]\n"
    "add x27, x28, x22, LSL #2\n"
    "mov z28.d, z16.d\n"
    "ld1w { z3.s }, p3/Z, [x8, #4, MUL VL]\n"
    "add x26, x15, x15\n"
    "ld1w { z4.s }, p3/Z, [x8, #5, MUL VL]\n"
    "add x25, x26, x15\n"
    "mul x19, x5, x21\n" // offset = tile_i * ld_output_row
    "add x24, x25, x15\n"
    "add x23, x24, x15\n"
    "madd x19, x6, x12, x19\n" // offset += tile_j * ld_output_col
    "mul x19, x19, x7\n" // offset *= output_tile_size
    "add x11, x11, x19, LSL #2\n" // outptrs[0] += offset * sizeof(float)
    "add x22, x11, x21, LSL #2\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1w { z5.s }, p2/Z, [x13]\n"
    "ld1w { z6.s }, p2/Z, [x13, x15, LSL #2]\n"
    "cmp x16, %x[n_channels]\n"
    "ld1w { z7.s }, p2/Z, [x20]\n"
    "addvl x8, x8, #6\n"
    "ld1w { z8.s }, p2/Z, [x20, x15, LSL #2]\n"
    "ld1w { z9.s }, p2/Z, [x13, x26, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x20, x26, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x13, x25, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x13, x24, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x20, x23, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x10]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "fmla z31.s, p3/M, z0.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x20, x25, LSL #2]\n"
    "whilelt p1.s, x16, %x[n_channels]\n"
    "fmla z30.s, p3/M, z0.s, z6.s\n"
    "incw x14\n"
    "fmla z29.s, p3/M, z0.s, z7.s\n"
    "mov p0.b, p2.b\n"
    "fmla z28.s, p3/M, z0.s, z8.s\n"
    "ld1w { z0.s }, p3/Z, [x8]\n"
    "incw x17\n"
    "fmla z31.s, p3/M, z1.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x20, x24, LSL #2]\n"
    "addvl x20, x20, #1\n"
    "fmla z30.s, p3/M, z1.s, z9.s\n"
    "incw x16\n"
    "fmla z29.s, p3/M, z1.s, z8.s\n"
    "fmla z28.s, p3/M, z1.s, z13.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x13, x23, LSL #2]\n"
    "addvl x13, x13, #1\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "fmla z28.s, p3/M, z2.s, z5.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #2, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z29.s, p3/M, z3.s, z5.s\n"
    "fmla z28.s, p3/M, z3.s, z6.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x10, x25, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z6.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #4, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z7.s\n"
    "ld1w { z7.s }, p1/Z, [x20]\n"
    "fmla z30.s, p3/M, z0.s, z8.s\n"
    "fmla z29.s, p3/M, z0.s, z14.s\n"
    "fmla z28.s, p3/M, z0.s, z11.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #5, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z8.s\n"
    "ld1w { z8.s }, p2/Z, [x10, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z13.s\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #6, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x24, LSL #2]\n"
    "addvl x10, x10, #1\n"
    "fmla z30.s, p3/M, z2.s, z5.s\n"
    "fmla z29.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #7, MUL VL]\n"
    "addvl x8, x8, #16\n"
    "fmla z31.s, p3/M, z3.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x9]\n"
    "ld1w { z16.s }, p3/Z, [x8, #4, MUL VL]\n"
    "fmla z30.s, p3/M, z3.s, z6.s\n"
    "fmla z29.s, p3/M, z3.s, z9.s\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #-8, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x9, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x9, x26, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z13.s\n"
    "fmla z28.s, p3/M, z4.s, z8.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #-7, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x9, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z29.s, p3/M, z0.s, z5.s\n"
    "fmla z28.s, p3/M, z0.s, z6.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #-6, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x25, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z1.s, z6.s\n"
    "fmla z28.s, p3/M, z1.s, z10.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #-5, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x9, x24, LSL #2]\n"
    "addvl x9, x9, #1\n"
    "fmla z30.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z2.s, z10.s\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #-4, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x28]\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "fmla z29.s, p3/M, z3.s, z11.s\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #-3, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x28, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z8.s\n"
    "ld1w { z8.s }, p2/Z, [x28, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z28.s, p3/M, z4.s, z14.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #-2, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x28, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z6.s\n"
    "fmla z29.s, p3/M, z0.s, z9.s\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #-1, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x28, x25, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z13.s\n"
    "fmla z28.s, p3/M, z1.s, z5.s\n"
    "ld1w { z1.s }, p3/Z, [x8]\n"
    "fmla z31.s, p3/M, z2.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x23, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z5.s\n"
    "fmla z28.s, p3/M, z2.s, z6.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27]\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z29.s, p3/M, z3.s, z6.s\n"
    "fmla z28.s, p3/M, z3.s, z8.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #2, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z14.s\n"
    "ld1w { z14.s }, p1/Z, [x10]\n"
    "fmla z29.s, p3/M, z4.s, z8.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x27, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z0.s, z12.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #5, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z13.s\n"
    "ld1w { z13.s }, p1/Z, [x20, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z5.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x24, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z9.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #6, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z5.s\n"
    "ld1w { z5.s }, p1/Z, [x13]\n"
    "fmla z30.s, p3/M, z2.s, z6.s\n"
    "fmla z29.s, p3/M, z2.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x27, x23, LSL #2]\n"
    "whilelt p2.s, x17, %x[n_channels]\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #7, MUL VL]\n"
    "addvl x27, x27, #1\n"
    "fmla z31.s, p3/M, z3.s, z6.s\n"
    "ld1w { z6.s }, p1/Z, [x13, x15, LSL #2]\n"
    "addvl x8, x8, #16\n"
    "fmla z30.s, p3/M, z3.s, z8.s\n"
    "cmp x16, %x[n_channels]\n"
    "fmla z29.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x13, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #-8, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z8.s\n"
    "ld1w { z8.s }, p1/Z, [x20, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "ld1w { z10.s }, p1/Z, [x20, x23, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p1/Z, [x13, x24, LSL #2]\n"
    "fmla z28.s, p3/M, z4.s, z9.s\n"
    "ld1w { z9.s }, p1/Z, [x13, x26, LSL #2]\n"
    "ld1w { z4.s }, p3/Z, [x8, #-7, MUL VL]\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "addvl x8, x8, #-6\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z31.s }, p0, [x11]\n"
    "mov z31.d, z16.d\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "st1w { z30.s }, p0, [x11, x12, LSL #2]\n"
    "mov z30.d, z16.d\n"
    "addvl x11, x11, #1\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z29.s }, p0, [x22]\n"
    "mov z29.d, z16.d\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z28.s }, p0, [x22, x12, LSL #2]\n"
    "mov z28.d, z16.d\n"
    "addvl x22, x22, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "fmla z31.s, p3/M, z0.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x20, x25, LSL #2]\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z0.s, z6.s\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x21, x5, #0x1\n"
    "fmla z29.s, p3/M, z0.s, z7.s\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "fmla z28.s, p3/M, z0.s, z8.s\n"
    "ld1w { z0.s }, p3/Z, [x8]\n"
    "add x6, x6, #0x1\n"
    "fmla z31.s, p3/M, z1.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x20, x24, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z9.s\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z29.s, p3/M, z1.s, z8.s\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x6, x19\n"
    "fmla z28.s, p3/M, z1.s, z13.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x13, x23, LSL #2]\n"
    "csel x6, x6, XZR, LT\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "csel x5, x5, x21, LT\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "cmp x5, x20\n"
    "fmla z28.s, p3/M, z2.s, z5.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #2, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z29.s, p3/M, z3.s, z5.s\n"
    "fmla z28.s, p3/M, z3.s, z6.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x10, x25, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z6.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #4, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z7.s\n"
    "fmla z30.s, p3/M, z0.s, z8.s\n"
    "fmla z29.s, p3/M, z0.s, z14.s\n"
    "fmla z28.s, p3/M, z0.s, z11.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #5, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z8.s\n"
    "ld1w { z8.s }, p2/Z, [x10, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z13.s\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #6, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x24, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z5.s\n"
    "fmla z29.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #7, MUL VL]\n"
    "addvl x8, x8, #16\n"
    "fmla z31.s, p3/M, z3.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x9]\n"
    "fmla z30.s, p3/M, z3.s, z6.s\n"
    "fmla z29.s, p3/M, z3.s, z9.s\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #-8, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x9, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x9, x26, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z13.s\n"
    "fmla z28.s, p3/M, z4.s, z8.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #-7, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x9, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z29.s, p3/M, z0.s, z5.s\n"
    "fmla z28.s, p3/M, z0.s, z6.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #-6, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x25, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z1.s, z6.s\n"
    "fmla z28.s, p3/M, z1.s, z10.s\n"
    "ld1w { z1.s }, p3/Z, [x8, #-5, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x9, x24, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z2.s, z10.s\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #-4, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x28]\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "fmla z29.s, p3/M, z3.s, z11.s\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #-3, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x28, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z8.s\n"
    "ld1w { z8.s }, p2/Z, [x28, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z28.s, p3/M, z4.s, z14.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #-2, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z5.s\n"
    "ld1w { z5.s }, p2/Z, [x28, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z6.s\n"
    "fmla z29.s, p3/M, z0.s, z9.s\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z0.s }, p3/Z, [x8, #-1, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x28, x25, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z13.s\n"
    "fmla z28.s, p3/M, z1.s, z5.s\n"
    "ld1w { z1.s }, p3/Z, [x8]\n"
    "fmla z31.s, p3/M, z2.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z5.s\n"
    "fmla z28.s, p3/M, z2.s, z6.s\n"
    "ld1w { z2.s }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27]\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z29.s, p3/M, z3.s, z6.s\n"
    "fmla z28.s, p3/M, z3.s, z8.s\n"
    "ld1w { z3.s }, p3/Z, [x8, #2, MUL VL]\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z14.s\n"
    "fmla z29.s, p3/M, z4.s, z8.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ld1w { z4.s }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x27, x26, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z0.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z13.s\n"
    "fmla z30.s, p3/M, z1.s, z5.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x24, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z9.s\n"
    "fmla z31.s, p3/M, z2.s, z5.s\n"
    "fmla z30.s, p3/M, z2.s, z6.s\n"
    "fmla z29.s, p3/M, z2.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x27, x23, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "fmla z31.s, p3/M, z3.s, z6.s\n"
    "fmla z30.s, p3/M, z3.s, z8.s\n"
    "fmla z29.s, p3/M, z3.s, z11.s\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "fmla z31.s, p3/M, z4.s, z8.s\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z28.s, p3/M, z4.s, z9.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z31.s }, p0, [x11]\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z30.s }, p0, [x11, x12, LSL #2]\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z29.s }, p0, [x22]\n"
    "st1w { z28.s }, p0, [x22, x12, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z16", "z17", "z18", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
