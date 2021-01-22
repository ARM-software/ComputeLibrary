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

#if defined(__ARM_FEATURE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl(
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
    "mov x3, #0x0\n"
    "mov x4, #0x0\n"
    "1:"  // Tile loop
    "str x3, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x22, #0x3\n"
    "str x4, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "cntb x5\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x5, x5, XZR, LSL #4\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cntb x7\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cntb x17\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x3, x20\n" // offset = tile_i * ld_input_row
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x4, x8, x19\n" // offset += tile_j * ld_input_col
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x22\n" // offset *= kernel_stride * output_size
    "ldr x14, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x16, x16, x19, LSL #2\n" // inptr[0] += offset * sizeof(float)
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x13, x16, x20, LSL #2\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "add x12, x13, x20, LSL #2\n"
    "ld1w { z16.s }, p3/Z, [x6]\n"
    "mov z31.d, z16.d\n"
    "ld1w { z0.s }, p3/Z, [x6, #1, MUL VL]\n"
    "add x11, x12, x20, LSL #2\n"
    "mov z30.d, z16.d\n"
    "ld1w { z1.s }, p3/Z, [x6, #2, MUL VL]\n"
    "add x10, x11, x20, LSL #2\n"
    "mov z29.d, z16.d\n"
    "ld1w { z2.s }, p3/Z, [x6, #3, MUL VL]\n"
    "add x9, x8, x8\n"
    "mov z28.d, z16.d\n"
    "ld1w { z3.s }, p3/Z, [x6, #4, MUL VL]\n"
    "add x28, x9, x8\n"
    "mov z27.d, z16.d\n"
    "ld1w { z4.s }, p3/Z, [x6, #5, MUL VL]\n"
    "add x27, x28, x8\n"
    "mov z26.d, z16.d\n"
    "ld1w { z5.s }, p3/Z, [x6, #6, MUL VL]\n"
    "add x7, x7, x8, LSL #4\n"
    "mov z25.d, z16.d\n"
    "ld1w { z6.s }, p3/Z, [x6, #7, MUL VL]\n"
    "add x17, x17, x9, LSL #4\n"
    "mov z24.d, z16.d\n"
    "prfm pldl1keep, [x12, x17]\n"
    "cntb x26\n"
    "mov z23.d, z16.d\n"
    "prfm pldl1keep, [x16, x5]\n"
    "add x26, x26, x28, LSL #4\n"
    "cntb x25\n"
    "mov x20, #0x3\n"
    "add x25, x25, x27, LSL #4\n"
    "prfm pldl1keep, [x16, x25]\n"
    "prfm pldl1keep, [x10, x5]\n"
    "mul x19, x3, x21\n" // offset = tile_i * ld_output_row
    "prfm pldl1keep, [x13, x17]\n"
    "madd x19, x4, x15, x19\n" // offset += tile_j * ld_output_col
    "add x24, x15, x15\n"
    "mul x19, x19, x20\n" // offset *= output_tile_size
    "add x14, x14, x19, LSL #2\n" // outptrs[0] += offset * sizeof(float)
    "add x23, x14, x21, LSL #2\n"
    "add x22, x23, x21, LSL #2\n"
    "mov x21, #0x0\n"
    "cntw x20\n"
    "sub x19, XZR, x20\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1w { z9.s }, p2/Z, [x12, x9, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x16]\n"
    "addvl x6, x6, #16\n"
    "ld1w { z11.s }, p2/Z, [x16, x27, LSL #2]\n"
    "cmp x20, %x[n_channels]\n"
    "ld1w { z7.s }, p3/Z, [x6, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x6, #-7, MUL VL]\n"
    "addvl x6, x6, #-6\n"
    "ld1w { z12.s }, p2/Z, [x10]\n"
    "ld1w { z13.s }, p2/Z, [x13, x9, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "fmla z31.s, p3/M, z8.s, z9.s\n"
    "prfm pldl1keep, [x10, x25]\n"
    "whilelt p1.s, x20, %x[n_channels]\n"
    "fmla z30.s, p3/M, z7.s, z9.s\n"
    "prfm pldl1keep, [x12, x7]\n"
    "incw x19\n"
    "fmla z29.s, p3/M, z6.s, z9.s\n"
    "prfm pldl1keep, [x16, x7]\n"
    "mov p0.b, p2.b\n"
    "fmla z28.s, p3/M, z5.s, z9.s\n"
    "prfm pldl1keep, [x16, x26]\n"
    "incw x21\n"
    "fmla z27.s, p3/M, z4.s, z9.s\n"
    "prfm pldl1keep, [x12, x26]\n"
    "incw x20\n"
    "fmla z26.s, p3/M, z3.s, z9.s\n"
    "prfm pldl1keep, [x13, x5]\n"
    "fmla z25.s, p3/M, z2.s, z9.s\n"
    "prfm pldl1keep, [x13, x25]\n"
    "fmla z24.s, p3/M, z1.s, z9.s\n"
    "prfm pldl1keep, [x11, x5]\n"
    "fmla z23.s, p3/M, z0.s, z9.s\n"
    "prfm pldl1keep, [x11, x17]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x12, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x8, LSL #2]\n"
    "fmla z25.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "prfm pldl1keep, [x11, x25]\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "prfm pldl1keep, [x10, x7]\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "prfm pldl1keep, [x13, x7]\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "prfm pldl1keep, [x13, x26]\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "prfm pldl1keep, [x10, x26]\n"
    "fmla z26.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x16, x8, LSL #2]\n"
    "fmla z23.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x16, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z11.s\n"
    "prfm pldl1keep, [x11, x7]\n"
    "fmla z30.s, p3/M, z6.s, z11.s\n"
    "prfm pldl1keep, [x16, x17]\n"
    "fmla z28.s, p3/M, z4.s, z11.s\n"
    "prfm pldl1keep, [x11, x26]\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "prfm pldl1keep, [x12, x5]\n"
    "fmla z25.s, p3/M, z1.s, z11.s\n"
    "prfm pldl1keep, [x12, x25]\n"
    "fmla z24.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13]\n"
    "fmla z31.s, p3/M, z1.s, z13.s\n"
    "prfm pldl1keep, [x10, x17]\n"
    "fmla z30.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x13, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z16.s }, p3/Z, [x6]\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "fmla z26.s, p3/M, z4.s, z10.s\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11]\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "fmla z24.s, p3/M, z2.s, z10.s\n"
    "fmla z23.s, p3/M, z1.s, z10.s\n"
    "fmla z30.s, p3/M, z8.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x11, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "fmla z28.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x11, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z13.s\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x8, LSL #2]\n"
    "fmla z25.s, p3/M, z3.s, z12.s\n"
    "fmla z28.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x13, x8, LSL #2]\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "fmla z26.s, p3/M, z6.s, z10.s\n"
    "fmla z25.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z24.s, p3/M, z4.s, z10.s\n"
    "fmla z23.s, p3/M, z3.s, z10.s\n"
    "fmla z26.s, p3/M, z8.s, z11.s\n"
    "fmla z25.s, p3/M, z7.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x28, LSL #2]\n"
    "addvl x13, x13, #1\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x8, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "fmla z30.s, p3/M, z5.s, z11.s\n"
    "fmla z26.s, p3/M, z1.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x16, x9, LSL #2]\n"
    "addvl x16, x16, #1\n"
    "fmla z24.s, p3/M, z8.s, z13.s\n"
    "ld1w { z10.s }, p1/Z, [x16]\n"
    "fmla z23.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x11, x28, LSL #2]\n"
    "addvl x11, x11, #1\n"
    "fmla z28.s, p3/M, z7.s, z12.s\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z25.s, p3/M, z4.s, z12.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x12]\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z30.s, p3/M, z1.s, z11.s\n"
    "ld1w { z1.s }, p3/Z, [x6, #2, MUL VL]\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x27, LSL #2]\n"
    "addvl x12, x12, #1\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "ld1w { z9.s }, p1/Z, [x12, x9, LSL #2]\n"
    "fmla z26.s, p3/M, z7.s, z13.s\n"
    "prfm pldl1keep, [x12, x17]\n"
    "fmla z24.s, p3/M, z5.s, z13.s\n"
    "prfm pldl1keep, [x16, x5]\n"
    "fmla z23.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x9, LSL #2]\n"
    "whilelt p2.s, x21, %x[n_channels]\n"
    "fmla z31.s, p3/M, z6.s, z12.s\n"
    "prfm pldl1keep, [x16, x25]\n"
    "addvl x10, x10, #1\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "prfm pldl1keep, [x10, x5]\n"
    "cmp x20, %x[n_channels]\n"
    "fmla z25.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p1/Z, [x10]\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "prfm pldl1keep, [x13, x17]\n"
    "fmla z26.s, p3/M, z5.s, z11.s\n"
    "ld1w { z0.s }, p3/Z, [x6, #1, MUL VL]\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x16, x27, LSL #2]\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "ld1w { z2.s }, p3/Z, [x6, #3, MUL VL]\n"
    "fmla z25.s, p3/M, z8.s, z13.s\n"
    "ld1w { z3.s }, p3/Z, [x6, #4, MUL VL]\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "ld1w { z4.s }, p3/Z, [x6, #5, MUL VL]\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "ld1w { z13.s }, p1/Z, [x13, x9, LSL #2]\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "ld1w { z5.s }, p3/Z, [x6, #6, MUL VL]\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "ld1w { z6.s }, p3/Z, [x6, #7, MUL VL]\n"
    "addvl x6, x6, #16\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "ld1w { z7.s }, p3/Z, [x6, #-8, MUL VL]\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "ld1w { z8.s }, p3/Z, [x6, #-7, MUL VL]\n"
    "addvl x6, x6, #-6\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "st1w { z31.s }, p0, [x14]\n"
    "mov z31.d, z16.d\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z30.s }, p0, [x14, x15, LSL #2]\n"
    "mov z30.d, z16.d\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z29.s }, p0, [x14, x24, LSL #2]\n"
    "mov z29.d, z16.d\n"
    "addvl x14, x14, #1\n"
    "fmax z27.s, p3/M, z27.s, z18.s\n"
    "st1w { z28.s }, p0, [x23]\n"
    "mov z28.d, z16.d\n"
    "fmax z26.s, p3/M, z26.s, z18.s\n"
    "fmax z25.s, p3/M, z25.s, z18.s\n"
    "fmax z24.s, p3/M, z24.s, z18.s\n"
    "fmin z27.s, p3/M, z27.s, z17.s\n"
    "st1w { z27.s }, p0, [x23, x15, LSL #2]\n"
    "mov z27.d, z16.d\n"
    "fmin z26.s, p3/M, z26.s, z17.s\n"
    "st1w { z26.s }, p0, [x23, x24, LSL #2]\n"
    "mov z26.d, z16.d\n"
    "addvl x23, x23, #1\n"
    "fmin z25.s, p3/M, z25.s, z17.s\n"
    "st1w { z25.s }, p0, [x22]\n"
    "mov z25.d, z16.d\n"
    "fmin z24.s, p3/M, z24.s, z17.s\n"
    "st1w { z24.s }, p0, [x22, x15, LSL #2]\n"
    "mov z24.d, z16.d\n"
    "fmax z23.s, p3/M, z23.s, z18.s\n"
    "fmin z23.s, p3/M, z23.s, z17.s\n"
    "st1w { z23.s }, p0, [x22, x24, LSL #2]\n"
    "mov z23.d, z16.d\n"
    "addvl x22, x22, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "fmla z31.s, p3/M, z8.s, z9.s\n"
    "prfm pldl1keep, [x10, x25]\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z7.s, z9.s\n"
    "prfm pldl1keep, [x12, x7]\n"
    "fmla z29.s, p3/M, z6.s, z9.s\n"
    "prfm pldl1keep, [x16, x7]\n"
    "fmla z28.s, p3/M, z5.s, z9.s\n"
    "prfm pldl1keep, [x16, x26]\n"
    "fmla z27.s, p3/M, z4.s, z9.s\n"
    "prfm pldl1keep, [x12, x26]\n"
    "fmla z26.s, p3/M, z3.s, z9.s\n"
    "prfm pldl1keep, [x13, x5]\n"
    "fmla z25.s, p3/M, z2.s, z9.s\n"
    "prfm pldl1keep, [x13, x25]\n"
    "fmla z24.s, p3/M, z1.s, z9.s\n"
    "prfm pldl1keep, [x11, x5]\n"
    "fmla z23.s, p3/M, z0.s, z9.s\n"
    "prfm pldl1keep, [x11, x17]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x12, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x8, LSL #2]\n"
    "fmla z25.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "prfm pldl1keep, [x11, x25]\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "prfm pldl1keep, [x10, x7]\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "prfm pldl1keep, [x13, x7]\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "prfm pldl1keep, [x13, x26]\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "prfm pldl1keep, [x10, x26]\n"
    "fmla z26.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x16, x8, LSL #2]\n"
    "fmla z23.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x16, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z11.s\n"
    "prfm pldl1keep, [x11, x7]\n"
    "fmla z30.s, p3/M, z6.s, z11.s\n"
    "prfm pldl1keep, [x16, x17]\n"
    "fmla z28.s, p3/M, z4.s, z11.s\n"
    "prfm pldl1keep, [x11, x26]\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "prfm pldl1keep, [x12, x5]\n"
    "fmla z25.s, p3/M, z1.s, z11.s\n"
    "prfm pldl1keep, [x12, x25]\n"
    "fmla z24.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13]\n"
    "fmla z31.s, p3/M, z1.s, z13.s\n"
    "prfm pldl1keep, [x10, x17]\n"
    "fmla z30.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x13, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ldr x3, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "add x21, x3, #0x1\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11]\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ldr x4, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "add x4, x4, #0x1\n"
    "fmla z30.s, p3/M, z8.s, z10.s\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "cmp x4, x19\n"
    "fmla z26.s, p3/M, z4.s, z10.s\n"
    "fmla z24.s, p3/M, z2.s, z10.s\n"
    "csel x4, x4, XZR, LT\n"
    "fmla z23.s, p3/M, z1.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x11, x9, LSL #2]\n"
    "csel x3, x3, x21, LT\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "cmp x3, x20\n"
    "fmla z28.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x11, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z13.s\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x8, LSL #2]\n"
    "fmla z25.s, p3/M, z3.s, z12.s\n"
    "fmla z28.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x13, x8, LSL #2]\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "fmla z26.s, p3/M, z6.s, z10.s\n"
    "fmla z25.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z24.s, p3/M, z4.s, z10.s\n"
    "fmla z23.s, p3/M, z3.s, z10.s\n"
    "fmla z26.s, p3/M, z8.s, z11.s\n"
    "fmla z25.s, p3/M, z7.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z12.s\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x8, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "fmla z30.s, p3/M, z5.s, z11.s\n"
    "fmla z26.s, p3/M, z1.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x16, x9, LSL #2]\n"
    "fmla z24.s, p3/M, z8.s, z13.s\n"
    "fmla z23.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x11, x28, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z12.s\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z25.s, p3/M, z4.s, z12.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x12]\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z30.s, p3/M, z1.s, z11.s\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x27, LSL #2]\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "fmla z26.s, p3/M, z7.s, z13.s\n"
    "fmla z24.s, p3/M, z5.s, z13.s\n"
    "fmla z23.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z6.s, z12.s\n"
    "fmla z28.s, p3/M, z3.s, z12.s\n"
    "fmla z25.s, p3/M, z0.s, z12.s\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "fmla z26.s, p3/M, z5.s, z11.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "fmla z25.s, p3/M, z8.s, z13.s\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z31.s }, p0, [x14]\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z30.s }, p0, [x14, x15, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmax z27.s, p3/M, z27.s, z18.s\n"
    "st1w { z29.s }, p0, [x14, x24, LSL #2]\n"
    "fmax z26.s, p3/M, z26.s, z18.s\n"
    "fmax z25.s, p3/M, z25.s, z18.s\n"
    "fmax z24.s, p3/M, z24.s, z18.s\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z28.s }, p0, [x23]\n"
    "fmin z27.s, p3/M, z27.s, z17.s\n"
    "fmin z26.s, p3/M, z26.s, z17.s\n"
    "st1w { z27.s }, p0, [x23, x15, LSL #2]\n"
    "fmin z25.s, p3/M, z25.s, z17.s\n"
    "fmin z24.s, p3/M, z24.s, z17.s\n"
    "st1w { z26.s }, p0, [x23, x24, LSL #2]\n"
    "fmax z23.s, p3/M, z23.s, z18.s\n"
    "st1w { z25.s }, p0, [x22]\n"
    "fmin z23.s, p3/M, z23.s, z17.s\n"
    "st1w { z24.s }, p0, [x22, x15, LSL #2]\n"
    "st1w { z23.s }, p0, [x22, x24, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE)
