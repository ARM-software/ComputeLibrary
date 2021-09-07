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

#if __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl(
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
    "ptrue p3.b\n"
    "mov x6, #0x0\n"
    "mov x7, #0x0\n"
    "1:"  // Tile loop
    "str x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x24, #0x3\n"
    "str x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x23, #0x3\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_params]]\n"
    "mov x17, #0x0\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cnth x16\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "sub x21, XZR, x16\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x6, x22\n" // offset = tile_i * ld_input_row
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x7, x15, x19\n" // offset += tile_j * ld_input_col
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x24\n" // offset *= kernel_stride * output_size
    "ldr x12, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x14, x14, x19, LSL #1\n" // inptr[0] += offset * sizeof(__fp16)
    "ld1rh { z18.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x11, x14, x22, LSL #1\n"
    "ld1rh { z17.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "add x10, x11, x22, LSL #1\n"
    "ld1h { z16.h }, p3/Z, [x8]\n"
    "add x9, x10, x22, LSL #1\n"
    "ld1h { z0.h }, p3/Z, [x8, #1, MUL VL]\n"
    "add x28, x9, x22, LSL #1\n"
    "ld1h { z1.h }, p3/Z, [x8, #2, MUL VL]\n"
    "add x27, x15, x15\n"
    "ld1h { z2.h }, p3/Z, [x8, #3, MUL VL]\n"
    "add x26, x27, x15\n"
    "ld1h { z3.h }, p3/Z, [x8, #4, MUL VL]\n"
    "add x25, x26, x15\n"
    "ld1h { z4.h }, p3/Z, [x8, #5, MUL VL]\n"
    "mul x19, x6, x20\n" // offset = tile_i * ld_output_row
    "ld1h { z5.h }, p3/Z, [x8, #6, MUL VL]\n"
    "madd x19, x7, x13, x19\n" // offset += tile_j * ld_output_col
    "ld1h { z6.h }, p3/Z, [x8, #7, MUL VL]\n"
    "mul x19, x19, x23\n" // offset *= output_tile_size
    "add x24, x13, x13\n"
    "add x12, x12, x19, LSL #1\n" // outptrs[0] += offset * sizeof(__fp16)
    "add x23, x12, x20, LSL #1\n"
    "add x22, x23, x20, LSL #1\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1h { z9.h }, p2/Z, [x10, x27, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x14]\n"
    "addvl x8, x8, #16\n"
    "ld1h { z11.h }, p2/Z, [x14, x25, LSL #1]\n"
    "cmp x16, %x[n_channels]\n"
    "ld1h { z7.h }, p3/Z, [x8, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x8, #-7, MUL VL]\n"
    "addvl x8, x8, #-6\n"
    "ld1h { z12.h }, p2/Z, [x28]\n"
    "ld1h { z13.h }, p2/Z, [x11, x27, LSL #1]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z31, z16\n fmla z31.h, p3/M, z8.h, z9.h\n"
    "whilelt p1.h, x16, %x[n_channels]\n"
    "movprfx z30, z16\n fmla z30.h, p3/M, z7.h, z9.h\n"
    "inch x21\n"
    "movprfx z29, z16\n fmla z29.h, p3/M, z6.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "movprfx z28, z16\n fmla z28.h, p3/M, z5.h, z9.h\n"
    "inch x17\n"
    "movprfx z27, z16\n fmla z27.h, p3/M, z4.h, z9.h\n"
    "inch x16\n"
    "movprfx z26, z16\n fmla z26.h, p3/M, z3.h, z9.h\n"
    "movprfx z25, z16\n fmla z25.h, p3/M, z2.h, z9.h\n"
    "movprfx z24, z16\n fmla z24.h, p3/M, z1.h, z9.h\n"
    "movprfx z23, z16\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "ld1h { z16.h }, p3/Z, [x8]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x10, x26, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x10, x15, LSL #1]\n"
    "fmla z25.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x28, x25, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z31.h, p3/M, z5.h, z13.h\n"
    "fmla z29.h, p3/M, z3.h, z13.h\n"
    "fmla z28.h, p3/M, z2.h, z13.h\n"
    "fmla z27.h, p3/M, z1.h, z13.h\n"
    "fmla z26.h, p3/M, z0.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x14, x15, LSL #1]\n"
    "fmla z23.h, p3/M, z8.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x26, LSL #1]\n"
    "fmla z31.h, p3/M, z7.h, z11.h\n"
    "fmla z30.h, p3/M, z6.h, z11.h\n"
    "fmla z28.h, p3/M, z4.h, z11.h\n"
    "fmla z27.h, p3/M, z3.h, z11.h\n"
    "fmla z25.h, p3/M, z1.h, z11.h\n"
    "fmla z24.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11]\n"
    "fmla z31.h, p3/M, z1.h, z13.h\n"
    "fmla z30.h, p3/M, z0.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x11, x25, LSL #1]\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "fmla z27.h, p3/M, z5.h, z10.h\n"
    "fmla z26.h, p3/M, z4.h, z10.h\n"
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x9]\n"
    "fmla z29.h, p3/M, z7.h, z10.h\n"
    "fmla z24.h, p3/M, z2.h, z10.h\n"
    "fmla z23.h, p3/M, z1.h, z10.h\n"
    "fmla z30.h, p3/M, z8.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x9, x27, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "fmla z28.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x9, x25, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z13.h\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x15, LSL #1]\n"
    "fmla z25.h, p3/M, z3.h, z12.h\n"
    "fmla z28.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x15, LSL #1]\n"
    "fmla z27.h, p3/M, z7.h, z10.h\n"
    "fmla z26.h, p3/M, z6.h, z10.h\n"
    "fmla z25.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z8.h, z10.h\n"
    "fmla z24.h, p3/M, z4.h, z10.h\n"
    "fmla z23.h, p3/M, z3.h, z10.h\n"
    "fmla z26.h, p3/M, z8.h, z11.h\n"
    "fmla z25.h, p3/M, z7.h, z13.h\n"
    "fmla z24.h, p3/M, z6.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x26, LSL #1]\n"
    "fmla z23.h, p3/M, z5.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x26, LSL #1]\n"
    "addvl x11, x11, #1\n"
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z27.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x9, x15, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "fmla z30.h, p3/M, z5.h, z11.h\n"
    "fmla z26.h, p3/M, z1.h, z11.h\n"
    "fmla z27.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x27, LSL #1]\n"
    "addvl x14, x14, #1\n"
    "fmla z24.h, p3/M, z8.h, z13.h\n"
    "ld1h { z10.h }, p1/Z, [x14]\n"
    "fmla z23.h, p3/M, z7.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x9, x26, LSL #1]\n"
    "addvl x9, x9, #1\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmla z27.h, p3/M, z6.h, z12.h\n"
    "fmla z25.h, p3/M, z4.h, z12.h\n"
    "fmla z24.h, p3/M, z3.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10]\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "fmla z30.h, p3/M, z1.h, z11.h\n"
    "ld1h { z1.h }, p3/Z, [x8, #2, MUL VL]\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x10, x25, LSL #1]\n"
    "addvl x10, x10, #1\n"
    "fmla z27.h, p3/M, z8.h, z13.h\n"
    "ld1h { z9.h }, p1/Z, [x10, x27, LSL #1]\n"
    "fmla z26.h, p3/M, z7.h, z13.h\n"
    "fmla z24.h, p3/M, z5.h, z13.h\n"
    "fmla z23.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x27, LSL #1]\n"
    "whilelt p2.h, x17, %x[n_channels]\n"
    "fmla z31.h, p3/M, z6.h, z12.h\n"
    "ld1h { z4.h }, p3/Z, [x8, #5, MUL VL]\n"
    "addvl x28, x28, #1\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "ld1h { z3.h }, p3/Z, [x8, #4, MUL VL]\n"
    "cmp x16, %x[n_channels]\n"
    "fmla z25.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x28]\n"
    "fmla z29.h, p3/M, z8.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z26.h, p3/M, z5.h, z11.h\n"
    "ld1h { z5.h }, p3/Z, [x8, #6, MUL VL]\n"
    "fmla z23.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x14, x25, LSL #1]\n"
    "fmla z24.h, p3/M, z7.h, z13.h\n"
    "ld1h { z2.h }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z25.h, p3/M, z8.h, z13.h\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "fmla z23.h, p3/M, z6.h, z13.h\n"
    "ld1h { z13.h }, p1/Z, [x11, x27, LSL #1]\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "ld1h { z6.h }, p3/Z, [x8, #7, MUL VL]\n"
    "addvl x8, x8, #16\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "ld1h { z7.h }, p3/Z, [x8, #-8, MUL VL]\n"
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "ld1h { z8.h }, p3/Z, [x8, #-7, MUL VL]\n"
    "addvl x8, x8, #-6\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "st1h { z31.h }, p0, [x12]\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "st1h { z30.h }, p0, [x12, x13, LSL #1]\n"
    "fmax z27.h, p3/M, z27.h, z18.h\n"
    "fmax z26.h, p3/M, z26.h, z18.h\n"
    "st1h { z29.h }, p0, [x12, x24, LSL #1]\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "addvl x12, x12, #1\n"
    "fmax z25.h, p3/M, z25.h, z18.h\n"
    "st1h { z28.h }, p0, [x23]\n"
    "fmin z27.h, p3/M, z27.h, z17.h\n"
    "fmin z26.h, p3/M, z26.h, z17.h\n"
    "st1h { z27.h }, p0, [x23, x13, LSL #1]\n"
    "fmin z25.h, p3/M, z25.h, z17.h\n"
    "fmax z24.h, p3/M, z24.h, z18.h\n"
    "st1h { z26.h }, p0, [x23, x24, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "fmax z23.h, p3/M, z23.h, z18.h\n"
    "st1h { z25.h }, p0, [x22]\n"
    "fmin z24.h, p3/M, z24.h, z17.h\n"
    "fmin z23.h, p3/M, z23.h, z17.h\n"
    "st1h { z24.h }, p0, [x22, x13, LSL #1]\n"
    "st1h { z23.h }, p0, [x22, x24, LSL #1]\n"
    "addvl x22, x22, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z31, z16\n fmla z31.h, p3/M, z8.h, z9.h\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z30, z16\n fmla z30.h, p3/M, z7.h, z9.h\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "add x21, x6, #0x1\n"
    "movprfx z29, z16\n fmla z29.h, p3/M, z6.h, z9.h\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "movprfx z28, z16\n fmla z28.h, p3/M, z5.h, z9.h\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "add x7, x7, #0x1\n"
    "movprfx z27, z16\n fmla z27.h, p3/M, z4.h, z9.h\n"
    "cmp x7, x19\n"
    "movprfx z26, z16\n fmla z26.h, p3/M, z3.h, z9.h\n"
    "movprfx z25, z16\n fmla z25.h, p3/M, z2.h, z9.h\n"
    "csel x7, x7, XZR, LT\n"
    "movprfx z24, z16\n fmla z24.h, p3/M, z1.h, z9.h\n"
    "csel x6, x6, x21, LT\n"
    "movprfx z23, z16\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "cmp x6, x20\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x10, x26, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x10, x15, LSL #1]\n"
    "fmla z25.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x28, x25, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z31.h, p3/M, z5.h, z13.h\n"
    "fmla z29.h, p3/M, z3.h, z13.h\n"
    "fmla z28.h, p3/M, z2.h, z13.h\n"
    "fmla z27.h, p3/M, z1.h, z13.h\n"
    "fmla z26.h, p3/M, z0.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x14, x15, LSL #1]\n"
    "fmla z23.h, p3/M, z8.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x26, LSL #1]\n"
    "fmla z31.h, p3/M, z7.h, z11.h\n"
    "fmla z30.h, p3/M, z6.h, z11.h\n"
    "fmla z28.h, p3/M, z4.h, z11.h\n"
    "fmla z27.h, p3/M, z3.h, z11.h\n"
    "fmla z25.h, p3/M, z1.h, z11.h\n"
    "fmla z24.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11]\n"
    "fmla z31.h, p3/M, z1.h, z13.h\n"
    "fmla z30.h, p3/M, z0.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x11, x25, LSL #1]\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "fmla z27.h, p3/M, z5.h, z10.h\n"
    "fmla z26.h, p3/M, z4.h, z10.h\n"
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x9]\n"
    "fmla z29.h, p3/M, z7.h, z10.h\n"
    "fmla z24.h, p3/M, z2.h, z10.h\n"
    "fmla z23.h, p3/M, z1.h, z10.h\n"
    "fmla z30.h, p3/M, z8.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x9, x27, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "fmla z28.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x9, x25, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z13.h\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x15, LSL #1]\n"
    "fmla z25.h, p3/M, z3.h, z12.h\n"
    "fmla z28.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x15, LSL #1]\n"
    "fmla z27.h, p3/M, z7.h, z10.h\n"
    "fmla z26.h, p3/M, z6.h, z10.h\n"
    "fmla z25.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z8.h, z10.h\n"
    "fmla z24.h, p3/M, z4.h, z10.h\n"
    "fmla z23.h, p3/M, z3.h, z10.h\n"
    "fmla z26.h, p3/M, z8.h, z11.h\n"
    "fmla z25.h, p3/M, z7.h, z13.h\n"
    "fmla z24.h, p3/M, z6.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x26, LSL #1]\n"
    "fmla z23.h, p3/M, z5.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x26, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z27.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x9, x15, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "fmla z30.h, p3/M, z5.h, z11.h\n"
    "fmla z26.h, p3/M, z1.h, z11.h\n"
    "fmla z27.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x27, LSL #1]\n"
    "fmla z24.h, p3/M, z8.h, z13.h\n"
    "fmla z23.h, p3/M, z7.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x9, x26, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmla z27.h, p3/M, z6.h, z12.h\n"
    "fmla z25.h, p3/M, z4.h, z12.h\n"
    "fmla z24.h, p3/M, z3.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10]\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "fmla z30.h, p3/M, z1.h, z11.h\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x10, x25, LSL #1]\n"
    "fmla z27.h, p3/M, z8.h, z13.h\n"
    "fmla z26.h, p3/M, z7.h, z13.h\n"
    "fmla z24.h, p3/M, z5.h, z13.h\n"
    "fmla z23.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x28, x27, LSL #1]\n"
    "fmla z31.h, p3/M, z6.h, z12.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "fmla z25.h, p3/M, z0.h, z12.h\n"
    "fmla z29.h, p3/M, z8.h, z11.h\n"
    "fmla z26.h, p3/M, z5.h, z11.h\n"
    "fmla z23.h, p3/M, z2.h, z11.h\n"
    "fmla z25.h, p3/M, z8.h, z13.h\n"
    "fmla z24.h, p3/M, z7.h, z13.h\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "fmla z23.h, p3/M, z6.h, z13.h\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "st1h { z31.h }, p0, [x12]\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "st1h { z30.h }, p0, [x12, x13, LSL #1]\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "fmax z27.h, p3/M, z27.h, z18.h\n"
    "st1h { z29.h }, p0, [x12, x24, LSL #1]\n"
    "fmax z26.h, p3/M, z26.h, z18.h\n"
    "fmax z25.h, p3/M, z25.h, z18.h\n"
    "fmax z24.h, p3/M, z24.h, z18.h\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "st1h { z28.h }, p0, [x23]\n"
    "fmin z27.h, p3/M, z27.h, z17.h\n"
    "fmin z26.h, p3/M, z26.h, z17.h\n"
    "st1h { z27.h }, p0, [x23, x13, LSL #1]\n"
    "fmin z25.h, p3/M, z25.h, z17.h\n"
    "fmin z24.h, p3/M, z24.h, z17.h\n"
    "st1h { z26.h }, p0, [x23, x24, LSL #1]\n"
    "fmax z23.h, p3/M, z23.h, z18.h\n"
    "st1h { z25.h }, p0, [x22]\n"
    "fmin z23.h, p3/M, z23.h, z17.h\n"
    "st1h { z24.h }, p0, [x22, x13, LSL #1]\n"
    "st1h { z23.h }, p0, [x22, x24, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)
