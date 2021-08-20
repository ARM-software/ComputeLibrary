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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst_direct_impl(
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
    "mov x2, #0x0\n"
    "mov x3, #0x0\n"
    "1:"  // Tile loop
    "str x2, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x24, #0x4\n"
    "str x3, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x23, #0x4\n"
    "ldr x4, [%x[params_struct], %[offsetof_args_params]]\n"
    "mov x5, #0x0\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cnth x6\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "sub x21, XZR, x6\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x2, x22\n" // offset = tile_i * ld_input_row
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x3, x7, x19\n" // offset += tile_j * ld_input_col
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x24\n" // offset *= kernel_stride * output_size
    "ldr x16, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x8, x8, x19, LSL #1\n" // inptr[0] += offset * sizeof(__fp16)
    "ld1rh { z15.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x15, x8, x22, LSL #1\n"
    "ld1rh { z14.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "add x14, x15, x22, LSL #1\n"
    "ld1h { z13.h }, p3/Z, [x4]\n" // Load from weights and bias
    "mov z31.d, z13.d\n"
    "ld1h { z0.h }, p3/Z, [x4, #1, MUL VL]\n" // Load from weights and bias
    "add x13, x14, x22, LSL #1\n"
    "mov z30.d, z13.d\n"
    "ld1h { z1.h }, p3/Z, [x4, #2, MUL VL]\n" // Load from weights and bias
    "add x12, x13, x22, LSL #1\n"
    "mov z29.d, z13.d\n"
    "ld1h { z2.h }, p3/Z, [x4, #3, MUL VL]\n" // Load from weights and bias
    "add x11, x12, x22, LSL #1\n"
    "mov z28.d, z13.d\n"
    "ld1h { z3.h }, p3/Z, [x4, #4, MUL VL]\n" // Load from weights and bias
    "add x10, x7, x7\n"
    "mov z27.d, z13.d\n"
    "ld1h { z4.h }, p3/Z, [x4, #5, MUL VL]\n" // Load from weights and bias
    "add x9, x10, x7\n"
    "mov z26.d, z13.d\n"
    "ld1h { z5.h }, p3/Z, [x4, #6, MUL VL]\n" // Load from weights and bias
    "add x28, x9, x7\n"
    "mov z25.d, z13.d\n"
    "ld1h { z6.h }, p3/Z, [x4, #7, MUL VL]\n" // Load from weights and bias
    "add x27, x28, x7\n"
    "mov z24.d, z13.d\n"
    "mul x19, x2, x20\n" // offset = tile_i * ld_output_row
    "mov z23.d, z13.d\n"
    "madd x19, x3, x17, x19\n" // offset += tile_j * ld_output_col
    "mov z22.d, z13.d\n"
    "mul x19, x19, x23\n" // offset *= output_tile_size
    "mov z21.d, z13.d\n"
    "add x16, x16, x19, LSL #1\n" // outptrs[0] += offset * sizeof(__fp16)
    "mov z20.d, z13.d\n"
    "add x26, x16, x20, LSL #1\n"
    "mov z19.d, z13.d\n"
    "add x25, x26, x20, LSL #1\n"
    "mov z18.d, z13.d\n"
    "add x24, x25, x20, LSL #1\n"
    "mov z17.d, z13.d\n"
    "add x23, x17, x17\n"
    "mov z16.d, z13.d\n"
    "add x22, x23, x17\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1h { z9.h }, p2/Z, [x14, x10, LSL #1]\n" // Load input point (2, 2)
    "ld1h { z10.h }, p2/Z, [x8]\n" // Load input point (0, 0)
    "addvl x4, x4, #16\n"
    "ld1h { z11.h }, p2/Z, [x8, x27, LSL #1]\n" // Load input point (0, 5)
    "cmp x6, %x[n_channels]\n"
    "ld1h { z7.h }, p3/Z, [x4, #-8, MUL VL]\n" // Load from weights and bias
    "ld1h { z8.h }, p3/Z, [x4, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #-6\n"
    "ld1h { z12.h }, p2/Z, [x14, x9, LSL #1]\n" // Load input point (2, 3)
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "fmla z31.h, p3/M, z8.h, z9.h\n"
    "ld1h { z13.h }, p3/Z, [x4]\n" // Load from weights and bias
    "whilelt p1.h, x6, %x[n_channels]\n"
    "fmla z30.h, p3/M, z7.h, z9.h\n"
    "inch x21\n"
    "fmla z29.h, p3/M, z6.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "fmla z27.h, p3/M, z5.h, z9.h\n"
    "inch x5\n"
    "fmla z26.h, p3/M, z4.h, z9.h\n"
    "inch x6\n"
    "fmla z25.h, p3/M, z3.h, z9.h\n"
    "fmla z23.h, p3/M, z2.h, z9.h\n"
    "fmla z22.h, p3/M, z1.h, z9.h\n"
    "fmla z21.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x13, x10, LSL #1]\n" // Load input point (3, 2)
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x11]\n" // Load input point (5, 0)
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x27, LSL #1]\n" // Load input point (5, 5)
    "fmla z30.h, p3/M, z8.h, z12.h\n"
    "fmla z29.h, p3/M, z7.h, z12.h\n"
    "fmla z26.h, p3/M, z5.h, z12.h\n"
    "fmla z28.h, p3/M, z6.h, z12.h\n"
    "fmla z25.h, p3/M, z4.h, z12.h\n"
    "fmla z24.h, p3/M, z3.h, z12.h\n"
    "fmla z22.h, p3/M, z2.h, z12.h\n"
    "fmla z21.h, p3/M, z1.h, z12.h\n"
    "fmla z20.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x7, LSL #1]\n" // Load input point (0, 1)
    "fmla z19.h, p3/M, z6.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x13, x9, LSL #1]\n" // Load input point (3, 3)
    "fmla z16.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x8, x28, LSL #1]\n" // Load input point (0, 4)
    "fmla z27.h, p3/M, z8.h, z9.h\n"
    "fmla z26.h, p3/M, z7.h, z9.h\n"
    "fmla z25.h, p3/M, z6.h, z9.h\n"
    "fmla z23.h, p3/M, z5.h, z9.h\n"
    "fmla z22.h, p3/M, z4.h, z9.h\n"
    "fmla z21.h, p3/M, z3.h, z9.h\n"
    "fmla z19.h, p3/M, z2.h, z9.h\n"
    "fmla z18.h, p3/M, z1.h, z9.h\n"
    "fmla z17.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x15]\n" // Load input point (1, 0)
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x15, x27, LSL #1]\n" // Load input point (1, 5)
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "fmla z28.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12]\n" // Load input point (4, 0)
    "fmla z26.h, p3/M, z8.h, z10.h\n"
    "fmla z25.h, p3/M, z7.h, z10.h\n"
    "fmla z24.h, p3/M, z6.h, z10.h\n"
    "fmla z22.h, p3/M, z5.h, z10.h\n"
    "fmla z21.h, p3/M, z4.h, z10.h\n"
    "fmla z20.h, p3/M, z3.h, z10.h\n"
    "fmla z18.h, p3/M, z2.h, z10.h\n"
    "fmla z17.h, p3/M, z1.h, z10.h\n"
    "fmla z16.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x15, x10, LSL #1]\n" // Load input point (1, 2)
    "fmla z31.h, p3/M, z3.h, z9.h\n"
    "fmla z27.h, p3/M, z0.h, z9.h\n"
    "fmla z28.h, p3/M, z5.h, z12.h\n"
    "fmla z24.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x15, x9, LSL #1]\n" // Load input point (1, 3)
    "fmla z23.h, p3/M, z6.h, z11.h\n"
    "fmla z19.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x27, LSL #1]\n" // Load input point (4, 5)
    "fmla z31.h, p3/M, z5.h, z10.h\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "fmla z29.h, p3/M, z3.h, z10.h\n"
    "fmla z27.h, p3/M, z2.h, z10.h\n"
    "fmla z26.h, p3/M, z1.h, z10.h\n"
    "fmla z25.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x14, x7, LSL #1]\n" // Load input point (2, 1)
    "fmla z20.h, p3/M, z8.h, z11.h\n"
    "fmla z16.h, p3/M, z5.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x7, LSL #1]\n" // Load input point (5, 1)
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "fmla z26.h, p3/M, z2.h, z12.h\n"
    "fmla z25.h, p3/M, z1.h, z12.h\n"
    "fmla z24.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x28, LSL #1]\n" // Load input point (2, 4)
    "fmla z19.h, p3/M, z7.h, z11.h\n"
    "fmla z18.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x28, LSL #1]\n" // Load input point (5, 4)
    "fmla z31.h, p3/M, z7.h, z10.h\n"
    "fmla z30.h, p3/M, z6.h, z10.h\n"
    "fmla z27.h, p3/M, z4.h, z10.h\n"
    "fmla z26.h, p3/M, z3.h, z10.h\n"
    "fmla z23.h, p3/M, z1.h, z10.h\n"
    "fmla z22.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x8, x10, LSL #1]\n" // Load input point (0, 2)
    "fmla z17.h, p3/M, z8.h, z11.h\n"
    "fmla z16.h, p3/M, z7.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x7, LSL #1]\n" // Load input point (3, 1)
    "fmla z29.h, p3/M, z8.h, z12.h\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmla z25.h, p3/M, z5.h, z12.h\n"
    "fmla z24.h, p3/M, z4.h, z12.h\n"
    "fmla z21.h, p3/M, z2.h, z12.h\n"
    "fmla z20.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x9, LSL #1]\n" // Load input point (0, 3)
    "addvl x8, x8, #1\n"
    "fmla z31.h, p3/M, z2.h, z10.h\n"
    "fmla z30.h, p3/M, z1.h, z10.h\n"
    "fmla z29.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x14]\n" // Load input point (2, 0)
    "fmla z27.h, p3/M, z7.h, z11.h\n"
    "fmla z26.h, p3/M, z6.h, z11.h\n"
    "fmla z23.h, p3/M, z4.h, z11.h\n"
    "fmla z22.h, p3/M, z3.h, z11.h\n"
    "fmla z19.h, p3/M, z1.h, z11.h\n"
    "fmla z18.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x28, LSL #1]\n" // Load input point (3, 4)
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "fmla z28.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x27, LSL #1]\n" // Load input point (2, 5)
    "addvl x14, x14, #1\n"
    "fmla z31.h, p3/M, z6.h, z10.h\n"
    "ld1h { z9.h }, p1/Z, [x14, x10, LSL #1]\n" // Load input point (2, 2)
    "fmla z27.h, p3/M, z3.h, z10.h\n"
    "fmla z23.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x13]\n" // Load input point (3, 0)
    "fmla z25.h, p3/M, z8.h, z11.h\n"
    "fmla z24.h, p3/M, z7.h, z11.h\n"
    "fmla z21.h, p3/M, z5.h, z11.h\n"
    "fmla z20.h, p3/M, z4.h, z11.h\n"
    "fmla z17.h, p3/M, z2.h, z11.h\n"
    "fmla z16.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x10, LSL #1]\n" // Load input point (4, 2)
    "fmla z28.h, p3/M, z8.h, z12.h\n"
    "fmla z24.h, p3/M, z5.h, z12.h\n"
    "fmla z20.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x13, x27, LSL #1]\n" // Load input point (3, 5)
    "addvl x13, x13, #1\n"
    "fmla z27.h, p3/M, z6.h, z10.h\n"
    "fmla z23.h, p3/M, z3.h, z10.h\n"
    "fmla z19.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x11, x10, LSL #1]\n" // Load input point (5, 2)
    "fmla z22.h, p3/M, z7.h, z11.h\n"
    "fmla z21.h, p3/M, z6.h, z11.h\n"
    "fmla z23.h, p3/M, z8.h, z11.h\n"
    "fmla z19.h, p3/M, z5.h, z11.h\n"
    "fmla z18.h, p3/M, z4.h, z11.h\n"
    "fmla z17.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x9, LSL #1]\n" // Load input point (4, 3)
    "fmla z24.h, p3/M, z8.h, z12.h\n"
    "fmla z20.h, p3/M, z5.h, z12.h\n"
    "fmla z16.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x9, LSL #1]\n" // Load input point (5, 3)
    "addvl x11, x11, #1\n"
    "fmla z19.h, p3/M, z8.h, z10.h\n"
    "fmla z18.h, p3/M, z7.h, z10.h\n"
    "fmla z17.h, p3/M, z6.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x15, x7, LSL #1]\n" // Load input point (1, 1)
    "fmla z22.h, p3/M, z8.h, z11.h\n"
    "fmla z21.h, p3/M, z7.h, z11.h\n"
    "fmla z20.h, p3/M, z6.h, z11.h\n"
    "fmla z18.h, p3/M, z5.h, z11.h\n"
    "fmla z17.h, p3/M, z4.h, z11.h\n"
    "fmla z16.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x15, x28, LSL #1]\n" // Load input point (1, 4)
    "addvl x15, x15, #1\n"
    "fmla z18.h, p3/M, z8.h, z12.h\n"
    "fmla z31.h, p3/M, z4.h, z10.h\n"
    "fmla z17.h, p3/M, z7.h, z12.h\n"
    "fmla z16.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x12, x7, LSL #1]\n" // Load input point (4, 1)
    "fmla z30.h, p3/M, z3.h, z10.h\n"
    "fmla z27.h, p3/M, z1.h, z10.h\n"
    "fmla z26.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x12, x28, LSL #1]\n" // Load input point (4, 4)
    "whilelt p2.h, x5, %x[n_channels]\n"
    "fmla z29.h, p3/M, z5.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x4, #1, MUL VL]\n" // Load from weights and bias
    "addvl x12, x12, #1\n"
    "fmla z28.h, p3/M, z4.h, z11.h\n"
    "cmp x6, %x[n_channels]\n"
    "fmla z25.h, p3/M, z2.h, z11.h\n"
    "ld1h { z2.h }, p3/Z, [x4, #3, MUL VL]\n" // Load from weights and bias
    "fmla z24.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x8, x27, LSL #1]\n" // Load input point (0, 5)
    "fmla z23.h, p3/M, z7.h, z12.h\n"
    "ld1h { z1.h }, p3/Z, [x4, #2, MUL VL]\n" // Load from weights and bias
    "fmla z22.h, p3/M, z6.h, z12.h\n"
    "ld1h { z6.h }, p3/Z, [x4, #7, MUL VL]\n" // Load from weights and bias
    "fmla z19.h, p3/M, z4.h, z12.h\n"
    "fmla z18.h, p3/M, z3.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x14, x9, LSL #1]\n" // Load input point (2, 3)
    "fmla z21.h, p3/M, z8.h, z10.h\n"
    "ld1h { z3.h }, p3/Z, [x4, #4, MUL VL]\n" // Load from weights and bias
    "fmla z20.h, p3/M, z7.h, z10.h\n"
    "fmla z17.h, p3/M, z5.h, z10.h\n"
    "ld1h { z5.h }, p3/Z, [x4, #6, MUL VL]\n" // Load from weights and bias
    "fmla z16.h, p3/M, z4.h, z10.h\n"
    "ld1h { z10.h }, p1/Z, [x8]\n" // Load input point (0, 0)
    "fmax z31.h, p3/M, z31.h, z15.h\n"
    "ld1h { z4.h }, p3/Z, [x4, #5, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #16\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "ld1h { z7.h }, p3/Z, [x4, #-8, MUL VL]\n" // Load from weights and bias
    "fmax z29.h, p3/M, z29.h, z15.h\n"
    "ld1h { z8.h }, p3/Z, [x4, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #-6\n"
    "fmin z31.h, p3/M, z31.h, z14.h\n"
    "st1h { z31.h }, p0, [x16]\n" // Store output point (0, 0)
    "mov z31.d, z13.d\n"
    "fmin z30.h, p3/M, z30.h, z14.h\n"
    "st1h { z30.h }, p0, [x16, x17, LSL #1]\n" // Store output point (0, 1)
    "mov z30.d, z13.d\n"
    "fmin z29.h, p3/M, z29.h, z14.h\n"
    "st1h { z29.h }, p0, [x16, x23, LSL #1]\n" // Store output point (0, 2)
    "mov z29.d, z13.d\n"
    "fmax z28.h, p3/M, z28.h, z15.h\n"
    "fmax z27.h, p3/M, z27.h, z15.h\n"
    "fmax z26.h, p3/M, z26.h, z15.h\n"
    "fmax z25.h, p3/M, z25.h, z15.h\n"
    "fmin z28.h, p3/M, z28.h, z14.h\n"
    "st1h { z28.h }, p0, [x16, x22, LSL #1]\n" // Store output point (0, 3)
    "mov z28.d, z13.d\n"
    "addvl x16, x16, #1\n"
    "fmin z27.h, p3/M, z27.h, z14.h\n"
    "st1h { z27.h }, p0, [x26]\n" // Store output point (1, 0)
    "mov z27.d, z13.d\n"
    "fmin z26.h, p3/M, z26.h, z14.h\n"
    "st1h { z26.h }, p0, [x26, x17, LSL #1]\n" // Store output point (1, 1)
    "mov z26.d, z13.d\n"
    "fmin z25.h, p3/M, z25.h, z14.h\n"
    "st1h { z25.h }, p0, [x26, x23, LSL #1]\n" // Store output point (1, 2)
    "mov z25.d, z13.d\n"
    "fmax z24.h, p3/M, z24.h, z15.h\n"
    "fmax z23.h, p3/M, z23.h, z15.h\n"
    "fmax z22.h, p3/M, z22.h, z15.h\n"
    "fmax z21.h, p3/M, z21.h, z15.h\n"
    "fmin z24.h, p3/M, z24.h, z14.h\n"
    "st1h { z24.h }, p0, [x26, x22, LSL #1]\n" // Store output point (1, 3)
    "mov z24.d, z13.d\n"
    "addvl x26, x26, #1\n"
    "fmin z23.h, p3/M, z23.h, z14.h\n"
    "st1h { z23.h }, p0, [x25]\n" // Store output point (2, 0)
    "mov z23.d, z13.d\n"
    "fmin z22.h, p3/M, z22.h, z14.h\n"
    "st1h { z22.h }, p0, [x25, x17, LSL #1]\n" // Store output point (2, 1)
    "mov z22.d, z13.d\n"
    "fmin z21.h, p3/M, z21.h, z14.h\n"
    "st1h { z21.h }, p0, [x25, x23, LSL #1]\n" // Store output point (2, 2)
    "mov z21.d, z13.d\n"
    "fmax z20.h, p3/M, z20.h, z15.h\n"
    "fmax z19.h, p3/M, z19.h, z15.h\n"
    "fmax z18.h, p3/M, z18.h, z15.h\n"
    "fmax z17.h, p3/M, z17.h, z15.h\n"
    "fmin z20.h, p3/M, z20.h, z14.h\n"
    "st1h { z20.h }, p0, [x25, x22, LSL #1]\n" // Store output point (2, 3)
    "mov z20.d, z13.d\n"
    "addvl x25, x25, #1\n"
    "fmin z19.h, p3/M, z19.h, z14.h\n"
    "st1h { z19.h }, p0, [x24]\n" // Store output point (3, 0)
    "mov z19.d, z13.d\n"
    "fmin z18.h, p3/M, z18.h, z14.h\n"
    "st1h { z18.h }, p0, [x24, x17, LSL #1]\n" // Store output point (3, 1)
    "mov z18.d, z13.d\n"
    "fmin z17.h, p3/M, z17.h, z14.h\n"
    "st1h { z17.h }, p0, [x24, x23, LSL #1]\n" // Store output point (3, 2)
    "mov z17.d, z13.d\n"
    "fmax z16.h, p3/M, z16.h, z15.h\n"
    "fmin z16.h, p3/M, z16.h, z14.h\n"
    "st1h { z16.h }, p0, [x24, x22, LSL #1]\n" // Store output point (3, 3)
    "mov z16.d, z13.d\n"
    "addvl x24, x24, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "fmla z31.h, p3/M, z8.h, z9.h\n"
    "ldr x2, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov p0.b, p2.b\n"
    "fmla z30.h, p3/M, z7.h, z9.h\n"
    "ldr x3, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "add x21, x2, #0x1\n"
    "fmla z29.h, p3/M, z6.h, z9.h\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z27.h, p3/M, z5.h, z9.h\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "add x3, x3, #0x1\n"
    "fmla z26.h, p3/M, z4.h, z9.h\n"
    "cmp x3, x19\n"
    "fmla z25.h, p3/M, z3.h, z9.h\n"
    "fmla z23.h, p3/M, z2.h, z9.h\n"
    "csel x3, x3, XZR, LT\n"
    "fmla z22.h, p3/M, z1.h, z9.h\n"
    "csel x2, x2, x21, LT\n"
    "fmla z21.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x13, x10, LSL #1]\n" // Load input point (3, 2)
    "cmp x2, x20\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x11]\n" // Load input point (5, 0)
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x27, LSL #1]\n" // Load input point (5, 5)
    "fmla z30.h, p3/M, z8.h, z12.h\n"
    "fmla z29.h, p3/M, z7.h, z12.h\n"
    "fmla z26.h, p3/M, z5.h, z12.h\n"
    "fmla z28.h, p3/M, z6.h, z12.h\n"
    "fmla z25.h, p3/M, z4.h, z12.h\n"
    "fmla z24.h, p3/M, z3.h, z12.h\n"
    "fmla z22.h, p3/M, z2.h, z12.h\n"
    "fmla z21.h, p3/M, z1.h, z12.h\n"
    "fmla z20.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x7, LSL #1]\n" // Load input point (0, 1)
    "fmla z19.h, p3/M, z6.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x13, x9, LSL #1]\n" // Load input point (3, 3)
    "fmla z16.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x8, x28, LSL #1]\n" // Load input point (0, 4)
    "fmla z27.h, p3/M, z8.h, z9.h\n"
    "fmla z26.h, p3/M, z7.h, z9.h\n"
    "fmla z25.h, p3/M, z6.h, z9.h\n"
    "fmla z23.h, p3/M, z5.h, z9.h\n"
    "fmla z22.h, p3/M, z4.h, z9.h\n"
    "fmla z21.h, p3/M, z3.h, z9.h\n"
    "fmla z19.h, p3/M, z2.h, z9.h\n"
    "fmla z18.h, p3/M, z1.h, z9.h\n"
    "fmla z17.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x15]\n" // Load input point (1, 0)
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x15, x27, LSL #1]\n" // Load input point (1, 5)
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "fmla z28.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12]\n" // Load input point (4, 0)
    "fmla z26.h, p3/M, z8.h, z10.h\n"
    "fmla z25.h, p3/M, z7.h, z10.h\n"
    "fmla z24.h, p3/M, z6.h, z10.h\n"
    "fmla z22.h, p3/M, z5.h, z10.h\n"
    "fmla z21.h, p3/M, z4.h, z10.h\n"
    "fmla z20.h, p3/M, z3.h, z10.h\n"
    "fmla z18.h, p3/M, z2.h, z10.h\n"
    "fmla z17.h, p3/M, z1.h, z10.h\n"
    "fmla z16.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x15, x10, LSL #1]\n" // Load input point (1, 2)
    "fmla z31.h, p3/M, z3.h, z9.h\n"
    "fmla z27.h, p3/M, z0.h, z9.h\n"
    "fmla z28.h, p3/M, z5.h, z12.h\n"
    "fmla z24.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x15, x9, LSL #1]\n" // Load input point (1, 3)
    "fmla z23.h, p3/M, z6.h, z11.h\n"
    "fmla z19.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x27, LSL #1]\n" // Load input point (4, 5)
    "fmla z31.h, p3/M, z5.h, z10.h\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "fmla z29.h, p3/M, z3.h, z10.h\n"
    "fmla z27.h, p3/M, z2.h, z10.h\n"
    "fmla z26.h, p3/M, z1.h, z10.h\n"
    "fmla z25.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x14, x7, LSL #1]\n" // Load input point (2, 1)
    "fmla z20.h, p3/M, z8.h, z11.h\n"
    "fmla z16.h, p3/M, z5.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x7, LSL #1]\n" // Load input point (5, 1)
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "fmla z26.h, p3/M, z2.h, z12.h\n"
    "fmla z25.h, p3/M, z1.h, z12.h\n"
    "fmla z24.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x28, LSL #1]\n" // Load input point (2, 4)
    "fmla z19.h, p3/M, z7.h, z11.h\n"
    "fmla z18.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x28, LSL #1]\n" // Load input point (5, 4)
    "fmla z31.h, p3/M, z7.h, z10.h\n"
    "fmla z30.h, p3/M, z6.h, z10.h\n"
    "fmla z27.h, p3/M, z4.h, z10.h\n"
    "fmla z26.h, p3/M, z3.h, z10.h\n"
    "fmla z23.h, p3/M, z1.h, z10.h\n"
    "fmla z22.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x8, x10, LSL #1]\n" // Load input point (0, 2)
    "fmla z17.h, p3/M, z8.h, z11.h\n"
    "fmla z16.h, p3/M, z7.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x7, LSL #1]\n" // Load input point (3, 1)
    "fmla z29.h, p3/M, z8.h, z12.h\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmla z25.h, p3/M, z5.h, z12.h\n"
    "fmla z24.h, p3/M, z4.h, z12.h\n"
    "fmla z21.h, p3/M, z2.h, z12.h\n"
    "fmla z20.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x8, x9, LSL #1]\n" // Load input point (0, 3)
    "fmla z31.h, p3/M, z2.h, z10.h\n"
    "fmla z30.h, p3/M, z1.h, z10.h\n"
    "fmla z29.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x14]\n" // Load input point (2, 0)
    "fmla z27.h, p3/M, z7.h, z11.h\n"
    "fmla z26.h, p3/M, z6.h, z11.h\n"
    "fmla z23.h, p3/M, z4.h, z11.h\n"
    "fmla z22.h, p3/M, z3.h, z11.h\n"
    "fmla z19.h, p3/M, z1.h, z11.h\n"
    "fmla z18.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x13, x28, LSL #1]\n" // Load input point (3, 4)
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "fmla z28.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x14, x27, LSL #1]\n" // Load input point (2, 5)
    "fmla z31.h, p3/M, z6.h, z10.h\n"
    "fmla z27.h, p3/M, z3.h, z10.h\n"
    "fmla z23.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x13]\n" // Load input point (3, 0)
    "fmla z25.h, p3/M, z8.h, z11.h\n"
    "fmla z24.h, p3/M, z7.h, z11.h\n"
    "fmla z21.h, p3/M, z5.h, z11.h\n"
    "fmla z20.h, p3/M, z4.h, z11.h\n"
    "fmla z17.h, p3/M, z2.h, z11.h\n"
    "fmla z16.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x10, LSL #1]\n" // Load input point (4, 2)
    "fmla z28.h, p3/M, z8.h, z12.h\n"
    "fmla z24.h, p3/M, z5.h, z12.h\n"
    "fmla z20.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x13, x27, LSL #1]\n" // Load input point (3, 5)
    "fmla z27.h, p3/M, z6.h, z10.h\n"
    "fmla z23.h, p3/M, z3.h, z10.h\n"
    "fmla z19.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x11, x10, LSL #1]\n" // Load input point (5, 2)
    "fmla z22.h, p3/M, z7.h, z11.h\n"
    "fmla z21.h, p3/M, z6.h, z11.h\n"
    "fmla z23.h, p3/M, z8.h, z11.h\n"
    "fmla z19.h, p3/M, z5.h, z11.h\n"
    "fmla z18.h, p3/M, z4.h, z11.h\n"
    "fmla z17.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x12, x9, LSL #1]\n" // Load input point (4, 3)
    "fmla z24.h, p3/M, z8.h, z12.h\n"
    "fmla z20.h, p3/M, z5.h, z12.h\n"
    "fmla z16.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x9, LSL #1]\n" // Load input point (5, 3)
    "fmla z19.h, p3/M, z8.h, z10.h\n"
    "fmla z18.h, p3/M, z7.h, z10.h\n"
    "fmla z17.h, p3/M, z6.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x15, x7, LSL #1]\n" // Load input point (1, 1)
    "fmla z22.h, p3/M, z8.h, z11.h\n"
    "fmla z21.h, p3/M, z7.h, z11.h\n"
    "fmla z20.h, p3/M, z6.h, z11.h\n"
    "fmla z18.h, p3/M, z5.h, z11.h\n"
    "fmla z17.h, p3/M, z4.h, z11.h\n"
    "fmla z16.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x15, x28, LSL #1]\n" // Load input point (1, 4)
    "fmla z31.h, p3/M, z4.h, z10.h\n"
    "fmla z18.h, p3/M, z8.h, z12.h\n"
    "fmla z17.h, p3/M, z7.h, z12.h\n"
    "fmla z16.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x12, x7, LSL #1]\n" // Load input point (4, 1)
    "fmla z30.h, p3/M, z3.h, z10.h\n"
    "fmla z27.h, p3/M, z1.h, z10.h\n"
    "fmla z26.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x12, x28, LSL #1]\n" // Load input point (4, 4)
    "fmla z29.h, p3/M, z5.h, z11.h\n"
    "fmla z28.h, p3/M, z4.h, z11.h\n"
    "fmla z25.h, p3/M, z2.h, z11.h\n"
    "fmla z24.h, p3/M, z1.h, z11.h\n"
    "fmla z23.h, p3/M, z7.h, z12.h\n"
    "fmla z22.h, p3/M, z6.h, z12.h\n"
    "fmla z19.h, p3/M, z4.h, z12.h\n"
    "fmla z18.h, p3/M, z3.h, z12.h\n"
    "fmla z21.h, p3/M, z8.h, z10.h\n"
    "fmla z20.h, p3/M, z7.h, z10.h\n"
    "fmla z17.h, p3/M, z5.h, z10.h\n"
    "fmla z16.h, p3/M, z4.h, z10.h\n"
    "fmax z31.h, p3/M, z31.h, z15.h\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "fmax z29.h, p3/M, z29.h, z15.h\n"
    "fmax z28.h, p3/M, z28.h, z15.h\n"
    "fmin z31.h, p3/M, z31.h, z14.h\n"
    "st1h { z31.h }, p0, [x16]\n" // Store output point (0, 0)
    "fmin z30.h, p3/M, z30.h, z14.h\n"
    "fmin z29.h, p3/M, z29.h, z14.h\n"
    "st1h { z30.h }, p0, [x16, x17, LSL #1]\n" // Store output point (0, 1)
    "fmin z28.h, p3/M, z28.h, z14.h\n"
    "fmax z27.h, p3/M, z27.h, z15.h\n"
    "st1h { z29.h }, p0, [x16, x23, LSL #1]\n" // Store output point (0, 2)
    "fmax z26.h, p3/M, z26.h, z15.h\n"
    "st1h { z28.h }, p0, [x16, x22, LSL #1]\n" // Store output point (0, 3)
    "fmin z27.h, p3/M, z27.h, z14.h\n"
    "fmax z25.h, p3/M, z25.h, z15.h\n"
    "st1h { z27.h }, p0, [x26]\n" // Store output point (1, 0)
    "fmin z26.h, p3/M, z26.h, z14.h\n"
    "fmin z25.h, p3/M, z25.h, z14.h\n"
    "st1h { z26.h }, p0, [x26, x17, LSL #1]\n" // Store output point (1, 1)
    "fmax z24.h, p3/M, z24.h, z15.h\n"
    "fmax z23.h, p3/M, z23.h, z15.h\n"
    "st1h { z25.h }, p0, [x26, x23, LSL #1]\n" // Store output point (1, 2)
    "fmax z22.h, p3/M, z22.h, z15.h\n"
    "fmax z21.h, p3/M, z21.h, z15.h\n"
    "fmax z20.h, p3/M, z20.h, z15.h\n"
    "fmin z24.h, p3/M, z24.h, z14.h\n"
    "st1h { z24.h }, p0, [x26, x22, LSL #1]\n" // Store output point (1, 3)
    "fmin z23.h, p3/M, z23.h, z14.h\n"
    "fmin z22.h, p3/M, z22.h, z14.h\n"
    "st1h { z23.h }, p0, [x25]\n" // Store output point (2, 0)
    "fmin z21.h, p3/M, z21.h, z14.h\n"
    "fmin z20.h, p3/M, z20.h, z14.h\n"
    "st1h { z22.h }, p0, [x25, x17, LSL #1]\n" // Store output point (2, 1)
    "fmax z19.h, p3/M, z19.h, z15.h\n"
    "st1h { z21.h }, p0, [x25, x23, LSL #1]\n" // Store output point (2, 2)
    "fmax z18.h, p3/M, z18.h, z15.h\n"
    "fmax z17.h, p3/M, z17.h, z15.h\n"
    "st1h { z20.h }, p0, [x25, x22, LSL #1]\n" // Store output point (2, 3)
    "fmin z19.h, p3/M, z19.h, z14.h\n"
    "st1h { z19.h }, p0, [x24]\n" // Store output point (3, 0)
    "fmin z18.h, p3/M, z18.h, z14.h\n"
    "fmin z17.h, p3/M, z17.h, z14.h\n"
    "st1h { z18.h }, p0, [x24, x17, LSL #1]\n" // Store output point (3, 1)
    "fmax z16.h, p3/M, z16.h, z15.h\n"
    "st1h { z17.h }, p0, [x24, x23, LSL #1]\n" // Store output point (3, 2)
    "fmin z16.h, p3/M, z16.h, z14.h\n"
    "st1h { z16.h }, p0, [x24, x22, LSL #1]\n" // Store output point (3, 3)
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)
