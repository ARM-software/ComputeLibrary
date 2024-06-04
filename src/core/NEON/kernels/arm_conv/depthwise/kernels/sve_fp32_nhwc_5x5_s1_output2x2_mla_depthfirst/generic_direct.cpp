/*
 * Copyright (c) 2021, 2023-2024 Arm Limited.
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
    "mov x6, #0x0\n"
    "mov x7, #0x0\n"
    "1:"  // Tile loop
    "str x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x2\n"
    "mov x25, #0x2\n"
    "str x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cntw x17\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "mov x15, #0x0\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "mul x20, x6, x24\n"  // offset = tile_i * ld_input_row
    "add x12, x8, x8\n"
    "ldr x11, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x10, x12, x8\n"
    "cmp x17, %x[n_channels]\n"
    "ld1rw { z15.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mul x22, x6, x23\n"  // offset = tile_i * ld_output_row
    "add x9, x10, x8\n"
    "ld1rw { z28.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x21, XZR, x17\n"
    "madd x20, x7, x8, x20\n"  // offset += tile_j * ld_input_col
    "add x28, x9, x8\n"
    "ld1w { z29.s }, p3/Z, [x11]\n"
    "ld1w { z0.s }, p3/Z, [x11, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x11, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x11, #3, MUL VL]\n"
    "madd x22, x7, x16, x22\n"  // offset += tile_j * ld_output_col
    "ld1w { z3.s }, p3/Z, [x11, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x11, #5, MUL VL]\n"
    "addvl x11, x11, #6\n"
    "mul x20, x20, x26\n"  // offset *= kernel_stride * output_size
    "mul x22, x22, x25\n"  // offset *= output_tile_size
    "add x14, x14, x20, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x20, x14, x24, LSL #2\n"
    "add x27, x20, x24, LSL #2\n"
    "ld1w { z5.s }, p2/Z, [x14]\n"
    "ld1w { z6.s }, p2/Z, [x14, x8, LSL #2]\n"
    "add x26, x27, x24, LSL #2\n"
    "add x25, x26, x24, LSL #2\n"
    "ld1w { z7.s }, p2/Z, [x20]\n"
    "ld1w { z8.s }, p2/Z, [x20, x8, LSL #2]\n"
    "add x13, x13, x22, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "add x24, x25, x24, LSL #2\n"
    "add x23, x13, x23, LSL #2\n"
    "ld1w { z9.s }, p2/Z, [x14, x12, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x20, x12, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x14, x10, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x14, x9, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x20, x28, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x27]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z30, z29\n fmla z30.s, p3/M, z0.s, z5.s\n"
    "movprfx z31, z29\n fmla z31.s, p3/M, z0.s, z6.s\n"
    "ld1w { z25.s }, p2/Z, [x20, x10, LSL #2]\n"
    "whilelt p1.s, x17, %x[n_channels]\n"
    "movprfx z27, z29\n fmla z27.s, p3/M, z0.s, z7.s\n"
    "movprfx z26, z29\n fmla z26.s, p3/M, z0.s, z8.s\n"
    "ld1w { z23.s }, p3/Z, [x11]\n"
    "incw x15\n"
    "incw x17\n"
    "mov p0.b, p2.b\n"
    "incw x21\n"
    "fmla z30.s, p3/M, z1.s, z6.s\n"
    "ld1w { z22.s }, p2/Z, [x20, x9, LSL #2]\n"
    "addvl x20, x20, #1\n"
    "fmla z31.s, p3/M, z1.s, z9.s\n"
    "fmla z27.s, p3/M, z1.s, z8.s\n"
    "fmla z26.s, p3/M, z1.s, z13.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.s, p3/M, z2.s, z9.s\n"
    "ld1w { z18.s }, p2/Z, [x14, x28, LSL #2]\n"
    "addvl x14, x14, #1\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z13.s\n"
    "fmla z26.s, p3/M, z2.s, z25.s\n"
    "ld1w { z16.s }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.s, p3/M, z3.s, z11.s\n"
    "ld1w { z20.s }, p2/Z, [x27, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z12.s\n"
    "fmla z27.s, p3/M, z3.s, z25.s\n"
    "fmla z26.s, p3/M, z3.s, z22.s\n"
    "ld1w { z17.s }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.s, p3/M, z4.s, z12.s\n"
    "ld1w { z19.s }, p2/Z, [x27, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z18.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x10, LSL #2]\n"
    "fmla z27.s, p3/M, z4.s, z22.s\n"
    "fmla z26.s, p3/M, z4.s, z10.s\n"
    "ld1w { z0.s }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z30.s, p3/M, z23.s, z7.s\n"
    "ld1w { z7.s }, p1/Z, [x20]\n"
    "fmla z31.s, p3/M, z23.s, z8.s\n"
    "fmla z27.s, p3/M, z23.s, z14.s\n"
    "fmla z26.s, p3/M, z23.s, z20.s\n"
    "ld1w { z18.s }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.s, p3/M, z21.s, z8.s\n"
    "ld1w { z1.s }, p2/Z, [x27, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z21.s, z13.s\n"
    "fmla z27.s, p3/M, z21.s, z20.s\n"
    "fmla z26.s, p3/M, z21.s, z19.s\n"
    "ld1w { z5.s }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.s, p3/M, z16.s, z13.s\n"
    "ld1w { z24.s }, p2/Z, [x27, x9, LSL #2]\n"
    "addvl x27, x27, #1\n"
    "fmla z31.s, p3/M, z16.s, z25.s\n"
    "fmla z27.s, p3/M, z16.s, z19.s\n"
    "fmla z26.s, p3/M, z16.s, z12.s\n"
    "ld1w { z16.s }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.s, p3/M, z17.s, z25.s\n"
    "ld1w { z25.s }, p2/Z, [x26]\n"
    "fmla z31.s, p3/M, z17.s, z22.s\n"
    "fmla z27.s, p3/M, z17.s, z12.s\n"
    "ld1w { z29.s }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z26.s, p3/M, z17.s, z24.s\n"
    "ld1w { z17.s }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.s, p3/M, z0.s, z22.s\n"
    "ld1w { z23.s }, p2/Z, [x26, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ld1w { z22.s }, p2/Z, [x26, x12, LSL #2]\n"
    "fmla z27.s, p3/M, z0.s, z24.s\n"
    "fmla z26.s, p3/M, z0.s, z1.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #-7, MUL VL]\n"
    "fmla z30.s, p3/M, z18.s, z14.s\n"
    "ld1w { z10.s }, p2/Z, [x26, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z18.s, z20.s\n"
    "fmla z27.s, p3/M, z18.s, z25.s\n"
    "fmla z26.s, p3/M, z18.s, z23.s\n"
    "ld1w { z6.s }, p3/Z, [x11, #-6, MUL VL]\n"
    "fmla z30.s, p3/M, z5.s, z20.s\n"
    "ld1w { z0.s }, p2/Z, [x26, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z5.s, z19.s\n"
    "fmla z27.s, p3/M, z5.s, z23.s\n"
    "fmla z26.s, p3/M, z5.s, z22.s\n"
    "ld1w { z20.s }, p3/Z, [x11, #-5, MUL VL]\n"
    "fmla z30.s, p3/M, z16.s, z19.s\n"
    "ld1w { z19.s }, p2/Z, [x26, x9, LSL #2]\n"
    "addvl x26, x26, #1\n"
    "fmla z31.s, p3/M, z16.s, z12.s\n"
    "fmla z27.s, p3/M, z16.s, z22.s\n"
    "fmla z26.s, p3/M, z16.s, z0.s\n"
    "ld1w { z18.s }, p3/Z, [x11, #-4, MUL VL]\n"
    "fmla z30.s, p3/M, z17.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x25]\n"
    "fmla z31.s, p3/M, z17.s, z24.s\n"
    "fmla z27.s, p3/M, z17.s, z0.s\n"
    "fmla z26.s, p3/M, z17.s, z19.s\n"
    "ld1w { z17.s }, p3/Z, [x11, #-3, MUL VL]\n"
    "fmla z30.s, p3/M, z21.s, z24.s\n"
    "ld1w { z9.s }, p2/Z, [x25, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z21.s, z1.s\n"
    "ld1w { z8.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z27.s, p3/M, z21.s, z19.s\n"
    "fmla z26.s, p3/M, z21.s, z10.s\n"
    "ld1w { z5.s }, p3/Z, [x11, #-2, MUL VL]\n"
    "fmla z30.s, p3/M, z6.s, z25.s\n"
    "ld1w { z25.s }, p2/Z, [x25, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z6.s, z23.s\n"
    "fmla z27.s, p3/M, z6.s, z16.s\n"
    "fmla z26.s, p3/M, z6.s, z9.s\n"
    "ld1w { z4.s }, p3/Z, [x11, #-1, MUL VL]\n"
    "fmla z30.s, p3/M, z20.s, z23.s\n"
    "ld1w { z24.s }, p2/Z, [x25, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z20.s, z22.s\n"
    "fmla z27.s, p3/M, z20.s, z9.s\n"
    "fmla z26.s, p3/M, z20.s, z25.s\n"
    "ld1w { z23.s }, p3/Z, [x11]\n"
    "fmla z30.s, p3/M, z18.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [x25, x28, LSL #2]\n"
    "addvl x25, x25, #1\n"
    "fmla z31.s, p3/M, z18.s, z0.s\n"
    "fmla z27.s, p3/M, z18.s, z25.s\n"
    "fmla z26.s, p3/M, z18.s, z24.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.s, p3/M, z17.s, z0.s\n"
    "ld1w { z18.s }, p2/Z, [x24]\n"
    "fmla z31.s, p3/M, z17.s, z19.s\n"
    "fmla z27.s, p3/M, z17.s, z24.s\n"
    "fmla z26.s, p3/M, z17.s, z8.s\n"
    "ld1w { z20.s }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.s, p3/M, z5.s, z19.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "ld1w { z14.s }, p1/Z, [x27]\n"
    "fmla z27.s, p3/M, z5.s, z8.s\n"
    "fmla z26.s, p3/M, z5.s, z22.s\n"
    "ld1w { z19.s }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.s, p3/M, z4.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x24, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z9.s\n"
    "fmla z27.s, p3/M, z4.s, z18.s\n"
    "ld1w { z18.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmla z26.s, p3/M, z4.s, z17.s\n"
    "ld1w { z0.s }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.s, p3/M, z23.s, z9.s\n"
    "ld1w { z13.s }, p1/Z, [x20, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z23.s, z25.s\n"
    "fmla z27.s, p3/M, z23.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x9, LSL #2]\n"
    "fmla z26.s, p3/M, z23.s, z16.s\n"
    "ld1w { z1.s }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.s, p3/M, z21.s, z25.s\n"
    "ld1w { z5.s }, p1/Z, [x14]\n"
    "fmla z31.s, p3/M, z21.s, z24.s\n"
    "fmla z27.s, p3/M, z21.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x24, x28, LSL #2]\n"
    "whilelt p2.s, x15, %x[n_channels]\n"
    "cmp x17, %x[n_channels]\n"
    "addvl x24, x24, #1\n"
    "fmla z26.s, p3/M, z21.s, z18.s\n"
    "ld1w { z2.s }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.s, p3/M, z20.s, z24.s\n"
    "ld1w { z6.s }, p1/Z, [x14, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z20.s, z8.s\n"
    "fmla z27.s, p3/M, z20.s, z18.s\n"
    "ld1w { z11.s }, p1/Z, [x14, x10, LSL #2]\n"
    "fmla z26.s, p3/M, z20.s, z17.s\n"
    "ld1w { z3.s }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.s, p3/M, z19.s, z8.s\n"
    "ld1w { z8.s }, p1/Z, [x20, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z19.s, z22.s\n"
    "ld1w { z10.s }, p1/Z, [x20, x28, LSL #2]\n"
    "fmla z27.s, p3/M, z19.s, z17.s\n"
    "ld1w { z12.s }, p1/Z, [x14, x9, LSL #2]\n"
    "fmla z26.s, p3/M, z19.s, z16.s\n"
    "ld1w { z9.s }, p1/Z, [x14, x12, LSL #2]\n"
    "ld1w { z4.s }, p3/Z, [x11, #-7, MUL VL]\n"
    "addvl x11, x11, #-6\n"
    "fmax z30.s, p3/M, z30.s, z15.s\n"
    "fmax z31.s, p3/M, z31.s, z15.s\n"
    "fmax z27.s, p3/M, z27.s, z15.s\n"
    "fmax z26.s, p3/M, z26.s, z15.s\n"
    "fmin z30.s, p3/M, z30.s, z28.s\n"
    "fmin z31.s, p3/M, z31.s, z28.s\n"
    "fmin z27.s, p3/M, z27.s, z28.s\n"
    "fmin z26.s, p3/M, z26.s, z28.s\n"
    "st1w { z30.s }, p0, [x13]\n"
    "st1w { z31.s }, p0, [x13, x16, LSL #2]\n"
    "addvl x13, x13, #1\n"
    "st1w { z27.s }, p0, [x23]\n"
    "st1w { z26.s }, p0, [x23, x16, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z30, z29\n fmla z30.s, p3/M, z0.s, z5.s\n"
    "movprfx z31, z29\n fmla z31.s, p3/M, z0.s, z6.s\n"
    "ld1w { z22.s }, p2/Z, [x20, x10, LSL #2]\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "movprfx z5, z29\n fmla z5.s, p3/M, z0.s, z7.s\n"
    "fmla z29.s, p3/M, z0.s, z8.s\n"
    "ld1w { z20.s }, p3/Z, [x11]\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "mov p0.b, p2.b\n"
    "add x7, x7, #0x1\n"
    "fmla z30.s, p3/M, z1.s, z6.s\n"
    "ld1w { z6.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z1.s, z9.s\n"
    "add x20, x6, #0x1\n"
    "fmla z5.s, p3/M, z1.s, z8.s\n"
    "fmla z29.s, p3/M, z1.s, z13.s\n"
    "ld1w { z19.s }, p3/Z, [x11, #1, MUL VL]\n"
    "cmp x7, x22\n"
    "csel x6, x6, x20, LT\n"
    "csel x7, x7, XZR, LT\n"
    "fmla z30.s, p3/M, z2.s, z9.s\n"
    "ld1w { z16.s }, p2/Z, [x14, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z5.s, p3/M, z2.s, z13.s\n"
    "fmla z29.s, p3/M, z2.s, z22.s\n"
    "ld1w { z18.s }, p3/Z, [x11, #2, MUL VL]\n"
    "cmp x6, x21\n"
    "fmla z30.s, p3/M, z3.s, z11.s\n"
    "ld1w { z1.s }, p2/Z, [x27, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z12.s\n"
    "fmla z5.s, p3/M, z3.s, z22.s\n"
    "fmla z29.s, p3/M, z3.s, z6.s\n"
    "ld1w { z17.s }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.s, p3/M, z4.s, z12.s\n"
    "ld1w { z0.s }, p2/Z, [x27, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z16.s\n"
    "ld1w { z27.s }, p2/Z, [x27, x10, LSL #2]\n"
    "fmla z5.s, p3/M, z4.s, z6.s\n"
    "fmla z29.s, p3/M, z4.s, z10.s\n"
    "ld1w { z16.s }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z30.s, p3/M, z20.s, z7.s\n"
    "fmla z31.s, p3/M, z20.s, z8.s\n"
    "fmla z5.s, p3/M, z20.s, z14.s\n"
    "fmla z29.s, p3/M, z20.s, z1.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.s, p3/M, z19.s, z8.s\n"
    "ld1w { z26.s }, p2/Z, [x27, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z19.s, z13.s\n"
    "fmla z5.s, p3/M, z19.s, z1.s\n"
    "fmla z29.s, p3/M, z19.s, z0.s\n"
    "ld1w { z25.s }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.s, p3/M, z18.s, z13.s\n"
    "ld1w { z24.s }, p2/Z, [x27, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z18.s, z22.s\n"
    "fmla z5.s, p3/M, z18.s, z0.s\n"
    "fmla z29.s, p3/M, z18.s, z27.s\n"
    "ld1w { z23.s }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.s, p3/M, z17.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [x26]\n"
    "fmla z31.s, p3/M, z17.s, z6.s\n"
    "fmla z5.s, p3/M, z17.s, z27.s\n"
    "fmla z29.s, p3/M, z17.s, z24.s\n"
    "ld1w { z20.s }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.s, p3/M, z16.s, z6.s\n"
    "ld1w { z18.s }, p2/Z, [x26, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z16.s, z10.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x12, LSL #2]\n"
    "fmla z5.s, p3/M, z16.s, z24.s\n"
    "fmla z29.s, p3/M, z16.s, z26.s\n"
    "ld1w { z16.s }, p3/Z, [x11, #-7, MUL VL]\n"
    "fmla z30.s, p3/M, z21.s, z14.s\n"
    "ld1w { z19.s }, p2/Z, [x26, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z21.s, z1.s\n"
    "fmla z5.s, p3/M, z21.s, z22.s\n"
    "fmla z29.s, p3/M, z21.s, z18.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #-6, MUL VL]\n"
    "fmla z30.s, p3/M, z25.s, z1.s\n"
    "ld1w { z8.s }, p2/Z, [x26, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z25.s, z0.s\n"
    "fmla z5.s, p3/M, z25.s, z18.s\n"
    "fmla z29.s, p3/M, z25.s, z17.s\n"
    "ld1w { z9.s }, p3/Z, [x11, #-5, MUL VL]\n"
    "fmla z30.s, p3/M, z23.s, z0.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z23.s, z27.s\n"
    "fmla z5.s, p3/M, z23.s, z17.s\n"
    "fmla z29.s, p3/M, z23.s, z8.s\n"
    "ld1w { z6.s }, p3/Z, [x11, #-4, MUL VL]\n"
    "fmla z30.s, p3/M, z20.s, z27.s\n"
    "ld1w { z0.s }, p2/Z, [x25]\n"
    "fmla z31.s, p3/M, z20.s, z24.s\n"
    "fmla z5.s, p3/M, z20.s, z8.s\n"
    "fmla z29.s, p3/M, z20.s, z11.s\n"
    "ld1w { z4.s }, p3/Z, [x11, #-3, MUL VL]\n"
    "fmla z30.s, p3/M, z16.s, z24.s\n"
    "ld1w { z2.s }, p2/Z, [x25, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z16.s, z26.s\n"
    "ld1w { z27.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z5.s, p3/M, z16.s, z11.s\n"
    "fmla z29.s, p3/M, z16.s, z19.s\n"
    "ld1w { z16.s }, p3/Z, [x11, #-2, MUL VL]\n"
    "fmla z30.s, p3/M, z21.s, z22.s\n"
    "ld1w { z26.s }, p2/Z, [x25, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z21.s, z18.s\n"
    "fmla z5.s, p3/M, z21.s, z0.s\n"
    "fmla z29.s, p3/M, z21.s, z2.s\n"
    "ld1w { z25.s }, p3/Z, [x11, #-1, MUL VL]\n"
    "fmla z30.s, p3/M, z9.s, z18.s\n"
    "ld1w { z24.s }, p2/Z, [x25, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z9.s, z17.s\n"
    "fmla z5.s, p3/M, z9.s, z2.s\n"
    "fmla z29.s, p3/M, z9.s, z26.s\n"
    "ld1w { z23.s }, p3/Z, [x11]\n"
    "fmla z30.s, p3/M, z6.s, z17.s\n"
    "ld1w { z22.s }, p2/Z, [x25, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z6.s, z8.s\n"
    "fmla z5.s, p3/M, z6.s, z26.s\n"
    "fmla z29.s, p3/M, z6.s, z24.s\n"
    "ld1w { z21.s }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.s, p3/M, z4.s, z8.s\n"
    "ld1w { z18.s }, p2/Z, [x24]\n"
    "fmla z31.s, p3/M, z4.s, z11.s\n"
    "fmla z5.s, p3/M, z4.s, z24.s\n"
    "fmla z29.s, p3/M, z4.s, z27.s\n"
    "ld1w { z20.s }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.s, p3/M, z16.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x8, LSL #2]\n"
    "fmla z31.s, p3/M, z16.s, z19.s\n"
    "fmla z5.s, p3/M, z16.s, z27.s\n"
    "fmla z29.s, p3/M, z16.s, z22.s\n"
    "ld1w { z19.s }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.s, p3/M, z25.s, z0.s\n"
    "ld1w { z16.s }, p2/Z, [x24, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z25.s, z2.s\n"
    "fmla z5.s, p3/M, z25.s, z18.s\n"
    "ld1w { z18.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmla z29.s, p3/M, z25.s, z17.s\n"
    "fmla z30.s, p3/M, z23.s, z2.s\n"
    "fmla z31.s, p3/M, z23.s, z26.s\n"
    "fmla z5.s, p3/M, z23.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x9, LSL #2]\n"
    "fmla z29.s, p3/M, z23.s, z16.s\n"
    "fmla z30.s, p3/M, z21.s, z26.s\n"
    "fmla z31.s, p3/M, z21.s, z24.s\n"
    "fmla z5.s, p3/M, z21.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z21.s, z18.s\n"
    "fmla z30.s, p3/M, z20.s, z24.s\n"
    "fmla z31.s, p3/M, z20.s, z27.s\n"
    "fmla z5.s, p3/M, z20.s, z18.s\n"
    "fmla z29.s, p3/M, z20.s, z17.s\n"
    "fmla z30.s, p3/M, z19.s, z27.s\n"
    "fmla z31.s, p3/M, z19.s, z22.s\n"
    "fmla z5.s, p3/M, z19.s, z17.s\n"
    "fmla z29.s, p3/M, z19.s, z16.s\n"
    "fmax z30.s, p3/M, z30.s, z15.s\n"
    "fmax z31.s, p3/M, z31.s, z15.s\n"
    "fmax z5.s, p3/M, z5.s, z15.s\n"
    "fmin z30.s, p3/M, z30.s, z28.s\n"
    "fmin z31.s, p3/M, z31.s, z28.s\n"
    "fmax z29.s, p3/M, z29.s, z15.s\n"
    "fmin z5.s, p3/M, z5.s, z28.s\n"
    "st1w { z30.s }, p0, [x13]\n"
    "fmin z29.s, p3/M, z29.s, z28.s\n"
    "st1w { z31.s }, p0, [x13, x16, LSL #2]\n"
    "st1w { z5.s }, p0, [x23]\n"
    "st1w { z29.s }, p0, [x23, x16, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
