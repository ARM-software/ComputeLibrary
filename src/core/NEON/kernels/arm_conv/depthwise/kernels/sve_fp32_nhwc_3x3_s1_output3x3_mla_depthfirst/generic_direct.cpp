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
    "mov x5, #0x0\n"
    "mov x6, #0x0\n"
    "1:"  // Tile loop
    "str x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x3\n"
    "mov x25, #0x3\n"
    "str x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cntw x8\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "mov x16, #0x0\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x22, x5, x24\n"  // offset = tile_i * ld_input_row
    "ldr x13, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x12, x7, x7\n"
    "cmp x8, %x[n_channels]\n"
    "ld1rw { z15.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mul x21, x5, x23\n"  // offset = tile_i * ld_output_row
    "add x11, x12, x7\n"
    "add x10, x17, x17\n"
    "ld1rw { z14.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "madd x22, x6, x7, x22\n"  // offset += tile_j * ld_input_col
    "ld1w { z31.s }, p3/Z, [x14]\n"
    "ld1w { z0.s }, p3/Z, [x14, #1, MUL VL]\n"
    "add x9, x11, x7\n"
    "ld1w { z1.s }, p3/Z, [x14, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x14, #3, MUL VL]\n"
    "sub x20, XZR, x8\n"
    "madd x21, x6, x17, x21\n"  // offset += tile_j * ld_output_col
    "ld1w { z3.s }, p3/Z, [x14, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x14, #5, MUL VL]\n"
    "mul x22, x22, x26\n"  // offset *= kernel_stride * output_size
    "ld1w { z5.s }, p3/Z, [x14, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "mul x21, x21, x25\n"  // offset *= output_tile_size
    "add x15, x15, x22, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x28, x15, x24, LSL #2\n"
    "add x27, x28, x24, LSL #2\n"
    "ld1w { z10.s }, p2/Z, [x15]\n"
    "ld1w { z11.s }, p2/Z, [x15, x9, LSL #2]\n"
    "add x26, x27, x24, LSL #2\n"
    "add x13, x13, x21, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "add x25, x26, x24, LSL #2\n"
    "ld1w { z7.s }, p3/Z, [x14, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x14, #-7, MUL VL]\n"
    "add x24, x13, x23, LSL #2\n"
    "ld1w { z9.s }, p2/Z, [x27, x12, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x25]\n"
    "addvl x14, x14, #-6\n"
    "add x23, x24, x23, LSL #2\n"
    "ld1w { z13.s }, p2/Z, [x28, x12, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z30, z31\n fmla z30.s, p3/M, z7.s, z9.s\n"
    "movprfx z29, z31\n fmla z29.s, p3/M, z8.s, z9.s\n"
    "whilelt p1.s, x8, %x[n_channels]\n"
    "incw x16\n"
    "movprfx z28, z31\n fmla z28.s, p3/M, z6.s, z9.s\n"
    "movprfx z27, z31\n fmla z27.s, p3/M, z5.s, z9.s\n"
    "incw x8\n"
    "mov p0.b, p2.b\n"
    "movprfx z26, z31\n fmla z26.s, p3/M, z4.s, z9.s\n"
    "movprfx z25, z31\n fmla z25.s, p3/M, z3.s, z9.s\n"
    "incw x20\n"
    "movprfx z24, z31\n fmla z24.s, p3/M, z2.s, z9.s\n"
    "movprfx z23, z31\n fmla z23.s, p3/M, z0.s, z9.s\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z10.s\n"
    "ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x27, x7, LSL #2]\n"
    "fmla z27.s, p3/M, z2.s, z13.s\n"
    "fmla z26.s, p3/M, z1.s, z13.s\n"
    "fmla z25.s, p3/M, z0.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x9, LSL #2]\n"
    "movprfx z21, z31\n fmla z21.s, p3/M, z1.s, z9.s\n"
    "ld1w { z31.s }, p3/Z, [x14]\n"
    "fmla z30.s, p3/M, z6.s, z17.s\n"
    "fmla z29.s, p3/M, z5.s, z13.s\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x15, x7, LSL #2]\n"
    "fmla z27.s, p3/M, z4.s, z17.s\n"
    "fmla z23.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x15, x11, LSL #2]\n"
    "fmla z26.s, p3/M, z3.s, z17.s\n"
    "fmla z21.s, p3/M, z0.s, z17.s\n"
    "fmla z24.s, p3/M, z1.s, z17.s\n"
    "fmla z30.s, p3/M, z0.s, z18.s\n"
    "fmla z29.s, p3/M, z7.s, z17.s\n"
    "ld1w { z20.s }, p2/Z, [x28]\n"
    "fmla z28.s, p3/M, z1.s, z16.s\n"
    "fmla z25.s, p3/M, z4.s, z22.s\n"
    "fmla z23.s, p3/M, z1.s, z22.s\n"
    "fmla z26.s, p3/M, z5.s, z22.s\n"
    "fmla z21.s, p3/M, z2.s, z22.s\n"
    "fmla z27.s, p3/M, z0.s, z20.s\n"
    "fmla z30.s, p3/M, z2.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "fmla z29.s, p3/M, z1.s, z18.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z22.s\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "fmla z25.s, p3/M, z2.s, z16.s\n"
    "fmla z27.s, p3/M, z6.s, z17.s\n"
    "ld1w { z19.s }, p2/Z, [x28, x7, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z22.s\n"
    "ld1w { z18.s }, p2/Z, [x26, x12, LSL #2]\n"
    "fmla z29.s, p3/M, z3.s, z20.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x7, LSL #2]\n"
    "fmla z21.s, p3/M, z4.s, z18.s\n"
    "fmla z23.s, p3/M, z3.s, z18.s\n"
    "fmla z26.s, p3/M, z7.s, z18.s\n"
    "fmla z24.s, p3/M, z5.s, z18.s\n"
    "fmla z25.s, p3/M, z6.s, z18.s\n"
    "fmla z27.s, p3/M, z8.s, z18.s\n"
    "fmla z30.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z6.s, z16.s\n"
    "fmla z29.s, p3/M, z4.s, z19.s\n"
    "fmla z23.s, p3/M, z5.s, z17.s\n"
    "fmla z26.s, p3/M, z0.s, z19.s\n"
    "fmla z24.s, p3/M, z7.s, z16.s\n"
    "ld1w { z18.s }, p2/Z, [x25, x11, LSL #2]\n"
    "fmla z25.s, p3/M, z8.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x11, LSL #2]\n"
    "fmla z27.s, p3/M, z1.s, z19.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x7, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "fmla z21.s, p3/M, z8.s, z18.s\n"
    "fmla z23.s, p3/M, z7.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x26, x11, LSL #2]\n"
    "addvl x26, x26, #1\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "fmla z28.s, p3/M, z4.s, z16.s\n"
    "fmla z26.s, p3/M, z2.s, z16.s\n"
    "fmla z25.s, p3/M, z1.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x15, x12, LSL #2]\n"
    "fmla z24.s, p3/M, z4.s, z17.s\n"
    "addvl x15, x15, #1\n"
    "fmla z21.s, p3/M, z3.s, z17.s\n"
    "fmla z27.s, p3/M, z7.s, z17.s\n"
    "fmla z23.s, p3/M, z4.s, z19.s\n"
    "ld1w { z4.s }, p3/Z, [x14, #5, MUL VL]\n"
    "fmla z26.s, p3/M, z6.s, z17.s\n"
    "ld1w { z18.s }, p2/Z, [x27]\n"
    "fmla z29.s, p3/M, z2.s, z16.s\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ld1w { z1.s }, p3/Z, [x14, #2, MUL VL]\n"
    "ld1w { z10.s }, p1/Z, [x15]\n"
    "fmla z28.s, p3/M, z0.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x27, x9, LSL #2]\n"
    "fmla z25.s, p3/M, z7.s, z19.s\n"
    "addvl x27, x27, #1\n"
    "fmla z21.s, p3/M, z5.s, z19.s\n"
    "fmla z24.s, p3/M, z0.s, z18.s\n"
    "ld1w { z0.s }, p3/Z, [x14, #1, MUL VL]\n"
    "fmla z26.s, p3/M, z8.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x12, LSL #2]\n"
    "fmla z27.s, p3/M, z3.s, z18.s\n"
    "addvl x25, x25, #1\n"
    "fmla z23.s, p3/M, z2.s, z17.s\n"
    "fmla z29.s, p3/M, z6.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z15.s\n"
    "ld1w { z2.s }, p3/Z, [x14, #3, MUL VL]\n"
    "fmla z28.s, p3/M, z8.s, z17.s\n"
    "fmla z25.s, p3/M, z5.s, z17.s\n"
    "ld1w { z3.s }, p3/Z, [x14, #4, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x14, #6, MUL VL]\n"
    "fmla z24.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "whilelt p2.s, x16, %x[n_channels]\n"
    "cmp x8, %x[n_channels]\n"
    "fmax z27.s, p3/M, z27.s, z15.s\n"
    "fmax z26.s, p3/M, z26.s, z15.s\n"
    "ld1w { z9.s }, p1/Z, [x27, x12, LSL #2]\n"
    "ld1w { z11.s }, p1/Z, [x15, x9, LSL #2]\n"
    "fmla z23.s, p3/M, z6.s, z16.s\n"
    "fmax z29.s, p3/M, z29.s, z15.s\n"
    "ld1w { z6.s }, p3/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "fmax z28.s, p3/M, z28.s, z15.s\n"
    "fmax z25.s, p3/M, z25.s, z15.s\n"
    "ld1w { z12.s }, p1/Z, [x25]\n"
    "ld1w { z13.s }, p1/Z, [x28, x12, LSL #2]\n"
    "fmax z24.s, p3/M, z24.s, z15.s\n"
    "fmax z21.s, p3/M, z21.s, z15.s\n"
    "fmin z29.s, p3/M, z29.s, z14.s\n"
    "fmin z30.s, p3/M, z30.s, z14.s\n"
    "ld1w { z7.s }, p3/Z, [x14, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x14, #-7, MUL VL]\n"
    "fmax z23.s, p3/M, z23.s, z15.s\n"
    "fmin z28.s, p3/M, z28.s, z14.s\n"
    "fmin z27.s, p3/M, z27.s, z14.s\n"
    "fmin z26.s, p3/M, z26.s, z14.s\n"
    "fmin z25.s, p3/M, z25.s, z14.s\n"
    "fmin z24.s, p3/M, z24.s, z14.s\n"
    "st1w { z29.s }, p0, [x13]\n"
    "fmin z21.s, p3/M, z21.s, z14.s\n"
    "fmin z23.s, p3/M, z23.s, z14.s\n"
    "st1w { z30.s }, p0, [x13, x17, LSL #2]\n"
    "st1w { z28.s }, p0, [x13, x10, LSL #2]\n"
    "addvl x13, x13, #1\n"
    "addvl x14, x14, #-6\n"
    "st1w { z27.s }, p0, [x24]\n"
    "st1w { z26.s }, p0, [x24, x17, LSL #2]\n"
    "st1w { z25.s }, p0, [x24, x10, LSL #2]\n"
    "addvl x24, x24, #1\n"
    "st1w { z24.s }, p0, [x23]\n"
    "st1w { z21.s }, p0, [x23, x17, LSL #2]\n"
    "st1w { z23.s }, p0, [x23, x10, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z30, z31\n fmla z30.s, p3/M, z7.s, z9.s\n"
    "movprfx z29, z31\n fmla z29.s, p3/M, z8.s, z9.s\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z28, z31\n fmla z28.s, p3/M, z6.s, z9.s\n"
    "movprfx z27, z31\n fmla z27.s, p3/M, z5.s, z9.s\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "movprfx z26, z31\n fmla z26.s, p3/M, z4.s, z9.s\n"
    "movprfx z25, z31\n fmla z25.s, p3/M, z3.s, z9.s\n"
    "mov p0.b, p2.b\n"
    "movprfx z24, z31\n fmla z24.s, p3/M, z2.s, z9.s\n"
    "movprfx z23, z31\n fmla z23.s, p3/M, z0.s, z9.s\n"
    "add x6, x6, #0x1\n"
    "add x20, x5, #0x1\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z10.s\n"
    "ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]\n"
    "cmp x6, x22\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x27, x7, LSL #2]\n"
    "fmla z27.s, p3/M, z2.s, z13.s\n"
    "csel x5, x5, x20, LT\n"
    "fmla z26.s, p3/M, z1.s, z13.s\n"
    "fmla z25.s, p3/M, z0.s, z13.s\n"
    "csel x6, x6, XZR, LT\n"
    "fmla z24.s, p3/M, z6.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x9, LSL #2]\n"
    "movprfx z21, z31\n fmla z21.s, p3/M, z1.s, z9.s\n"
    "fmla z30.s, p3/M, z6.s, z17.s\n"
    "fmla z29.s, p3/M, z5.s, z13.s\n"
    "cmp x5, x21\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x15, x7, LSL #2]\n"
    "fmla z27.s, p3/M, z4.s, z17.s\n"
    "fmla z23.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x15, x11, LSL #2]\n"
    "fmla z26.s, p3/M, z3.s, z17.s\n"
    "fmla z21.s, p3/M, z0.s, z17.s\n"
    "fmla z24.s, p3/M, z1.s, z17.s\n"
    "fmla z30.s, p3/M, z0.s, z18.s\n"
    "fmla z29.s, p3/M, z7.s, z17.s\n"
    "ld1w { z20.s }, p2/Z, [x28]\n"
    "fmla z28.s, p3/M, z1.s, z16.s\n"
    "fmla z25.s, p3/M, z4.s, z22.s\n"
    "fmla z23.s, p3/M, z1.s, z22.s\n"
    "fmla z26.s, p3/M, z5.s, z22.s\n"
    "fmla z21.s, p3/M, z2.s, z22.s\n"
    "fmla z27.s, p3/M, z0.s, z20.s\n"
    "fmla z30.s, p3/M, z2.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "fmla z29.s, p3/M, z1.s, z18.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z22.s\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "fmla z25.s, p3/M, z2.s, z16.s\n"
    "fmla z27.s, p3/M, z6.s, z17.s\n"
    "ld1w { z19.s }, p2/Z, [x28, x7, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z22.s\n"
    "ld1w { z18.s }, p2/Z, [x26, x12, LSL #2]\n"
    "fmla z29.s, p3/M, z3.s, z20.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x7, LSL #2]\n"
    "fmla z21.s, p3/M, z4.s, z18.s\n"
    "fmla z23.s, p3/M, z3.s, z18.s\n"
    "fmla z26.s, p3/M, z7.s, z18.s\n"
    "fmla z24.s, p3/M, z5.s, z18.s\n"
    "fmla z25.s, p3/M, z6.s, z18.s\n"
    "fmla z27.s, p3/M, z8.s, z18.s\n"
    "fmla z30.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z6.s, z16.s\n"
    "fmla z29.s, p3/M, z4.s, z19.s\n"
    "fmla z23.s, p3/M, z5.s, z17.s\n"
    "fmla z26.s, p3/M, z0.s, z19.s\n"
    "fmla z24.s, p3/M, z7.s, z16.s\n"
    "ld1w { z18.s }, p2/Z, [x25, x11, LSL #2]\n"
    "fmla z25.s, p3/M, z8.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x11, LSL #2]\n"
    "fmla z27.s, p3/M, z1.s, z19.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x7, LSL #2]\n"
    "fmla z21.s, p3/M, z8.s, z18.s\n"
    "fmla z23.s, p3/M, z7.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x26, x11, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "fmla z28.s, p3/M, z4.s, z16.s\n"
    "fmla z26.s, p3/M, z2.s, z16.s\n"
    "fmla z25.s, p3/M, z1.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x15, x12, LSL #2]\n"
    "fmla z24.s, p3/M, z4.s, z17.s\n"
    "fmla z21.s, p3/M, z3.s, z17.s\n"
    "fmla z27.s, p3/M, z7.s, z17.s\n"
    "fmla z23.s, p3/M, z4.s, z19.s\n"
    "fmla z26.s, p3/M, z6.s, z17.s\n"
    "ld1w { z18.s }, p2/Z, [x27]\n"
    "fmla z29.s, p3/M, z2.s, z16.s\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "fmla z28.s, p3/M, z0.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x27, x9, LSL #2]\n"
    "fmla z25.s, p3/M, z7.s, z19.s\n"
    "fmla z21.s, p3/M, z5.s, z19.s\n"
    "fmla z24.s, p3/M, z0.s, z18.s\n"
    "fmla z26.s, p3/M, z8.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x12, LSL #2]\n"
    "fmla z27.s, p3/M, z3.s, z18.s\n"
    "fmla z23.s, p3/M, z2.s, z17.s\n"
    "fmla z29.s, p3/M, z6.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z15.s\n"
    "fmla z28.s, p3/M, z8.s, z17.s\n"
    "fmla z25.s, p3/M, z5.s, z17.s\n"
    "fmla z24.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "fmax z27.s, p3/M, z27.s, z15.s\n"
    "fmax z26.s, p3/M, z26.s, z15.s\n"
    "fmin z30.s, p3/M, z30.s, z14.s\n"
    "fmla z23.s, p3/M, z6.s, z16.s\n"
    "fmax z29.s, p3/M, z29.s, z15.s\n"
    "fmax z28.s, p3/M, z28.s, z15.s\n"
    "fmax z25.s, p3/M, z25.s, z15.s\n"
    "fmin z27.s, p3/M, z27.s, z14.s\n"
    "fmin z26.s, p3/M, z26.s, z14.s\n"
    "fmax z24.s, p3/M, z24.s, z15.s\n"
    "fmax z21.s, p3/M, z21.s, z15.s\n"
    "fmax z23.s, p3/M, z23.s, z15.s\n"
    "fmin z29.s, p3/M, z29.s, z14.s\n"
    "fmin z28.s, p3/M, z28.s, z14.s\n"
    "fmin z25.s, p3/M, z25.s, z14.s\n"
    "st1w { z27.s }, p0, [x24]\n"
    "fmin z24.s, p3/M, z24.s, z14.s\n"
    "fmin z21.s, p3/M, z21.s, z14.s\n"
    "st1w { z26.s }, p0, [x24, x17, LSL #2]\n"
    "fmin z23.s, p3/M, z23.s, z14.s\n"
    "st1w { z29.s }, p0, [x13]\n"
    "st1w { z30.s }, p0, [x13, x17, LSL #2]\n"
    "st1w { z28.s }, p0, [x13, x10, LSL #2]\n"
    "st1w { z25.s }, p0, [x24, x10, LSL #2]\n"
    "st1w { z24.s }, p0, [x23]\n"
    "st1w { z21.s }, p0, [x23, x17, LSL #2]\n"
    "st1w { z23.s }, p0, [x23, x10, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
