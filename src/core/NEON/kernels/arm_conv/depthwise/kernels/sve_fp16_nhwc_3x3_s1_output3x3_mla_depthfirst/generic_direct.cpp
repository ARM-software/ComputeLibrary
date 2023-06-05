/*
 * Copyright (c) 2021, 2023 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

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
    "mov x13, #0x0\n"
    "mov x8, #0x0\n"
    "1:"  // Tile loop
    "str x13, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x25, #0x3\n"
    "mov x24, #0x3\n"
    "str x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "mul x22, x13, x23\n"  // offset = tile_i * ld_input_row
    "ldr x21, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x22, x8, x17, x22\n"  // offset += tile_j * ld_input_col
    "ldr x16, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "cnth x15\n"
    "mul x20, x13, x21\n"  // offset = tile_i * ld_output_row
    "ldr x14, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x12, x17, x17\n"
    "mul x22, x22, x25\n"  // offset *= kernel_stride * output_size
    "add x14, x14, x22, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "ldr x11, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x10, x14, x23, LSL #1\n"
    "madd x20, x8, x16, x20\n"  // offset += tile_j * ld_output_col
    "add x9, x10, x23, LSL #1\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1h { z14.h }, p3/Z, [x13]\n"
    "mul x20, x20, x24\n"  // offset *= output_tile_size
    "ld1h { z0.h }, p3/Z, [x13, #1, MUL VL]\n"
    "ld1h { z1.h }, p3/Z, [x13, #2, MUL VL]\n"
    "add x28, x9, x23, LSL #1\n"
    "ld1h { z2.h }, p3/Z, [x13, #3, MUL VL]\n"
    "ld1h { z3.h }, p3/Z, [x13, #4, MUL VL]\n"
    "add x27, x12, x17\n"
    "add x11, x11, x20, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "ld1h { z4.h }, p3/Z, [x13, #5, MUL VL]\n"
    "ld1h { z5.h }, p3/Z, [x13, #6, MUL VL]\n"
    "add x26, x28, x23, LSL #1\n"
    "add x25, x27, x17\n"
    "ld1h { z6.h }, p3/Z, [x13, #7, MUL VL]\n"
    "addvl x13, x13, #16\n"
    "add x24, x11, x21, LSL #1\n"
    "ld1rh { z31.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "cmp x15, %x[n_channels]\n"
    "add x23, x24, x21, LSL #1\n"
    "ld1rh { z30.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1h { z7.h }, p3/Z, [x13, #-8, MUL VL]\n"
    "add x22, x16, x16\n"
    "mov x21, #0x0\n"
    "ld1h { z8.h }, p3/Z, [x13, #-7, MUL VL]\n"
    "ld1h { z9.h }, p2/Z, [x9, x12, LSL #1]\n"
    "sub x20, XZR, x15\n"
    "ld1h { z10.h }, p2/Z, [x14]\n"
    "ld1h { z11.h }, p2/Z, [x14, x25, LSL #1]\n"
    "addvl x13, x13, #-6\n"
    "ld1h { z12.h }, p2/Z, [x26]\n"
    "ld1h { z13.h }, p2/Z, [x10, x12, LSL #1]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z29, z14\n fmla z29.h, p3/M, z7.h, z9.h\n"
    "movprfx z28, z14\n fmla z28.h, p3/M, z8.h, z9.h\n"
    "whilelt p1.h, x15, %x[n_channels]\n"
    "inch x21\n"
    "movprfx z27, z14\n fmla z27.h, p3/M, z6.h, z9.h\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "inch x15\n"
    "mov p0.b, p2.b\n"
    "movprfx z26, z14\n fmla z26.h, p3/M, z5.h, z9.h\n"
    "movprfx z25, z14\n fmla z25.h, p3/M, z4.h, z9.h\n"
    "inch x20\n"
    "movprfx z24, z14\n fmla z24.h, p3/M, z3.h, z9.h\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "ld1h { z23.h }, p2/Z, [x9, x27, LSL #1]\n"
    "fmla z27.h, p3/M, z2.h, z11.h\n"
    "ld1h { z18.h }, p2/Z, [x9, x17, LSL #1]\n"
    "movprfx z22, z14\n fmla z22.h, p3/M, z2.h, z9.h\n"
    "fmla z29.h, p3/M, z6.h, z18.h\n"
    "movprfx z21, z14\n fmla z21.h, p3/M, z0.h, z9.h\n"
    "fmla z28.h, p3/M, z5.h, z13.h\n"
    "fmla z27.h, p3/M, z3.h, z13.h\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "fmla z25.h, p3/M, z1.h, z13.h\n"
    "fmla z24.h, p3/M, z0.h, z13.h\n"
    "ld1h { z17.h }, p2/Z, [x14, x17, LSL #1]\n"
    "fmla z22.h, p3/M, z6.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x25, LSL #1]\n"
    "movprfx z20, z14\n fmla z20.h, p3/M, z1.h, z9.h\n"
    "fmla z29.h, p3/M, z0.h, z17.h\n"
    "ld1h { z14.h }, p3/Z, [x13]\n"
    "fmla z21.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x27, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z18.h\n"
    "fmla z20.h, p3/M, z0.h, z18.h\n"
    "fmla z26.h, p3/M, z4.h, z18.h\n"
    "fmla z25.h, p3/M, z3.h, z18.h\n"
    "fmla z22.h, p3/M, z1.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x10]\n"
    "fmla z29.h, p3/M, z2.h, z16.h\n"
    "fmla z27.h, p3/M, z1.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x28]\n"
    "fmla z24.h, p3/M, z4.h, z23.h\n"
    "fmla z28.h, p3/M, z1.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x25, LSL #1]\n"
    "fmla z20.h, p3/M, z2.h, z23.h\n"
    "fmla z21.h, p3/M, z1.h, z23.h\n"
    "fmla z29.h, p3/M, z8.h, z23.h\n"
    "fmla z27.h, p3/M, z7.h, z23.h\n"
    "fmla z25.h, p3/M, z5.h, z23.h\n"
    "fmla z26.h, p3/M, z0.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x28, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z3.h, z18.h\n"
    "fmla z24.h, p3/M, z2.h, z16.h\n"
    "fmla z20.h, p3/M, z4.h, z17.h\n"
    "fmla z21.h, p3/M, z3.h, z17.h\n"
    "fmla z28.h, p3/M, z3.h, z19.h\n"
    "fmla z27.h, p3/M, z5.h, z16.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x25, LSL #1]\n"
    "ld1h { z16.h }, p2/Z, [x26, x17, LSL #1]\n"
    "fmla z26.h, p3/M, z6.h, z18.h\n"
    "fmla z25.h, p3/M, z7.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x10, x17, LSL #1]\n"
    "fmla z22.h, p3/M, z5.h, z17.h\n"
    "fmla z24.h, p3/M, z6.h, z17.h\n"
    "fmla z21.h, p3/M, z5.h, z19.h\n"
    "fmla z20.h, p3/M, z6.h, z16.h\n"
    "fmla z26.h, p3/M, z8.h, z17.h\n"
    "fmla z22.h, p3/M, z7.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x27, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z18.h\n"
    "fmla z25.h, p3/M, z0.h, z18.h\n"
    "fmla z24.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x27, LSL #1]\n"
    "fmla z20.h, p3/M, z8.h, z17.h\n"
    "addvl x10, x10, #1\n"
    "fmla z21.h, p3/M, z7.h, z17.h\n"
    "fmla z28.h, p3/M, z4.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x27, LSL #1]\n"
    "fmla z26.h, p3/M, z1.h, z18.h\n"
    "fmla z29.h, p3/M, z5.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x28, x17, LSL #1]\n"
    "addvl x28, x28, #1\n"
    "fmla z27.h, p3/M, z4.h, z16.h\n"
    "fmla z25.h, p3/M, z2.h, z16.h\n"
    "fmla z24.h, p3/M, z1.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z4.h, z17.h\n"
    "addvl x14, x14, #1\n"
    "fmla z20.h, p3/M, z3.h, z17.h\n"
    "fmla z21.h, p3/M, z4.h, z19.h\n"
    "ld1h { z4.h }, p3/Z, [x13, #5, MUL VL]\n"
    "ld1h { z10.h }, p1/Z, [x14]\n"
    "fmla z26.h, p3/M, z7.h, z17.h\n"
    "fmla z25.h, p3/M, z6.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x9]\n"
    "fmla z28.h, p3/M, z2.h, z16.h\n"
    "fmla z29.h, p3/M, z1.h, z16.h\n"
    "fmax z29.h, p3/M, z29.h, z31.h\n"
    "ld1h { z1.h }, p3/Z, [x13, #2, MUL VL]\n"
    "fmla z27.h, p3/M, z0.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x9, x25, LSL #1]\n"
    "fmla z24.h, p3/M, z7.h, z19.h\n"
    "addvl x9, x9, #1\n"
    "fmla z20.h, p3/M, z5.h, z19.h\n"
    "fmla z22.h, p3/M, z0.h, z18.h\n"
    "ld1h { z0.h }, p3/Z, [x13, #1, MUL VL]\n"
    "fmin z29.h, p3/M, z29.h, z30.h\n"
    "fmla z21.h, p3/M, z2.h, z17.h\n"
    "fmla z25.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmax z25.h, p3/M, z25.h, z31.h\n"
    "fmla z28.h, p3/M, z6.h, z18.h\n"
    "fmla z26.h, p3/M, z3.h, z18.h\n"
    "fmax z28.h, p3/M, z28.h, z31.h\n"
    "fmax z26.h, p3/M, z26.h, z31.h\n"
    "fmla z27.h, p3/M, z8.h, z17.h\n"
    "fmla z24.h, p3/M, z5.h, z17.h\n"
    "fmax z27.h, p3/M, z27.h, z31.h\n"
    "fmax z24.h, p3/M, z24.h, z31.h\n"
    "fmla z22.h, p3/M, z8.h, z16.h\n"
    "fmla z20.h, p3/M, z7.h, z16.h\n"
    "fmax z22.h, p3/M, z22.h, z31.h\n"
    "fmax z20.h, p3/M, z20.h, z31.h\n"
    "fmla z21.h, p3/M, z6.h, z16.h\n"
    "fmax z21.h, p3/M, z21.h, z31.h\n"
    "addvl x26, x26, #1\n"
    "ld1h { z2.h }, p3/Z, [x13, #3, MUL VL]\n"
    "ld1h { z3.h }, p3/Z, [x13, #4, MUL VL]\n"
    "ld1h { z5.h }, p3/Z, [x13, #6, MUL VL]\n"
    "whilelt p2.h, x21, %x[n_channels]\n"
    "cmp x15, %x[n_channels]\n"
    "ld1h { z6.h }, p3/Z, [x13, #7, MUL VL]\n"
    "addvl x13, x13, #16\n"
    "fmin z28.h, p3/M, z28.h, z30.h\n"
    "ld1h { z9.h }, p1/Z, [x9, x12, LSL #1]\n"
    "fmin z27.h, p3/M, z27.h, z30.h\n"
    "fmin z26.h, p3/M, z26.h, z30.h\n"
    "ld1h { z11.h }, p1/Z, [x14, x25, LSL #1]\n"
    "ld1h { z12.h }, p1/Z, [x26]\n"
    "fmin z25.h, p3/M, z25.h, z30.h\n"
    "fmin z24.h, p3/M, z24.h, z30.h\n"
    "ld1h { z13.h }, p1/Z, [x10, x12, LSL #1]\n"
    "st1h { z28.h }, p0, [x11]\n"
    "fmin z22.h, p3/M, z22.h, z30.h\n"
    "fmin z20.h, p3/M, z20.h, z30.h\n"
    "st1h { z29.h }, p0, [x11, x16, LSL #1]\n"
    "ld1h { z7.h }, p3/Z, [x13, #-8, MUL VL]\n"
    "fmin z21.h, p3/M, z21.h, z30.h\n"
    "st1h { z27.h }, p0, [x11, x22, LSL #1]\n"
    "addvl x11, x11, #1\n"
    "ld1h { z8.h }, p3/Z, [x13, #-7, MUL VL]\n"
    "st1h { z26.h }, p0, [x24]\n"
    "addvl x13, x13, #-6\n"
    "st1h { z25.h }, p0, [x24, x16, LSL #1]\n"
    "st1h { z24.h }, p0, [x24, x22, LSL #1]\n"
    "addvl x24, x24, #1\n"
    "st1h { z22.h }, p0, [x23]\n"
    "st1h { z20.h }, p0, [x23, x16, LSL #1]\n"
    "st1h { z21.h }, p0, [x23, x22, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z29, z14\n fmla z29.h, p3/M, z7.h, z9.h\n"
    "movprfx z28, z14\n fmla z28.h, p3/M, z8.h, z9.h\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z27, z14\n fmla z27.h, p3/M, z6.h, z9.h\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "add x8, x8, #0x1\n"
    "movprfx z26, z14\n fmla z26.h, p3/M, z5.h, z9.h\n"
    "movprfx z25, z14\n fmla z25.h, p3/M, z4.h, z9.h\n"
    "cmp x8, x20\n"
    "add x21, x13, #0x1\n"
    "movprfx z24, z14\n fmla z24.h, p3/M, z3.h, z9.h\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "ld1h { z23.h }, p2/Z, [x9, x27, LSL #1]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z27.h, p3/M, z2.h, z11.h\n"
    "ld1h { z18.h }, p2/Z, [x9, x17, LSL #1]\n"
    "movprfx z22, z14\n fmla z22.h, p3/M, z2.h, z9.h\n"
    "csel x13, x13, x21, LT\n"
    "fmla z29.h, p3/M, z6.h, z18.h\n"
    "movprfx z21, z14\n fmla z21.h, p3/M, z0.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "csel x8, x8, XZR, LT\n"
    "fmla z28.h, p3/M, z5.h, z13.h\n"
    "fmla z27.h, p3/M, z3.h, z13.h\n"
    "cmp x13, x20\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "fmla z25.h, p3/M, z1.h, z13.h\n"
    "fmla z24.h, p3/M, z0.h, z13.h\n"
    "ld1h { z17.h }, p2/Z, [x14, x17, LSL #1]\n"
    "fmla z22.h, p3/M, z6.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x25, LSL #1]\n"
    "movprfx z20, z14\n fmla z20.h, p3/M, z1.h, z9.h\n"
    "fmla z29.h, p3/M, z0.h, z17.h\n"
    "fmla z21.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x27, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z18.h\n"
    "fmla z20.h, p3/M, z0.h, z18.h\n"
    "fmla z26.h, p3/M, z4.h, z18.h\n"
    "fmla z25.h, p3/M, z3.h, z18.h\n"
    "fmla z22.h, p3/M, z1.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x10]\n"
    "fmla z29.h, p3/M, z2.h, z16.h\n"
    "fmla z27.h, p3/M, z1.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x28]\n"
    "fmla z24.h, p3/M, z4.h, z23.h\n"
    "fmla z28.h, p3/M, z1.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x25, LSL #1]\n"
    "fmla z20.h, p3/M, z2.h, z23.h\n"
    "fmla z21.h, p3/M, z1.h, z23.h\n"
    "fmla z29.h, p3/M, z8.h, z23.h\n"
    "fmla z27.h, p3/M, z7.h, z23.h\n"
    "fmla z25.h, p3/M, z5.h, z23.h\n"
    "fmla z26.h, p3/M, z0.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x28, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z3.h, z18.h\n"
    "fmla z24.h, p3/M, z2.h, z16.h\n"
    "fmla z20.h, p3/M, z4.h, z17.h\n"
    "fmla z21.h, p3/M, z3.h, z17.h\n"
    "fmla z28.h, p3/M, z3.h, z19.h\n"
    "fmla z27.h, p3/M, z5.h, z16.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x25, LSL #1]\n"
    "ld1h { z16.h }, p2/Z, [x26, x17, LSL #1]\n"
    "fmla z26.h, p3/M, z6.h, z18.h\n"
    "fmla z25.h, p3/M, z7.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x10, x17, LSL #1]\n"
    "fmla z22.h, p3/M, z5.h, z17.h\n"
    "fmla z24.h, p3/M, z6.h, z17.h\n"
    "fmla z21.h, p3/M, z5.h, z19.h\n"
    "fmla z20.h, p3/M, z6.h, z16.h\n"
    "fmla z26.h, p3/M, z8.h, z17.h\n"
    "fmla z22.h, p3/M, z7.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x27, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z18.h\n"
    "fmla z25.h, p3/M, z0.h, z18.h\n"
    "fmla z24.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x27, LSL #1]\n"
    "fmla z20.h, p3/M, z8.h, z17.h\n"
    "fmla z21.h, p3/M, z7.h, z17.h\n"
    "fmla z28.h, p3/M, z4.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x27, LSL #1]\n"
    "fmla z26.h, p3/M, z1.h, z18.h\n"
    "fmla z29.h, p3/M, z5.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x28, x17, LSL #1]\n"
    "fmla z27.h, p3/M, z4.h, z16.h\n"
    "fmla z25.h, p3/M, z2.h, z16.h\n"
    "fmla z24.h, p3/M, z1.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z4.h, z17.h\n"
    "fmla z20.h, p3/M, z3.h, z17.h\n"
    "fmla z21.h, p3/M, z4.h, z19.h\n"
    "fmla z26.h, p3/M, z7.h, z17.h\n"
    "fmla z25.h, p3/M, z6.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x9]\n"
    "fmla z28.h, p3/M, z2.h, z16.h\n"
    "fmla z29.h, p3/M, z1.h, z16.h\n"
    "fmax z29.h, p3/M, z29.h, z31.h\n"
    "fmin z29.h, p3/M, z29.h, z30.h\n"
    "fmla z27.h, p3/M, z0.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x9, x25, LSL #1]\n"
    "fmla z24.h, p3/M, z7.h, z19.h\n"
    "fmla z20.h, p3/M, z5.h, z19.h\n"
    "fmla z22.h, p3/M, z0.h, z18.h\n"
    "fmla z21.h, p3/M, z2.h, z17.h\n"
    "fmla z25.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmax z25.h, p3/M, z25.h, z31.h\n"
    "fmla z28.h, p3/M, z6.h, z18.h\n"
    "fmla z26.h, p3/M, z3.h, z18.h\n"
    "fmax z28.h, p3/M, z28.h, z31.h\n"
    "fmax z26.h, p3/M, z26.h, z31.h\n"
    "fmla z27.h, p3/M, z8.h, z17.h\n"
    "fmla z24.h, p3/M, z5.h, z17.h\n"
    "fmax z27.h, p3/M, z27.h, z31.h\n"
    "fmax z24.h, p3/M, z24.h, z31.h\n"
    "fmla z22.h, p3/M, z8.h, z16.h\n"
    "fmla z20.h, p3/M, z7.h, z16.h\n"
    "fmax z22.h, p3/M, z22.h, z31.h\n"
    "fmax z20.h, p3/M, z20.h, z31.h\n"
    "fmla z21.h, p3/M, z6.h, z16.h\n"
    "fmax z21.h, p3/M, z21.h, z31.h\n"
    "fmin z28.h, p3/M, z28.h, z30.h\n"
    "st1h { z28.h }, p0, [x11]\n"
    "fmin z27.h, p3/M, z27.h, z30.h\n"
    "fmin z26.h, p3/M, z26.h, z30.h\n"
    "st1h { z29.h }, p0, [x11, x16, LSL #1]\n"
    "fmin z25.h, p3/M, z25.h, z30.h\n"
    "fmin z24.h, p3/M, z24.h, z30.h\n"
    "st1h { z27.h }, p0, [x11, x22, LSL #1]\n"
    "fmin z22.h, p3/M, z22.h, z30.h\n"
    "fmin z20.h, p3/M, z20.h, z30.h\n"
    "st1h { z26.h }, p0, [x24]\n"
    "fmin z21.h, p3/M, z21.h, z30.h\n"
    "st1h { z25.h }, p0, [x24, x16, LSL #1]\n"
    "st1h { z24.h }, p0, [x24, x22, LSL #1]\n"
    "st1h { z22.h }, p0, [x23]\n"
    "st1h { z20.h }, p0, [x23, x16, LSL #1]\n"
    "st1h { z21.h }, p0, [x23, x22, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
