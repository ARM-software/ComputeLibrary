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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

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
    "mov x1, #0x0\n"
    "mov x2, #0x0\n"
    "1:"  // Tile loop
    "str x1, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x20, #0x4\n"
    "mov x25, #0x4\n"
    "str x2, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "cnth x3\n"
    "ldr x4, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "mov x6, #0x0\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x22, x1, x24\n"  // offset = tile_i * ld_input_row
    "mul x21, x1, x23\n"  // offset = tile_i * ld_output_row
    "ldr x17, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "cmp x3, %x[n_channels]\n"
    "ld1rh { z27.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x16, x4, x4\n"
    "add x15, x5, x5\n"
    "ld1rh { z29.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "madd x22, x2, x4, x22\n"  // offset += tile_j * ld_input_col
    "add x14, x16, x4\n"
    "ld1h { z13.h }, p3/Z, [x8]\n"
    "ld1h { z0.h }, p3/Z, [x8, #1, MUL VL]\n"
    "add x13, x15, x5\n"
    "madd x21, x2, x5, x21\n"  // offset += tile_j * ld_output_col
    "ld1h { z1.h }, p3/Z, [x8, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x8, #3, MUL VL]\n"
    "add x12, x14, x4\n"
    "mul x22, x22, x20\n"  // offset *= kernel_stride * output_size
    "ld1h { z3.h }, p3/Z, [x8, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x8, #5, MUL VL]\n"
    "add x11, x12, x4\n"
    "ld1h { z5.h }, p3/Z, [x8, #6, MUL VL]\n"
    "ld1h { z6.h }, p3/Z, [x8, #7, MUL VL]\n"
    "addvl x8, x8, #16\n"
    "sub x20, XZR, x3\n"
    "mul x21, x21, x25\n"  // offset *= output_tile_size
    "add x7, x7, x22, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x10, x7, x24, LSL #1\n"
    "add x9, x10, x24, LSL #1\n"
    "ld1h { z10.h }, p2/Z, [x7]\n"
    "ld1h { z11.h }, p2/Z, [x7, x11, LSL #1]\n"
    "add x28, x9, x24, LSL #1\n"
    "add x27, x28, x24, LSL #1\n"
    "ld1h { z7.h }, p3/Z, [x8, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x8, #-7, MUL VL]\n"
    "addvl x8, x8, #-6\n"
    "add x17, x17, x21, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "add x26, x27, x24, LSL #1\n"
    "ld1h { z9.h }, p2/Z, [x9, x16, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x9, x14, LSL #1]\n"
    "add x25, x17, x23, LSL #1\n"
    "add x24, x25, x23, LSL #1\n"
    "add x23, x24, x23, LSL #1\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z14, z13\n fmla z14.h, p3/M, z4.h, z9.h\n"
    "movprfx z19, z13\n fmla z19.h, p3/M, z8.h, z9.h\n"
    "whilelt p1.h, x3, %x[n_channels]\n"
    "inch x6\n"
    "movprfx z18, z13\n fmla z18.h, p3/M, z3.h, z9.h\n"
    "movprfx z26, z13\n fmla z26.h, p3/M, z1.h, z9.h\n"
    "inch x3\n"
    "mov p0.b, p2.b\n"
    "movprfx z15, z13\n fmla z15.h, p3/M, z0.h, z9.h\n"
    "movprfx z30, z13\n fmla z30.h, p3/M, z7.h, z9.h\n"
    "inch x20\n"
    "movprfx z28, z13\n fmla z28.h, p3/M, z6.h, z9.h\n"
    "movprfx z21, z13\n fmla z21.h, p3/M, z5.h, z9.h\n"
    "fmla z14.h, p3/M, z5.h, z12.h\n"
    "movprfx z24, z13\n fmla z24.h, p3/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x28, x16, LSL #1]\n"
    "fmla z19.h, p3/M, z0.h, z10.h\n"
    "movprfx z22, z13\n fmla z22.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x26]\n"
    "ld1h { z10.h }, p2/Z, [x26, x11, LSL #1]\n"
    "fmla z18.h, p3/M, z4.h, z12.h\n"
    "fmla z26.h, p3/M, z2.h, z12.h\n"
    "fmla z15.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z8.h, z12.h\n"
    "movprfx z25, z13\n fmla z25.h, p3/M, z6.h, z11.h\n"
    "fmla z14.h, p3/M, z7.h, z9.h\n"
    "ld1h { z11.h }, p2/Z, [x28, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmla z22.h, p3/M, z6.h, z12.h\n"
    "movprfx z31, z13\n fmla z31.h, p3/M, z3.h, z12.h\n"
    "movprfx z17, z13\n fmla z17.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x7, x4, LSL #1]\n"
    "movprfx z20, z13\n fmla z20.h, p3/M, z8.h, z10.h\n"
    "fmla z18.h, p3/M, z6.h, z9.h\n"
    "ld1h { z10.h }, p2/Z, [x7, x12, LSL #1]\n"
    "fmla z26.h, p3/M, z4.h, z9.h\n"
    "fmla z15.h, p3/M, z3.h, z9.h\n"
    "movprfx z16, z13\n fmla z16.h, p3/M, z1.h, z9.h\n"
    "movprfx z23, z13\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "ld1h { z13.h }, p3/Z, [x8]\n"
    "fmla z21.h, p3/M, z8.h, z9.h\n"
    "fmla z24.h, p3/M, z5.h, z9.h\n"
    "fmla z25.h, p3/M, z2.h, z9.h\n"
    "fmla z14.h, p3/M, z8.h, z11.h\n"
    "ld1h { z9.h }, p2/Z, [x10]\n"
    "fmla z19.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10, x11, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z10.h\n"
    "fmla z22.h, p3/M, z1.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27]\n"
    "fmla z18.h, p3/M, z7.h, z11.h\n"
    "fmla z31.h, p3/M, z6.h, z11.h\n"
    "fmla z26.h, p3/M, z5.h, z11.h\n"
    "fmla z15.h, p3/M, z4.h, z11.h\n"
    "fmla z17.h, p3/M, z3.h, z11.h\n"
    "fmla z16.h, p3/M, z2.h, z11.h\n"
    "fmla z23.h, p3/M, z1.h, z11.h\n"
    "fmla z20.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x10, x16, LSL #1]\n"
    "fmla z21.h, p3/M, z0.h, z9.h\n"
    "fmla z24.h, p3/M, z6.h, z10.h\n"
    "fmla z25.h, p3/M, z3.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27, x11, LSL #1]\n"
    "fmla z19.h, p3/M, z3.h, z9.h\n"
    "fmla z14.h, p3/M, z1.h, z11.h\n"
    "fmla z22.h, p3/M, z5.h, z12.h\n"
    "fmla z31.h, p3/M, z2.h, z12.h\n"
    "fmla z30.h, p3/M, z4.h, z11.h\n"
    "ld1h { z12.h }, p2/Z, [x10, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z11.h\n"
    "fmla z18.h, p3/M, z0.h, z11.h\n"
    "fmla z17.h, p3/M, z8.h, z10.h\n"
    "fmla z20.h, p3/M, z5.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x26, x4, LSL #1]\n"
    "fmla z21.h, p3/M, z2.h, z11.h\n"
    "fmla z14.h, p3/M, z2.h, z12.h\n"
    "fmla z19.h, p3/M, z5.h, z11.h\n"
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "ld1h { z11.h }, p2/Z, [x9, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z4.h, z12.h\n"
    "fmla z22.h, p3/M, z3.h, z12.h\n"
    "fmla z18.h, p3/M, z1.h, z12.h\n"
    "fmla z31.h, p3/M, z0.h, z12.h\n"
    "ld1h { z9.h }, p2/Z, [x9, x12, LSL #1]\n"
    "fmla z25.h, p3/M, z7.h, z10.h\n"
    "fmla z16.h, p3/M, z6.h, z10.h\n"
    "ld1h { z12.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z21.h, p3/M, z4.h, z11.h\n"
    "fmla z14.h, p3/M, z3.h, z11.h\n"
    "fmla z24.h, p3/M, z1.h, z11.h\n"
    "fmla z26.h, p3/M, z0.h, z11.h\n"
    "fmla z19.h, p3/M, z7.h, z11.h\n"
    "fmla z30.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x7, x16, LSL #1]\n"
    "fmla z23.h, p3/M, z8.h, z12.h\n"
    "fmla z20.h, p3/M, z7.h, z12.h\n"
    "ld1h { z10.h }, p2/Z, [x28, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z8.h, z9.h\n"
    "fmla z22.h, p3/M, z7.h, z9.h\n"
    "fmla z18.h, p3/M, z5.h, z9.h\n"
    "fmla z31.h, p3/M, z4.h, z9.h\n"
    "fmla z15.h, p3/M, z2.h, z9.h\n"
    "fmla z17.h, p3/M, z1.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x7, x14, LSL #1]\n"
    "addvl x7, x7, #1\n"
    "fmla z21.h, p3/M, z7.h, z10.h\n"
    "fmla z14.h, p3/M, z6.h, z10.h\n"
    "fmla z24.h, p3/M, z4.h, z10.h\n"
    "fmla z26.h, p3/M, z3.h, z10.h\n"
    "fmla z25.h, p3/M, z1.h, z10.h\n"
    "fmla z16.h, p3/M, z0.h, z10.h\n"
    "ld1h { z12.h }, p2/Z, [x28, x12, LSL #1]\n"
    "fmla z19.h, p3/M, z2.h, z11.h\n"
    "fmla z30.h, p3/M, z1.h, z11.h\n"
    "fmla z28.h, p3/M, z0.h, z11.h\n"
    "ld1h { z10.h }, p2/Z, [x9]\n"
    "fmla z22.h, p3/M, z0.h, z9.h\n"
    "fmla z23.h, p3/M, z2.h, z12.h\n"
    "fmla z18.h, p3/M, z8.h, z12.h\n"
    "fmla z31.h, p3/M, z7.h, z12.h\n"
    "fmla z15.h, p3/M, z5.h, z12.h\n"
    "fmla z21.h, p3/M, z3.h, z10.h\n"
    "fmla z24.h, p3/M, z0.h, z10.h\n"
    "fmla z17.h, p3/M, z4.h, z12.h\n"
    "fmla z20.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x16, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z9.h\n"
    "fmla z28.h, p3/M, z1.h, z9.h\n"
    "ld1h { z11.h }, p2/Z, [x9, x11, LSL #1]\n"
    "addvl x9, x9, #1\n"
    "fmla z19.h, p3/M, z6.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x28]\n"
    "fmla z16.h, p3/M, z4.h, z12.h\n"
    "fmla z23.h, p3/M, z3.h, z12.h\n"
    "fmla z26.h, p3/M, z7.h, z12.h\n"
    "fmla z22.h, p3/M, z8.h, z11.h\n"
    "fmla z31.h, p3/M, z5.h, z11.h\n"
    "ld1h { z9.h }, p1/Z, [x9, x16, LSL #1]\n"
    "fmla z17.h, p3/M, z2.h, z11.h\n"
    "fmla z21.h, p3/M, z6.h, z10.h\n"
    "ld1h { z11.h }, p2/Z, [x28, x11, LSL #1]\n"
    "addvl x28, x28, #1\n"
    "fmla z24.h, p3/M, z3.h, z10.h\n"
    "fmla z25.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x26, x16, LSL #1]\n"
    "fmla z15.h, p3/M, z6.h, z12.h\n"
    "fmla z20.h, p3/M, z2.h, z11.h\n"
    "fmla z31.h, p3/M, z8.h, z11.h\n"
    "fmla z16.h, p3/M, z7.h, z10.h\n"
    "fmla z23.h, p3/M, z6.h, z10.h\n"
    "fmla z17.h, p3/M, z5.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x26, x14, LSL #1]\n"
    "addvl x26, x26, #1\n"
    "fmla z24.h, p3/M, z8.h, z12.h\n"
    "fmla z25.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x14, LSL #1]\n"
    "fmla z16.h, p3/M, z5.h, z12.h\n"
    "fmla z23.h, p3/M, z4.h, z12.h\n"
    "fmla z20.h, p3/M, z3.h, z12.h\n"
    "fmla z26.h, p3/M, z8.h, z12.h\n"
    "fmla z15.h, p3/M, z7.h, z12.h\n"
    "fmla z17.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10, x12, LSL #1]\n"
    "fmla z25.h, p3/M, z8.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x10, x4, LSL #1]\n"
    "addvl x10, x10, #1\n"
    "fmla z16.h, p3/M, z8.h, z11.h\n"
    "fmla z23.h, p3/M, z7.h, z11.h\n"
    "fmla z20.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x27, x4, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z12.h\n"
    "fmla z22.h, p3/M, z4.h, z12.h\n"
    "fmla z19.h, p3/M, z4.h, z10.h\n"
    "fmla z30.h, p3/M, z3.h, z10.h\n"
    "fmla z21.h, p3/M, z1.h, z10.h\n"
    "fmla z14.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27, x12, LSL #1]\n"
    "ld1h { z0.h }, p3/Z, [x8, #1, MUL VL]\n"
    "fmla z18.h, p3/M, z2.h, z12.h\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "ld1h { z1.h }, p3/Z, [x8, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x8, #3, MUL VL]\n"
    "fmla z24.h, p3/M, z7.h, z11.h\n"
    "fmla z26.h, p3/M, z6.h, z11.h\n"
    "fmax z28.h, p3/M, z28.h, z27.h\n"
    "fmax z22.h, p3/M, z22.h, z27.h\n"
    "fmla z25.h, p3/M, z4.h, z11.h\n"
    "fmla z16.h, p3/M, z3.h, z11.h\n"
    "fmax z19.h, p3/M, z19.h, z27.h\n"
    "fmax z30.h, p3/M, z30.h, z27.h\n"
    "fmla z15.h, p3/M, z8.h, z10.h\n"
    "fmla z17.h, p3/M, z7.h, z10.h\n"
    "fmax z21.h, p3/M, z21.h, z27.h\n"
    "fmax z14.h, p3/M, z14.h, z27.h\n"
    "fmla z23.h, p3/M, z5.h, z10.h\n"
    "fmla z20.h, p3/M, z4.h, z10.h\n"
    "fmax z18.h, p3/M, z18.h, z27.h\n"
    "fmax z31.h, p3/M, z31.h, z27.h\n"
    "fmax z24.h, p3/M, z24.h, z27.h\n"
    "fmax z26.h, p3/M, z26.h, z27.h\n"
    "ld1h { z3.h }, p3/Z, [x8, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x8, #5, MUL VL]\n"
    "fmax z25.h, p3/M, z25.h, z27.h\n"
    "fmax z16.h, p3/M, z16.h, z27.h\n"
    "ld1h { z5.h }, p3/Z, [x8, #6, MUL VL]\n"
    "ld1h { z6.h }, p3/Z, [x8, #7, MUL VL]\n"
    "fmax z15.h, p3/M, z15.h, z27.h\n"
    "fmax z17.h, p3/M, z17.h, z27.h\n"
    "ld1h { z10.h }, p1/Z, [x7]\n"
    "ld1h { z11.h }, p1/Z, [x7, x11, LSL #1]\n"
    "fmax z23.h, p3/M, z23.h, z27.h\n"
    "fmax z20.h, p3/M, z20.h, z27.h\n"
    "ld1h { z12.h }, p1/Z, [x9, x14, LSL #1]\n"
    "addvl x8, x8, #16\n"
    "whilelt p2.h, x6, %x[n_channels]\n"
    "cmp x3, %x[n_channels]\n"
    "fmin z19.h, p3/M, z19.h, z29.h\n"
    "fmin z30.h, p3/M, z30.h, z29.h\n"
    "fmin z28.h, p3/M, z28.h, z29.h\n"
    "fmin z22.h, p3/M, z22.h, z29.h\n"
    "fmin z21.h, p3/M, z21.h, z29.h\n"
    "ld1h { z7.h }, p3/Z, [x8, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x8, #-7, MUL VL]\n"
    "fmin z14.h, p3/M, z14.h, z29.h\n"
    "fmin z18.h, p3/M, z18.h, z29.h\n"
    "st1h { z19.h }, p0, [x17]\n"
    "fmin z31.h, p3/M, z31.h, z29.h\n"
    "fmin z24.h, p3/M, z24.h, z29.h\n"
    "st1h { z30.h }, p0, [x17, x5, LSL #1]\n"
    "fmin z26.h, p3/M, z26.h, z29.h\n"
    "fmin z15.h, p3/M, z15.h, z29.h\n"
    "st1h { z28.h }, p0, [x17, x15, LSL #1]\n"
    "fmin z17.h, p3/M, z17.h, z29.h\n"
    "fmin z25.h, p3/M, z25.h, z29.h\n"
    "st1h { z22.h }, p0, [x17, x13, LSL #1]\n"
    "fmin z16.h, p3/M, z16.h, z29.h\n"
    "fmin z23.h, p3/M, z23.h, z29.h\n"
    "st1h { z21.h }, p0, [x25]\n"
    "fmin z20.h, p3/M, z20.h, z29.h\n"
    "addvl x27, x27, #1\n"
    "st1h { z14.h }, p0, [x25, x5, LSL #1]\n"
    "st1h { z18.h }, p0, [x25, x15, LSL #1]\n"
    "addvl x17, x17, #1\n"
    "addvl x8, x8, #-6\n"
    "st1h { z31.h }, p0, [x25, x13, LSL #1]\n"
    "addvl x25, x25, #1\n"
    "st1h { z24.h }, p0, [x24]\n"
    "st1h { z26.h }, p0, [x24, x5, LSL #1]\n"
    "st1h { z15.h }, p0, [x24, x15, LSL #1]\n"
    "st1h { z17.h }, p0, [x24, x13, LSL #1]\n"
    "addvl x24, x24, #1\n"
    "st1h { z25.h }, p0, [x23]\n"
    "st1h { z16.h }, p0, [x23, x5, LSL #1]\n"
    "st1h { z23.h }, p0, [x23, x15, LSL #1]\n"
    "st1h { z20.h }, p0, [x23, x13, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z14, z13\n fmla z14.h, p3/M, z4.h, z9.h\n"
    "movprfx z18, z13\n fmla z18.h, p3/M, z8.h, z9.h\n"
    "ldr x2, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x1, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z23, z13\n fmla z23.h, p3/M, z3.h, z9.h\n"
    "movprfx z30, z13\n fmla z30.h, p3/M, z1.h, z9.h\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "movprfx z20, z13\n fmla z20.h, p3/M, z0.h, z9.h\n"
    "movprfx z25, z13\n fmla z25.h, p3/M, z7.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "movprfx z19, z13\n fmla z19.h, p3/M, z6.h, z9.h\n"
    "movprfx z26, z13\n fmla z26.h, p3/M, z5.h, z9.h\n"
    "add x2, x2, #0x1\n"
    "add x20, x1, #0x1\n"
    "fmla z14.h, p3/M, z5.h, z12.h\n"
    "movprfx z28, z13\n fmla z28.h, p3/M, z2.h, z9.h\n"
    "ld1h { z15.h }, p2/Z, [x28, x16, LSL #1]\n"
    "cmp x2, x22\n"
    "fmla z18.h, p3/M, z0.h, z10.h\n"
    "movprfx z9, z13\n fmla z9.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x26]\n"
    "ld1h { z24.h }, p2/Z, [x26, x11, LSL #1]\n"
    "fmla z23.h, p3/M, z4.h, z12.h\n"
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "csel x1, x1, x20, LT\n"
    "csel x2, x2, XZR, LT\n"
    "fmla z20.h, p3/M, z1.h, z12.h\n"
    "fmla z25.h, p3/M, z8.h, z12.h\n"
    "movprfx z22, z13\n fmla z22.h, p3/M, z6.h, z17.h\n"
    "fmla z14.h, p3/M, z7.h, z15.h\n"
    "ld1h { z10.h }, p2/Z, [x28, x14, LSL #1]\n"
    "fmla z19.h, p3/M, z7.h, z12.h\n"
    "fmla z9.h, p3/M, z6.h, z12.h\n"
    "cmp x1, x21\n"
    "movprfx z31, z13\n fmla z31.h, p3/M, z3.h, z12.h\n"
    "movprfx z11, z13\n fmla z11.h, p3/M, z0.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x7, x4, LSL #1]\n"
    "movprfx z12, z13\n fmla z12.h, p3/M, z8.h, z24.h\n"
    "fmla z23.h, p3/M, z6.h, z15.h\n"
    "ld1h { z17.h }, p2/Z, [x7, x12, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z15.h\n"
    "fmla z20.h, p3/M, z3.h, z15.h\n"
    "movprfx z24, z13\n fmla z24.h, p3/M, z1.h, z15.h\n"
    "fmla z13.h, p3/M, z0.h, z15.h\n"
    "fmla z26.h, p3/M, z8.h, z15.h\n"
    "fmla z28.h, p3/M, z5.h, z15.h\n"
    "fmla z22.h, p3/M, z2.h, z15.h\n"
    "fmla z14.h, p3/M, z8.h, z10.h\n"
    "ld1h { z15.h }, p2/Z, [x10]\n"
    "fmla z18.h, p3/M, z1.h, z16.h\n"
    "fmla z25.h, p3/M, z0.h, z16.h\n"
    "ld1h { z21.h }, p2/Z, [x10, x11, LSL #1]\n"
    "fmla z19.h, p3/M, z2.h, z17.h\n"
    "fmla z9.h, p3/M, z1.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x27]\n"
    "fmla z23.h, p3/M, z7.h, z10.h\n"
    "fmla z31.h, p3/M, z6.h, z10.h\n"
    "fmla z30.h, p3/M, z5.h, z10.h\n"
    "fmla z20.h, p3/M, z4.h, z10.h\n"
    "fmla z11.h, p3/M, z3.h, z10.h\n"
    "fmla z24.h, p3/M, z2.h, z10.h\n"
    "fmla z13.h, p3/M, z1.h, z10.h\n"
    "fmla z12.h, p3/M, z0.h, z10.h\n"
    "ld1h { z17.h }, p2/Z, [x10, x16, LSL #1]\n"
    "fmla z26.h, p3/M, z0.h, z15.h\n"
    "fmla z28.h, p3/M, z6.h, z16.h\n"
    "fmla z22.h, p3/M, z3.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x27, x11, LSL #1]\n"
    "fmla z18.h, p3/M, z3.h, z15.h\n"
    "fmla z14.h, p3/M, z1.h, z17.h\n"
    "fmla z9.h, p3/M, z5.h, z21.h\n"
    "fmla z31.h, p3/M, z2.h, z21.h\n"
    "fmla z25.h, p3/M, z4.h, z17.h\n"
    "ld1h { z21.h }, p2/Z, [x10, x14, LSL #1]\n"
    "fmla z19.h, p3/M, z3.h, z17.h\n"
    "fmla z23.h, p3/M, z0.h, z17.h\n"
    "fmla z11.h, p3/M, z8.h, z16.h\n"
    "fmla z12.h, p3/M, z5.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x4, LSL #1]\n"
    "fmla z26.h, p3/M, z2.h, z17.h\n"
    "fmla z14.h, p3/M, z2.h, z21.h\n"
    "fmla z18.h, p3/M, z5.h, z17.h\n"
    "fmla z25.h, p3/M, z5.h, z21.h\n"
    "ld1h { z17.h }, p2/Z, [x9, x4, LSL #1]\n"
    "fmla z19.h, p3/M, z4.h, z21.h\n"
    "fmla z9.h, p3/M, z3.h, z21.h\n"
    "fmla z23.h, p3/M, z1.h, z21.h\n"
    "fmla z31.h, p3/M, z0.h, z21.h\n"
    "ld1h { z21.h }, p2/Z, [x9, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z7.h, z16.h\n"
    "fmla z24.h, p3/M, z6.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z26.h, p3/M, z4.h, z17.h\n"
    "fmla z14.h, p3/M, z3.h, z17.h\n"
    "fmla z28.h, p3/M, z1.h, z17.h\n"
    "fmla z30.h, p3/M, z0.h, z17.h\n"
    "fmla z18.h, p3/M, z7.h, z17.h\n"
    "fmla z25.h, p3/M, z6.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x7, x16, LSL #1]\n"
    "fmla z13.h, p3/M, z8.h, z16.h\n"
    "fmla z12.h, p3/M, z7.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x4, LSL #1]\n"
    "fmla z19.h, p3/M, z8.h, z21.h\n"
    "fmla z9.h, p3/M, z7.h, z21.h\n"
    "fmla z23.h, p3/M, z5.h, z21.h\n"
    "fmla z31.h, p3/M, z4.h, z21.h\n"
    "fmla z20.h, p3/M, z2.h, z21.h\n"
    "fmla z11.h, p3/M, z1.h, z21.h\n"
    "ld1h { z21.h }, p2/Z, [x7, x14, LSL #1]\n"
    "fmla z26.h, p3/M, z7.h, z16.h\n"
    "fmla z14.h, p3/M, z6.h, z16.h\n"
    "fmla z28.h, p3/M, z4.h, z16.h\n"
    "fmla z30.h, p3/M, z3.h, z16.h\n"
    "fmla z22.h, p3/M, z1.h, z16.h\n"
    "fmla z24.h, p3/M, z0.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x12, LSL #1]\n"
    "fmla z18.h, p3/M, z2.h, z17.h\n"
    "fmla z25.h, p3/M, z1.h, z17.h\n"
    "fmla z19.h, p3/M, z0.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x9]\n"
    "fmla z9.h, p3/M, z0.h, z21.h\n"
    "fmla z13.h, p3/M, z2.h, z16.h\n"
    "fmla z23.h, p3/M, z8.h, z16.h\n"
    "fmla z31.h, p3/M, z7.h, z16.h\n"
    "fmla z20.h, p3/M, z5.h, z16.h\n"
    "fmla z26.h, p3/M, z3.h, z17.h\n"
    "fmla z28.h, p3/M, z0.h, z17.h\n"
    "fmla z11.h, p3/M, z4.h, z16.h\n"
    "fmla z12.h, p3/M, z1.h, z16.h\n"
    "ld1h { z15.h }, p2/Z, [x27, x16, LSL #1]\n"
    "fmla z25.h, p3/M, z2.h, z21.h\n"
    "fmla z19.h, p3/M, z1.h, z21.h\n"
    "ld1h { z16.h }, p2/Z, [x9, x11, LSL #1]\n"
    "fmla z18.h, p3/M, z6.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x28]\n"
    "fmla z24.h, p3/M, z4.h, z15.h\n"
    "fmla z13.h, p3/M, z3.h, z15.h\n"
    "fmla z30.h, p3/M, z7.h, z15.h\n"
    "fmla z9.h, p3/M, z8.h, z16.h\n"
    "fmla z31.h, p3/M, z5.h, z16.h\n"
    "fmla z11.h, p3/M, z2.h, z16.h\n"
    "fmla z26.h, p3/M, z6.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x11, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z17.h\n"
    "fmla z22.h, p3/M, z0.h, z17.h\n"
    "ld1h { z21.h }, p2/Z, [x26, x16, LSL #1]\n"
    "fmla z20.h, p3/M, z6.h, z15.h\n"
    "fmla z12.h, p3/M, z2.h, z16.h\n"
    "fmla z31.h, p3/M, z8.h, z16.h\n"
    "fmla z24.h, p3/M, z7.h, z21.h\n"
    "fmla z13.h, p3/M, z6.h, z21.h\n"
    "fmla z11.h, p3/M, z5.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z8.h, z15.h\n"
    "fmla z22.h, p3/M, z5.h, z15.h\n"
    "ld1h { z16.h }, p2/Z, [x27, x14, LSL #1]\n"
    "fmla z24.h, p3/M, z5.h, z16.h\n"
    "fmla z13.h, p3/M, z4.h, z16.h\n"
    "fmla z12.h, p3/M, z3.h, z16.h\n"
    "fmla z30.h, p3/M, z8.h, z16.h\n"
    "fmla z20.h, p3/M, z7.h, z16.h\n"
    "fmla z11.h, p3/M, z6.h, z16.h\n"
    "ld1h { z15.h }, p2/Z, [x10, x12, LSL #1]\n"
    "fmla z22.h, p3/M, z8.h, z21.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x4, LSL #1]\n"
    "fmla z24.h, p3/M, z8.h, z17.h\n"
    "fmla z13.h, p3/M, z7.h, z17.h\n"
    "fmla z12.h, p3/M, z6.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x4, LSL #1]\n"
    "fmla z19.h, p3/M, z5.h, z15.h\n"
    "fmla z9.h, p3/M, z4.h, z15.h\n"
    "fmla z18.h, p3/M, z4.h, z16.h\n"
    "fmla z25.h, p3/M, z3.h, z16.h\n"
    "fmla z26.h, p3/M, z1.h, z16.h\n"
    "fmla z14.h, p3/M, z0.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x27, x12, LSL #1]\n"
    "fmla z23.h, p3/M, z2.h, z15.h\n"
    "fmla z31.h, p3/M, z1.h, z15.h\n"
    "fmla z28.h, p3/M, z7.h, z17.h\n"
    "fmla z30.h, p3/M, z6.h, z17.h\n"
    "fmax z19.h, p3/M, z19.h, z27.h\n"
    "fmax z9.h, p3/M, z9.h, z27.h\n"
    "fmla z22.h, p3/M, z4.h, z17.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "fmax z18.h, p3/M, z18.h, z27.h\n"
    "fmax z25.h, p3/M, z25.h, z27.h\n"
    "fmla z20.h, p3/M, z8.h, z16.h\n"
    "fmla z11.h, p3/M, z7.h, z16.h\n"
    "fmax z26.h, p3/M, z26.h, z27.h\n"
    "fmax z14.h, p3/M, z14.h, z27.h\n"
    "fmla z13.h, p3/M, z5.h, z16.h\n"
    "fmla z12.h, p3/M, z4.h, z16.h\n"
    "fmax z23.h, p3/M, z23.h, z27.h\n"
    "fmax z31.h, p3/M, z31.h, z27.h\n"
    "fmax z28.h, p3/M, z28.h, z27.h\n"
    "fmax z30.h, p3/M, z30.h, z27.h\n"
    "fmax z22.h, p3/M, z22.h, z27.h\n"
    "fmax z24.h, p3/M, z24.h, z27.h\n"
    "fmax z20.h, p3/M, z20.h, z27.h\n"
    "fmax z11.h, p3/M, z11.h, z27.h\n"
    "fmax z13.h, p3/M, z13.h, z27.h\n"
    "fmax z12.h, p3/M, z12.h, z27.h\n"
    "fmin z18.h, p3/M, z18.h, z29.h\n"
    "fmin z25.h, p3/M, z25.h, z29.h\n"
    "fmin z19.h, p3/M, z19.h, z29.h\n"
    "fmin z9.h, p3/M, z9.h, z29.h\n"
    "fmin z26.h, p3/M, z26.h, z29.h\n"
    "fmin z14.h, p3/M, z14.h, z29.h\n"
    "fmin z23.h, p3/M, z23.h, z29.h\n"
    "fmin z31.h, p3/M, z31.h, z29.h\n"
    "st1h { z18.h }, p0, [x17]\n"
    "fmin z28.h, p3/M, z28.h, z29.h\n"
    "fmin z30.h, p3/M, z30.h, z29.h\n"
    "st1h { z25.h }, p0, [x17, x5, LSL #1]\n"
    "fmin z20.h, p3/M, z20.h, z29.h\n"
    "fmin z11.h, p3/M, z11.h, z29.h\n"
    "st1h { z19.h }, p0, [x17, x15, LSL #1]\n"
    "fmin z22.h, p3/M, z22.h, z29.h\n"
    "fmin z24.h, p3/M, z24.h, z29.h\n"
    "st1h { z9.h }, p0, [x17, x13, LSL #1]\n"
    "fmin z13.h, p3/M, z13.h, z29.h\n"
    "fmin z12.h, p3/M, z12.h, z29.h\n"
    "st1h { z26.h }, p0, [x25]\n"
    "st1h { z14.h }, p0, [x25, x5, LSL #1]\n"
    "st1h { z23.h }, p0, [x25, x15, LSL #1]\n"
    "st1h { z31.h }, p0, [x25, x13, LSL #1]\n"
    "st1h { z28.h }, p0, [x24]\n"
    "st1h { z30.h }, p0, [x24, x5, LSL #1]\n"
    "st1h { z20.h }, p0, [x24, x15, LSL #1]\n"
    "st1h { z11.h }, p0, [x24, x13, LSL #1]\n"
    "st1h { z22.h }, p0, [x23]\n"
    "st1h { z24.h }, p0, [x23, x5, LSL #1]\n"
    "st1h { z13.h }, p0, [x23, x15, LSL #1]\n"
    "st1h { z12.h }, p0, [x23, x13, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
