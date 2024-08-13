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
    "mov x5, #0x0\n"
    "mov x6, #0x0\n"
    "1:"  // Tile loop
    "str x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x3\n"
    "mov x25, #0x3\n"
    "str x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cnth x8\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "mov x16, #0x0\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x22, x5, x24\n"  // offset = tile_i * ld_input_row
    "ldr x13, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x12, x7, x7\n"
    "cmp x8, %x[n_channels]\n"
    "ld1rh { z15.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mul x21, x5, x23\n"  // offset = tile_i * ld_output_row
    "add x11, x12, x7\n"
    "add x10, x17, x17\n"
    "ld1rh { z14.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "madd x22, x6, x7, x22\n"  // offset += tile_j * ld_input_col
    "ld1h { z31.h }, p3/Z, [x14]\n"
    "ld1h { z0.h }, p3/Z, [x14, #1, MUL VL]\n"
    "add x9, x11, x7\n"
    "ld1h { z1.h }, p3/Z, [x14, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x14, #3, MUL VL]\n"
    "sub x20, XZR, x8\n"
    "madd x21, x6, x17, x21\n"  // offset += tile_j * ld_output_col
    "ld1h { z3.h }, p3/Z, [x14, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x14, #5, MUL VL]\n"
    "mul x22, x22, x26\n"  // offset *= kernel_stride * output_size
    "ld1h { z5.h }, p3/Z, [x14, #6, MUL VL]\n"
    "ld1h { z6.h }, p3/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "mul x21, x21, x25\n"  // offset *= output_tile_size
    "add x15, x15, x22, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x28, x15, x24, LSL #1\n"
    "add x27, x28, x24, LSL #1\n"
    "ld1h { z10.h }, p2/Z, [x15]\n"
    "ld1h { z11.h }, p2/Z, [x15, x9, LSL #1]\n"
    "add x26, x27, x24, LSL #1\n"
    "add x13, x13, x21, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "add x25, x26, x24, LSL #1\n"
    "ld1h { z7.h }, p3/Z, [x14, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x14, #-7, MUL VL]\n"
    "add x24, x13, x23, LSL #1\n"
    "ld1h { z9.h }, p2/Z, [x27, x12, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x25]\n"
    "addvl x14, x14, #-6\n"
    "add x23, x24, x23, LSL #1\n"
    "ld1h { z13.h }, p2/Z, [x28, x12, LSL #1]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z30, z31\n fmla z30.h, p3/M, z7.h, z9.h\n"
    "movprfx z29, z31\n fmla z29.h, p3/M, z8.h, z9.h\n"
    "whilelt p1.h, x8, %x[n_channels]\n"
    "inch x16\n"
    "movprfx z28, z31\n fmla z28.h, p3/M, z6.h, z9.h\n"
    "movprfx z27, z31\n fmla z27.h, p3/M, z5.h, z9.h\n"
    "inch x8\n"
    "mov p0.b, p2.b\n"
    "movprfx z26, z31\n fmla z26.h, p3/M, z4.h, z9.h\n"
    "movprfx z25, z31\n fmla z25.h, p3/M, z3.h, z9.h\n"
    "inch x20\n"
    "movprfx z24, z31\n fmla z24.h, p3/M, z2.h, z9.h\n"
    "movprfx z23, z31\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z29.h, p3/M, z0.h, z10.h\n"
    "ld1h { z22.h }, p2/Z, [x27, x11, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x7, LSL #1]\n"
    "fmla z27.h, p3/M, z2.h, z13.h\n"
    "fmla z26.h, p3/M, z1.h, z13.h\n"
    "fmla z25.h, p3/M, z0.h, z13.h\n"
    "fmla z24.h, p3/M, z6.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x9, LSL #1]\n"
    "movprfx z21, z31\n fmla z21.h, p3/M, z1.h, z9.h\n"
    "ld1h { z31.h }, p3/Z, [x14]\n"
    "fmla z30.h, p3/M, z6.h, z17.h\n"
    "fmla z29.h, p3/M, z5.h, z13.h\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "ld1h { z18.h }, p2/Z, [x15, x7, LSL #1]\n"
    "fmla z27.h, p3/M, z4.h, z17.h\n"
    "fmla z23.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x15, x11, LSL #1]\n"
    "fmla z26.h, p3/M, z3.h, z17.h\n"
    "fmla z21.h, p3/M, z0.h, z17.h\n"
    "fmla z24.h, p3/M, z1.h, z17.h\n"
    "fmla z30.h, p3/M, z0.h, z18.h\n"
    "fmla z29.h, p3/M, z7.h, z17.h\n"
    "ld1h { z20.h }, p2/Z, [x28]\n"
    "fmla z28.h, p3/M, z1.h, z16.h\n"
    "fmla z25.h, p3/M, z4.h, z22.h\n"
    "fmla z23.h, p3/M, z1.h, z22.h\n"
    "fmla z26.h, p3/M, z5.h, z22.h\n"
    "fmla z21.h, p3/M, z2.h, z22.h\n"
    "fmla z27.h, p3/M, z0.h, z20.h\n"
    "fmla z30.h, p3/M, z2.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26]\n"
    "fmla z29.h, p3/M, z1.h, z18.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x9, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z22.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "fmla z25.h, p3/M, z2.h, z16.h\n"
    "fmla z27.h, p3/M, z6.h, z17.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z22.h\n"
    "ld1h { z18.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z20.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x9, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x7, LSL #1]\n"
    "fmla z21.h, p3/M, z4.h, z18.h\n"
    "fmla z23.h, p3/M, z3.h, z18.h\n"
    "fmla z26.h, p3/M, z7.h, z18.h\n"
    "fmla z24.h, p3/M, z5.h, z18.h\n"
    "fmla z25.h, p3/M, z6.h, z18.h\n"
    "fmla z27.h, p3/M, z8.h, z18.h\n"
    "fmla z30.h, p3/M, z3.h, z19.h\n"
    "fmla z21.h, p3/M, z6.h, z16.h\n"
    "fmla z29.h, p3/M, z4.h, z19.h\n"
    "fmla z23.h, p3/M, z5.h, z17.h\n"
    "fmla z26.h, p3/M, z0.h, z19.h\n"
    "fmla z24.h, p3/M, z7.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x25, x11, LSL #1]\n"
    "fmla z25.h, p3/M, z8.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x11, LSL #1]\n"
    "fmla z27.h, p3/M, z1.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x7, LSL #1]\n"
    "addvl x28, x28, #1\n"
    "fmla z21.h, p3/M, z8.h, z18.h\n"
    "fmla z23.h, p3/M, z7.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x26, x11, LSL #1]\n"
    "addvl x26, x26, #1\n"
    "fmla z30.h, p3/M, z5.h, z16.h\n"
    "fmla z28.h, p3/M, z4.h, z16.h\n"
    "fmla z26.h, p3/M, z2.h, z16.h\n"
    "fmla z25.h, p3/M, z1.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x15, x12, LSL #1]\n"
    "fmla z24.h, p3/M, z4.h, z17.h\n"
    "addvl x15, x15, #1\n"
    "fmla z21.h, p3/M, z3.h, z17.h\n"
    "fmla z27.h, p3/M, z7.h, z17.h\n"
    "fmla z23.h, p3/M, z4.h, z19.h\n"
    "ld1h { z4.h }, p3/Z, [x14, #5, MUL VL]\n"
    "fmla z26.h, p3/M, z6.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x27]\n"
    "fmla z29.h, p3/M, z2.h, z16.h\n"
    "fmla z30.h, p3/M, z1.h, z16.h\n"
    "ld1h { z1.h }, p3/Z, [x14, #2, MUL VL]\n"
    "ld1h { z10.h }, p1/Z, [x15]\n"
    "fmla z28.h, p3/M, z0.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x9, LSL #1]\n"
    "fmla z25.h, p3/M, z7.h, z19.h\n"
    "addvl x27, x27, #1\n"
    "fmla z21.h, p3/M, z5.h, z19.h\n"
    "fmla z24.h, p3/M, z0.h, z18.h\n"
    "ld1h { z0.h }, p3/Z, [x14, #1, MUL VL]\n"
    "fmla z26.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x12, LSL #1]\n"
    "fmla z27.h, p3/M, z3.h, z18.h\n"
    "addvl x25, x25, #1\n"
    "fmla z23.h, p3/M, z2.h, z17.h\n"
    "fmla z29.h, p3/M, z6.h, z18.h\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "ld1h { z2.h }, p3/Z, [x14, #3, MUL VL]\n"
    "fmla z28.h, p3/M, z8.h, z17.h\n"
    "fmla z25.h, p3/M, z5.h, z17.h\n"
    "ld1h { z3.h }, p3/Z, [x14, #4, MUL VL]\n"
    "ld1h { z5.h }, p3/Z, [x14, #6, MUL VL]\n"
    "fmla z24.h, p3/M, z8.h, z16.h\n"
    "fmla z21.h, p3/M, z7.h, z16.h\n"
    "whilelt p2.h, x16, %x[n_channels]\n"
    "cmp x8, %x[n_channels]\n"
    "fmax z27.h, p3/M, z27.h, z15.h\n"
    "fmax z26.h, p3/M, z26.h, z15.h\n"
    "ld1h { z9.h }, p1/Z, [x27, x12, LSL #1]\n"
    "ld1h { z11.h }, p1/Z, [x15, x9, LSL #1]\n"
    "fmla z23.h, p3/M, z6.h, z16.h\n"
    "fmax z29.h, p3/M, z29.h, z15.h\n"
    "ld1h { z6.h }, p3/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "fmax z28.h, p3/M, z28.h, z15.h\n"
    "fmax z25.h, p3/M, z25.h, z15.h\n"
    "ld1h { z12.h }, p1/Z, [x25]\n"
    "ld1h { z13.h }, p1/Z, [x28, x12, LSL #1]\n"
    "fmax z24.h, p3/M, z24.h, z15.h\n"
    "fmax z21.h, p3/M, z21.h, z15.h\n"
    "fmin z29.h, p3/M, z29.h, z14.h\n"
    "fmin z30.h, p3/M, z30.h, z14.h\n"
    "ld1h { z7.h }, p3/Z, [x14, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x14, #-7, MUL VL]\n"
    "fmax z23.h, p3/M, z23.h, z15.h\n"
    "fmin z28.h, p3/M, z28.h, z14.h\n"
    "fmin z27.h, p3/M, z27.h, z14.h\n"
    "fmin z26.h, p3/M, z26.h, z14.h\n"
    "fmin z25.h, p3/M, z25.h, z14.h\n"
    "fmin z24.h, p3/M, z24.h, z14.h\n"
    "st1h { z29.h }, p0, [x13]\n"
    "fmin z21.h, p3/M, z21.h, z14.h\n"
    "fmin z23.h, p3/M, z23.h, z14.h\n"
    "st1h { z30.h }, p0, [x13, x17, LSL #1]\n"
    "st1h { z28.h }, p0, [x13, x10, LSL #1]\n"
    "addvl x13, x13, #1\n"
    "addvl x14, x14, #-6\n"
    "st1h { z27.h }, p0, [x24]\n"
    "st1h { z26.h }, p0, [x24, x17, LSL #1]\n"
    "st1h { z25.h }, p0, [x24, x10, LSL #1]\n"
    "addvl x24, x24, #1\n"
    "st1h { z24.h }, p0, [x23]\n"
    "st1h { z21.h }, p0, [x23, x17, LSL #1]\n"
    "st1h { z23.h }, p0, [x23, x10, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z30, z31\n fmla z30.h, p3/M, z7.h, z9.h\n"
    "movprfx z29, z31\n fmla z29.h, p3/M, z8.h, z9.h\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z28, z31\n fmla z28.h, p3/M, z6.h, z9.h\n"
    "movprfx z27, z31\n fmla z27.h, p3/M, z5.h, z9.h\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "movprfx z26, z31\n fmla z26.h, p3/M, z4.h, z9.h\n"
    "movprfx z25, z31\n fmla z25.h, p3/M, z3.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "movprfx z24, z31\n fmla z24.h, p3/M, z2.h, z9.h\n"
    "movprfx z23, z31\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "add x6, x6, #0x1\n"
    "add x20, x5, #0x1\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z29.h, p3/M, z0.h, z10.h\n"
    "ld1h { z22.h }, p2/Z, [x27, x11, LSL #1]\n"
    "cmp x6, x22\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x7, LSL #1]\n"
    "fmla z27.h, p3/M, z2.h, z13.h\n"
    "csel x5, x5, x20, LT\n"
    "fmla z26.h, p3/M, z1.h, z13.h\n"
    "fmla z25.h, p3/M, z0.h, z13.h\n"
    "csel x6, x6, XZR, LT\n"
    "fmla z24.h, p3/M, z6.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x9, LSL #1]\n"
    "movprfx z21, z31\n fmla z21.h, p3/M, z1.h, z9.h\n"
    "fmla z30.h, p3/M, z6.h, z17.h\n"
    "fmla z29.h, p3/M, z5.h, z13.h\n"
    "cmp x5, x21\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "ld1h { z18.h }, p2/Z, [x15, x7, LSL #1]\n"
    "fmla z27.h, p3/M, z4.h, z17.h\n"
    "fmla z23.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x15, x11, LSL #1]\n"
    "fmla z26.h, p3/M, z3.h, z17.h\n"
    "fmla z21.h, p3/M, z0.h, z17.h\n"
    "fmla z24.h, p3/M, z1.h, z17.h\n"
    "fmla z30.h, p3/M, z0.h, z18.h\n"
    "fmla z29.h, p3/M, z7.h, z17.h\n"
    "ld1h { z20.h }, p2/Z, [x28]\n"
    "fmla z28.h, p3/M, z1.h, z16.h\n"
    "fmla z25.h, p3/M, z4.h, z22.h\n"
    "fmla z23.h, p3/M, z1.h, z22.h\n"
    "fmla z26.h, p3/M, z5.h, z22.h\n"
    "fmla z21.h, p3/M, z2.h, z22.h\n"
    "fmla z27.h, p3/M, z0.h, z20.h\n"
    "fmla z30.h, p3/M, z2.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26]\n"
    "fmla z29.h, p3/M, z1.h, z18.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x9, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z22.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "fmla z25.h, p3/M, z2.h, z16.h\n"
    "fmla z27.h, p3/M, z6.h, z17.h\n"
    "ld1h { z19.h }, p2/Z, [x28, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z22.h\n"
    "ld1h { z18.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z20.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x9, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x7, LSL #1]\n"
    "fmla z21.h, p3/M, z4.h, z18.h\n"
    "fmla z23.h, p3/M, z3.h, z18.h\n"
    "fmla z26.h, p3/M, z7.h, z18.h\n"
    "fmla z24.h, p3/M, z5.h, z18.h\n"
    "fmla z25.h, p3/M, z6.h, z18.h\n"
    "fmla z27.h, p3/M, z8.h, z18.h\n"
    "fmla z30.h, p3/M, z3.h, z19.h\n"
    "fmla z21.h, p3/M, z6.h, z16.h\n"
    "fmla z29.h, p3/M, z4.h, z19.h\n"
    "fmla z23.h, p3/M, z5.h, z17.h\n"
    "fmla z26.h, p3/M, z0.h, z19.h\n"
    "fmla z24.h, p3/M, z7.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x25, x11, LSL #1]\n"
    "fmla z25.h, p3/M, z8.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x28, x11, LSL #1]\n"
    "fmla z27.h, p3/M, z1.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x7, LSL #1]\n"
    "fmla z21.h, p3/M, z8.h, z18.h\n"
    "fmla z23.h, p3/M, z7.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x26, x11, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z16.h\n"
    "fmla z28.h, p3/M, z4.h, z16.h\n"
    "fmla z26.h, p3/M, z2.h, z16.h\n"
    "fmla z25.h, p3/M, z1.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x15, x12, LSL #1]\n"
    "fmla z24.h, p3/M, z4.h, z17.h\n"
    "fmla z21.h, p3/M, z3.h, z17.h\n"
    "fmla z27.h, p3/M, z7.h, z17.h\n"
    "fmla z23.h, p3/M, z4.h, z19.h\n"
    "fmla z26.h, p3/M, z6.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x27]\n"
    "fmla z29.h, p3/M, z2.h, z16.h\n"
    "fmla z30.h, p3/M, z1.h, z16.h\n"
    "fmla z28.h, p3/M, z0.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x9, LSL #1]\n"
    "fmla z25.h, p3/M, z7.h, z19.h\n"
    "fmla z21.h, p3/M, z5.h, z19.h\n"
    "fmla z24.h, p3/M, z0.h, z18.h\n"
    "fmla z26.h, p3/M, z8.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x25, x12, LSL #1]\n"
    "fmla z27.h, p3/M, z3.h, z18.h\n"
    "fmla z23.h, p3/M, z2.h, z17.h\n"
    "fmla z29.h, p3/M, z6.h, z18.h\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "fmla z28.h, p3/M, z8.h, z17.h\n"
    "fmla z25.h, p3/M, z5.h, z17.h\n"
    "fmla z24.h, p3/M, z8.h, z16.h\n"
    "fmla z21.h, p3/M, z7.h, z16.h\n"
    "fmax z27.h, p3/M, z27.h, z15.h\n"
    "fmax z26.h, p3/M, z26.h, z15.h\n"
    "fmin z30.h, p3/M, z30.h, z14.h\n"
    "fmla z23.h, p3/M, z6.h, z16.h\n"
    "fmax z29.h, p3/M, z29.h, z15.h\n"
    "fmax z28.h, p3/M, z28.h, z15.h\n"
    "fmax z25.h, p3/M, z25.h, z15.h\n"
    "fmin z27.h, p3/M, z27.h, z14.h\n"
    "fmin z26.h, p3/M, z26.h, z14.h\n"
    "fmax z24.h, p3/M, z24.h, z15.h\n"
    "fmax z21.h, p3/M, z21.h, z15.h\n"
    "fmax z23.h, p3/M, z23.h, z15.h\n"
    "fmin z29.h, p3/M, z29.h, z14.h\n"
    "fmin z28.h, p3/M, z28.h, z14.h\n"
    "fmin z25.h, p3/M, z25.h, z14.h\n"
    "st1h { z27.h }, p0, [x24]\n"
    "fmin z24.h, p3/M, z24.h, z14.h\n"
    "fmin z21.h, p3/M, z21.h, z14.h\n"
    "st1h { z26.h }, p0, [x24, x17, LSL #1]\n"
    "fmin z23.h, p3/M, z23.h, z14.h\n"
    "st1h { z29.h }, p0, [x13]\n"
    "st1h { z30.h }, p0, [x13, x17, LSL #1]\n"
    "st1h { z28.h }, p0, [x13, x10, LSL #1]\n"
    "st1h { z25.h }, p0, [x24, x10, LSL #1]\n"
    "st1h { z24.h }, p0, [x23]\n"
    "st1h { z21.h }, p0, [x23, x17, LSL #1]\n"
    "st1h { z23.h }, p0, [x23, x10, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
