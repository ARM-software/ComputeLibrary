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

void sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x26, #0x2\n"
    "mov x25, #0x2\n"
    "str x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cnth x17\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "mov x15, #0x0\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "mul x20, x6, x24\n"  // offset = tile_i * ld_input_row
    "add x12, x8, x8\n"
    "ldr x11, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x10, x12, x8\n"
    "cmp x17, %x[n_channels]\n"
    "ld1rh { z15.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mul x22, x6, x23\n"  // offset = tile_i * ld_output_row
    "add x9, x10, x8\n"
    "ld1rh { z28.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x21, XZR, x17\n"
    "madd x20, x7, x8, x20\n"  // offset += tile_j * ld_input_col
    "add x28, x9, x8\n"
    "ld1h { z29.h }, p3/Z, [x11]\n"
    "ld1h { z0.h }, p3/Z, [x11, #1, MUL VL]\n"
    "ld1h { z1.h }, p3/Z, [x11, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x11, #3, MUL VL]\n"
    "madd x22, x7, x16, x22\n"  // offset += tile_j * ld_output_col
    "ld1h { z3.h }, p3/Z, [x11, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x11, #5, MUL VL]\n"
    "addvl x11, x11, #6\n"
    "mul x20, x20, x26\n"  // offset *= kernel_stride * output_size
    "mul x22, x22, x25\n"  // offset *= output_tile_size
    "add x14, x14, x20, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x20, x14, x24, LSL #1\n"
    "add x27, x20, x24, LSL #1\n"
    "ld1h { z5.h }, p2/Z, [x14]\n"
    "ld1h { z6.h }, p2/Z, [x14, x8, LSL #1]\n"
    "add x26, x27, x24, LSL #1\n"
    "add x25, x26, x24, LSL #1\n"
    "ld1h { z7.h }, p2/Z, [x20]\n"
    "ld1h { z8.h }, p2/Z, [x20, x8, LSL #1]\n"
    "add x13, x13, x22, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "add x24, x25, x24, LSL #1\n"
    "add x23, x13, x23, LSL #1\n"
    "ld1h { z9.h }, p2/Z, [x14, x12, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x20, x12, LSL #1]\n"
    "ld1h { z11.h }, p2/Z, [x14, x10, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x14, x9, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x20, x28, LSL #1]\n"
    "ld1h { z14.h }, p2/Z, [x27]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z30, z29\n fmla z30.h, p3/M, z0.h, z5.h\n"
    "movprfx z31, z29\n fmla z31.h, p3/M, z0.h, z6.h\n"
    "ld1h { z25.h }, p2/Z, [x20, x10, LSL #1]\n"
    "whilelt p1.h, x17, %x[n_channels]\n"
    "movprfx z27, z29\n fmla z27.h, p3/M, z0.h, z7.h\n"
    "movprfx z26, z29\n fmla z26.h, p3/M, z0.h, z8.h\n"
    "ld1h { z23.h }, p3/Z, [x11]\n"
    "inch x15\n"
    "inch x17\n"
    "mov p0.b, p2.b\n"
    "inch x21\n"
    "fmla z30.h, p3/M, z1.h, z6.h\n"
    "ld1h { z22.h }, p2/Z, [x20, x9, LSL #1]\n"
    "addvl x20, x20, #1\n"
    "fmla z31.h, p3/M, z1.h, z9.h\n"
    "fmla z27.h, p3/M, z1.h, z8.h\n"
    "fmla z26.h, p3/M, z1.h, z13.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.h, p3/M, z2.h, z9.h\n"
    "ld1h { z18.h }, p2/Z, [x14, x28, LSL #1]\n"
    "addvl x14, x14, #1\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "fmla z27.h, p3/M, z2.h, z13.h\n"
    "fmla z26.h, p3/M, z2.h, z25.h\n"
    "ld1h { z16.h }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.h, p3/M, z3.h, z11.h\n"
    "ld1h { z20.h }, p2/Z, [x27, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z12.h\n"
    "fmla z27.h, p3/M, z3.h, z25.h\n"
    "fmla z26.h, p3/M, z3.h, z22.h\n"
    "ld1h { z17.h }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.h, p3/M, z4.h, z12.h\n"
    "ld1h { z19.h }, p2/Z, [x27, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z18.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x10, LSL #1]\n"
    "fmla z27.h, p3/M, z4.h, z22.h\n"
    "fmla z26.h, p3/M, z4.h, z10.h\n"
    "ld1h { z0.h }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z30.h, p3/M, z23.h, z7.h\n"
    "ld1h { z7.h }, p1/Z, [x20]\n"
    "fmla z31.h, p3/M, z23.h, z8.h\n"
    "fmla z27.h, p3/M, z23.h, z14.h\n"
    "fmla z26.h, p3/M, z23.h, z20.h\n"
    "ld1h { z18.h }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.h, p3/M, z21.h, z8.h\n"
    "ld1h { z1.h }, p2/Z, [x27, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z21.h, z13.h\n"
    "fmla z27.h, p3/M, z21.h, z20.h\n"
    "fmla z26.h, p3/M, z21.h, z19.h\n"
    "ld1h { z5.h }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.h, p3/M, z16.h, z13.h\n"
    "ld1h { z24.h }, p2/Z, [x27, x9, LSL #1]\n"
    "addvl x27, x27, #1\n"
    "fmla z31.h, p3/M, z16.h, z25.h\n"
    "fmla z27.h, p3/M, z16.h, z19.h\n"
    "fmla z26.h, p3/M, z16.h, z12.h\n"
    "ld1h { z16.h }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.h, p3/M, z17.h, z25.h\n"
    "ld1h { z25.h }, p2/Z, [x26]\n"
    "fmla z31.h, p3/M, z17.h, z22.h\n"
    "fmla z27.h, p3/M, z17.h, z12.h\n"
    "ld1h { z29.h }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z26.h, p3/M, z17.h, z24.h\n"
    "ld1h { z17.h }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.h, p3/M, z0.h, z22.h\n"
    "ld1h { z23.h }, p2/Z, [x26, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z22.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z27.h, p3/M, z0.h, z24.h\n"
    "fmla z26.h, p3/M, z0.h, z1.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #-7, MUL VL]\n"
    "fmla z30.h, p3/M, z18.h, z14.h\n"
    "ld1h { z10.h }, p2/Z, [x26, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z18.h, z20.h\n"
    "fmla z27.h, p3/M, z18.h, z25.h\n"
    "fmla z26.h, p3/M, z18.h, z23.h\n"
    "ld1h { z6.h }, p3/Z, [x11, #-6, MUL VL]\n"
    "fmla z30.h, p3/M, z5.h, z20.h\n"
    "ld1h { z0.h }, p2/Z, [x26, x10, LSL #1]\n"
    "fmla z31.h, p3/M, z5.h, z19.h\n"
    "fmla z27.h, p3/M, z5.h, z23.h\n"
    "fmla z26.h, p3/M, z5.h, z22.h\n"
    "ld1h { z20.h }, p3/Z, [x11, #-5, MUL VL]\n"
    "fmla z30.h, p3/M, z16.h, z19.h\n"
    "ld1h { z19.h }, p2/Z, [x26, x9, LSL #1]\n"
    "addvl x26, x26, #1\n"
    "fmla z31.h, p3/M, z16.h, z12.h\n"
    "fmla z27.h, p3/M, z16.h, z22.h\n"
    "fmla z26.h, p3/M, z16.h, z0.h\n"
    "ld1h { z18.h }, p3/Z, [x11, #-4, MUL VL]\n"
    "fmla z30.h, p3/M, z17.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x25]\n"
    "fmla z31.h, p3/M, z17.h, z24.h\n"
    "fmla z27.h, p3/M, z17.h, z0.h\n"
    "fmla z26.h, p3/M, z17.h, z19.h\n"
    "ld1h { z17.h }, p3/Z, [x11, #-3, MUL VL]\n"
    "fmla z30.h, p3/M, z21.h, z24.h\n"
    "ld1h { z9.h }, p2/Z, [x25, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z21.h, z1.h\n"
    "ld1h { z8.h }, p2/Z, [x25, x9, LSL #1]\n"
    "fmla z27.h, p3/M, z21.h, z19.h\n"
    "fmla z26.h, p3/M, z21.h, z10.h\n"
    "ld1h { z5.h }, p3/Z, [x11, #-2, MUL VL]\n"
    "fmla z30.h, p3/M, z6.h, z25.h\n"
    "ld1h { z25.h }, p2/Z, [x25, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z6.h, z23.h\n"
    "fmla z27.h, p3/M, z6.h, z16.h\n"
    "fmla z26.h, p3/M, z6.h, z9.h\n"
    "ld1h { z4.h }, p3/Z, [x11, #-1, MUL VL]\n"
    "fmla z30.h, p3/M, z20.h, z23.h\n"
    "ld1h { z24.h }, p2/Z, [x25, x10, LSL #1]\n"
    "fmla z31.h, p3/M, z20.h, z22.h\n"
    "fmla z27.h, p3/M, z20.h, z9.h\n"
    "fmla z26.h, p3/M, z20.h, z25.h\n"
    "ld1h { z23.h }, p3/Z, [x11]\n"
    "fmla z30.h, p3/M, z18.h, z22.h\n"
    "ld1h { z22.h }, p2/Z, [x25, x28, LSL #1]\n"
    "addvl x25, x25, #1\n"
    "fmla z31.h, p3/M, z18.h, z0.h\n"
    "fmla z27.h, p3/M, z18.h, z25.h\n"
    "fmla z26.h, p3/M, z18.h, z24.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.h, p3/M, z17.h, z0.h\n"
    "ld1h { z18.h }, p2/Z, [x24]\n"
    "fmla z31.h, p3/M, z17.h, z19.h\n"
    "fmla z27.h, p3/M, z17.h, z24.h\n"
    "fmla z26.h, p3/M, z17.h, z8.h\n"
    "ld1h { z20.h }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.h, p3/M, z5.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z5.h, z10.h\n"
    "ld1h { z14.h }, p1/Z, [x27]\n"
    "fmla z27.h, p3/M, z5.h, z8.h\n"
    "fmla z26.h, p3/M, z5.h, z22.h\n"
    "ld1h { z19.h }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.h, p3/M, z4.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z9.h\n"
    "fmla z27.h, p3/M, z4.h, z18.h\n"
    "ld1h { z18.h }, p2/Z, [x24, x10, LSL #1]\n"
    "fmla z26.h, p3/M, z4.h, z17.h\n"
    "ld1h { z0.h }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.h, p3/M, z23.h, z9.h\n"
    "ld1h { z13.h }, p1/Z, [x20, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z23.h, z25.h\n"
    "fmla z27.h, p3/M, z23.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x9, LSL #1]\n"
    "fmla z26.h, p3/M, z23.h, z16.h\n"
    "ld1h { z1.h }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.h, p3/M, z21.h, z25.h\n"
    "ld1h { z5.h }, p1/Z, [x14]\n"
    "fmla z31.h, p3/M, z21.h, z24.h\n"
    "fmla z27.h, p3/M, z21.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x28, LSL #1]\n"
    "whilelt p2.h, x15, %x[n_channels]\n"
    "cmp x17, %x[n_channels]\n"
    "addvl x24, x24, #1\n"
    "fmla z26.h, p3/M, z21.h, z18.h\n"
    "ld1h { z2.h }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.h, p3/M, z20.h, z24.h\n"
    "ld1h { z6.h }, p1/Z, [x14, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z20.h, z8.h\n"
    "fmla z27.h, p3/M, z20.h, z18.h\n"
    "ld1h { z11.h }, p1/Z, [x14, x10, LSL #1]\n"
    "fmla z26.h, p3/M, z20.h, z17.h\n"
    "ld1h { z3.h }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.h, p3/M, z19.h, z8.h\n"
    "ld1h { z8.h }, p1/Z, [x20, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z19.h, z22.h\n"
    "ld1h { z10.h }, p1/Z, [x20, x28, LSL #1]\n"
    "fmla z27.h, p3/M, z19.h, z17.h\n"
    "ld1h { z12.h }, p1/Z, [x14, x9, LSL #1]\n"
    "fmla z26.h, p3/M, z19.h, z16.h\n"
    "ld1h { z9.h }, p1/Z, [x14, x12, LSL #1]\n"
    "ld1h { z4.h }, p3/Z, [x11, #-7, MUL VL]\n"
    "addvl x11, x11, #-6\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "fmax z31.h, p3/M, z31.h, z15.h\n"
    "fmax z27.h, p3/M, z27.h, z15.h\n"
    "fmax z26.h, p3/M, z26.h, z15.h\n"
    "fmin z30.h, p3/M, z30.h, z28.h\n"
    "fmin z31.h, p3/M, z31.h, z28.h\n"
    "fmin z27.h, p3/M, z27.h, z28.h\n"
    "fmin z26.h, p3/M, z26.h, z28.h\n"
    "st1h { z30.h }, p0, [x13]\n"
    "st1h { z31.h }, p0, [x13, x16, LSL #1]\n"
    "addvl x13, x13, #1\n"
    "st1h { z27.h }, p0, [x23]\n"
    "st1h { z26.h }, p0, [x23, x16, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z30, z29\n fmla z30.h, p3/M, z0.h, z5.h\n"
    "movprfx z31, z29\n fmla z31.h, p3/M, z0.h, z6.h\n"
    "ld1h { z22.h }, p2/Z, [x20, x10, LSL #1]\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "movprfx z5, z29\n fmla z5.h, p3/M, z0.h, z7.h\n"
    "fmla z29.h, p3/M, z0.h, z8.h\n"
    "ld1h { z20.h }, p3/Z, [x11]\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "mov p0.b, p2.b\n"
    "add x7, x7, #0x1\n"
    "fmla z30.h, p3/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x20, x9, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z9.h\n"
    "add x20, x6, #0x1\n"
    "fmla z5.h, p3/M, z1.h, z8.h\n"
    "fmla z29.h, p3/M, z1.h, z13.h\n"
    "ld1h { z19.h }, p3/Z, [x11, #1, MUL VL]\n"
    "cmp x7, x22\n"
    "csel x6, x6, x20, LT\n"
    "csel x7, x7, XZR, LT\n"
    "fmla z30.h, p3/M, z2.h, z9.h\n"
    "ld1h { z16.h }, p2/Z, [x14, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z2.h, z11.h\n"
    "fmla z5.h, p3/M, z2.h, z13.h\n"
    "fmla z29.h, p3/M, z2.h, z22.h\n"
    "ld1h { z18.h }, p3/Z, [x11, #2, MUL VL]\n"
    "cmp x6, x21\n"
    "fmla z30.h, p3/M, z3.h, z11.h\n"
    "ld1h { z1.h }, p2/Z, [x27, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z12.h\n"
    "fmla z5.h, p3/M, z3.h, z22.h\n"
    "fmla z29.h, p3/M, z3.h, z6.h\n"
    "ld1h { z17.h }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.h, p3/M, z4.h, z12.h\n"
    "ld1h { z0.h }, p2/Z, [x27, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z16.h\n"
    "ld1h { z27.h }, p2/Z, [x27, x10, LSL #1]\n"
    "fmla z5.h, p3/M, z4.h, z6.h\n"
    "fmla z29.h, p3/M, z4.h, z10.h\n"
    "ld1h { z16.h }, p3/Z, [x11, #4, MUL VL]\n"
    "fmla z30.h, p3/M, z20.h, z7.h\n"
    "fmla z31.h, p3/M, z20.h, z8.h\n"
    "fmla z5.h, p3/M, z20.h, z14.h\n"
    "fmla z29.h, p3/M, z20.h, z1.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z30.h, p3/M, z19.h, z8.h\n"
    "ld1h { z26.h }, p2/Z, [x27, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z19.h, z13.h\n"
    "fmla z5.h, p3/M, z19.h, z1.h\n"
    "fmla z29.h, p3/M, z19.h, z0.h\n"
    "ld1h { z25.h }, p3/Z, [x11, #6, MUL VL]\n"
    "fmla z30.h, p3/M, z18.h, z13.h\n"
    "ld1h { z24.h }, p2/Z, [x27, x9, LSL #1]\n"
    "fmla z31.h, p3/M, z18.h, z22.h\n"
    "fmla z5.h, p3/M, z18.h, z0.h\n"
    "fmla z29.h, p3/M, z18.h, z27.h\n"
    "ld1h { z23.h }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "fmla z30.h, p3/M, z17.h, z22.h\n"
    "ld1h { z22.h }, p2/Z, [x26]\n"
    "fmla z31.h, p3/M, z17.h, z6.h\n"
    "fmla z5.h, p3/M, z17.h, z27.h\n"
    "fmla z29.h, p3/M, z17.h, z24.h\n"
    "ld1h { z20.h }, p3/Z, [x11, #-8, MUL VL]\n"
    "fmla z30.h, p3/M, z16.h, z6.h\n"
    "ld1h { z18.h }, p2/Z, [x26, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z16.h, z10.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x12, LSL #1]\n"
    "fmla z5.h, p3/M, z16.h, z24.h\n"
    "fmla z29.h, p3/M, z16.h, z26.h\n"
    "ld1h { z16.h }, p3/Z, [x11, #-7, MUL VL]\n"
    "fmla z30.h, p3/M, z21.h, z14.h\n"
    "ld1h { z19.h }, p2/Z, [x26, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z21.h, z1.h\n"
    "fmla z5.h, p3/M, z21.h, z22.h\n"
    "fmla z29.h, p3/M, z21.h, z18.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #-6, MUL VL]\n"
    "fmla z30.h, p3/M, z25.h, z1.h\n"
    "ld1h { z8.h }, p2/Z, [x26, x10, LSL #1]\n"
    "fmla z31.h, p3/M, z25.h, z0.h\n"
    "fmla z5.h, p3/M, z25.h, z18.h\n"
    "fmla z29.h, p3/M, z25.h, z17.h\n"
    "ld1h { z9.h }, p3/Z, [x11, #-5, MUL VL]\n"
    "fmla z30.h, p3/M, z23.h, z0.h\n"
    "ld1h { z11.h }, p2/Z, [x26, x9, LSL #1]\n"
    "fmla z31.h, p3/M, z23.h, z27.h\n"
    "fmla z5.h, p3/M, z23.h, z17.h\n"
    "fmla z29.h, p3/M, z23.h, z8.h\n"
    "ld1h { z6.h }, p3/Z, [x11, #-4, MUL VL]\n"
    "fmla z30.h, p3/M, z20.h, z27.h\n"
    "ld1h { z0.h }, p2/Z, [x25]\n"
    "fmla z31.h, p3/M, z20.h, z24.h\n"
    "fmla z5.h, p3/M, z20.h, z8.h\n"
    "fmla z29.h, p3/M, z20.h, z11.h\n"
    "ld1h { z4.h }, p3/Z, [x11, #-3, MUL VL]\n"
    "fmla z30.h, p3/M, z16.h, z24.h\n"
    "ld1h { z2.h }, p2/Z, [x25, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z16.h, z26.h\n"
    "ld1h { z27.h }, p2/Z, [x25, x9, LSL #1]\n"
    "fmla z5.h, p3/M, z16.h, z11.h\n"
    "fmla z29.h, p3/M, z16.h, z19.h\n"
    "ld1h { z16.h }, p3/Z, [x11, #-2, MUL VL]\n"
    "fmla z30.h, p3/M, z21.h, z22.h\n"
    "ld1h { z26.h }, p2/Z, [x25, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z21.h, z18.h\n"
    "fmla z5.h, p3/M, z21.h, z0.h\n"
    "fmla z29.h, p3/M, z21.h, z2.h\n"
    "ld1h { z25.h }, p3/Z, [x11, #-1, MUL VL]\n"
    "fmla z30.h, p3/M, z9.h, z18.h\n"
    "ld1h { z24.h }, p2/Z, [x25, x10, LSL #1]\n"
    "fmla z31.h, p3/M, z9.h, z17.h\n"
    "fmla z5.h, p3/M, z9.h, z2.h\n"
    "fmla z29.h, p3/M, z9.h, z26.h\n"
    "ld1h { z23.h }, p3/Z, [x11]\n"
    "fmla z30.h, p3/M, z6.h, z17.h\n"
    "ld1h { z22.h }, p2/Z, [x25, x28, LSL #1]\n"
    "fmla z31.h, p3/M, z6.h, z8.h\n"
    "fmla z5.h, p3/M, z6.h, z26.h\n"
    "fmla z29.h, p3/M, z6.h, z24.h\n"
    "ld1h { z21.h }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z30.h, p3/M, z4.h, z8.h\n"
    "ld1h { z18.h }, p2/Z, [x24]\n"
    "fmla z31.h, p3/M, z4.h, z11.h\n"
    "fmla z5.h, p3/M, z4.h, z24.h\n"
    "fmla z29.h, p3/M, z4.h, z27.h\n"
    "ld1h { z20.h }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z30.h, p3/M, z16.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z16.h, z19.h\n"
    "fmla z5.h, p3/M, z16.h, z27.h\n"
    "fmla z29.h, p3/M, z16.h, z22.h\n"
    "ld1h { z19.h }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z30.h, p3/M, z25.h, z0.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x12, LSL #1]\n"
    "fmla z31.h, p3/M, z25.h, z2.h\n"
    "fmla z5.h, p3/M, z25.h, z18.h\n"
    "ld1h { z18.h }, p2/Z, [x24, x10, LSL #1]\n"
    "fmla z29.h, p3/M, z25.h, z17.h\n"
    "fmla z30.h, p3/M, z23.h, z2.h\n"
    "fmla z31.h, p3/M, z23.h, z26.h\n"
    "fmla z5.h, p3/M, z23.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x9, LSL #1]\n"
    "fmla z29.h, p3/M, z23.h, z16.h\n"
    "fmla z30.h, p3/M, z21.h, z26.h\n"
    "fmla z31.h, p3/M, z21.h, z24.h\n"
    "fmla z5.h, p3/M, z21.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x28, LSL #1]\n"
    "fmla z29.h, p3/M, z21.h, z18.h\n"
    "fmla z30.h, p3/M, z20.h, z24.h\n"
    "fmla z31.h, p3/M, z20.h, z27.h\n"
    "fmla z5.h, p3/M, z20.h, z18.h\n"
    "fmla z29.h, p3/M, z20.h, z17.h\n"
    "fmla z30.h, p3/M, z19.h, z27.h\n"
    "fmla z31.h, p3/M, z19.h, z22.h\n"
    "fmla z5.h, p3/M, z19.h, z17.h\n"
    "fmla z29.h, p3/M, z19.h, z16.h\n"
    "fmax z30.h, p3/M, z30.h, z15.h\n"
    "fmax z31.h, p3/M, z31.h, z15.h\n"
    "fmax z5.h, p3/M, z5.h, z15.h\n"
    "fmin z30.h, p3/M, z30.h, z28.h\n"
    "fmin z31.h, p3/M, z31.h, z28.h\n"
    "fmax z29.h, p3/M, z29.h, z15.h\n"
    "fmin z5.h, p3/M, z5.h, z28.h\n"
    "st1h { z30.h }, p0, [x13]\n"
    "fmin z29.h, p3/M, z29.h, z28.h\n"
    "st1h { z31.h }, p0, [x13, x16, LSL #1]\n"
    "st1h { z5.h }, p0, [x23]\n"
    "st1h { z29.h }, p0, [x23, x16, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
