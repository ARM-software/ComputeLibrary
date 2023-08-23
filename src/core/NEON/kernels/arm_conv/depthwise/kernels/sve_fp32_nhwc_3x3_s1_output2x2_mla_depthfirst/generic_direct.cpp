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

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x10, #0x0\n"
    "mov x14, #0x0\n"
    "1:"  // Tile loop
    "str x10, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x25, #0x2\n"
    "mov x24, #0x2\n"
    "str x14, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "mul x21, x10, x23\n"  // offset = tile_i * ld_input_row
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ldr x12, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x20, x10, x22\n"  // offset = tile_i * ld_output_row
    "cntw x11\n"
    "madd x21, x14, x13, x21\n"  // offset += tile_j * ld_input_col
    "ldr x10, [%x[params_struct], %[offsetof_args_params]]\n"
    "ldr x9, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "madd x20, x14, x12, x20\n"  // offset += tile_j * ld_output_col
    "ldr x28, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "ld1w { z27.s }, p3/Z, [x10]\n"
    "add x27, x13, x13\n"
    "mul x21, x21, x25\n"  // offset *= kernel_stride * output_size
    "add x9, x9, x21, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "ld1w { z0.s }, p3/Z, [x10, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x10, #2, MUL VL]\n"
    "mul x20, x20, x24\n"  // offset *= output_tile_size
    "ld1w { z2.s }, p3/Z, [x10, #3, MUL VL]\n"
    "ld1w { z3.s }, p3/Z, [x10, #4, MUL VL]\n"
    "add x26, x9, x23, LSL #2\n"
    "ld1w { z4.s }, p3/Z, [x10, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x10, #6, MUL VL]\n"
    "add x25, x26, x23, LSL #2\n"
    "add x24, x27, x13\n"
    "ld1w { z6.s }, p3/Z, [x10, #7, MUL VL]\n"
    "addvl x10, x10, #16\n"
    "add x28, x28, x20, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "ld1rw { z26.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "cmp x11, %x[n_channels]\n"
    "add x23, x25, x23, LSL #2\n"
    "ld1rw { z25.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1w { z7.s }, p3/Z, [x10, #-8, MUL VL]\n"
    "add x22, x28, x22, LSL #2\n"
    "mov x21, #0x0\n"
    "ld1w { z8.s }, p3/Z, [x10, #-7, MUL VL]\n"
    "ld1w { z9.s }, p2/Z, [x26, x13, LSL #2]\n"
    "sub x20, XZR, x11\n"
    "ld1w { z10.s }, p2/Z, [x9]\n"
    "ld1w { z11.s }, p2/Z, [x9, x24, LSL #2]\n"
    "addvl x10, x10, #-6\n"
    "ld1w { z12.s }, p2/Z, [x26, x27, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x25, x13, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z24, z27\n fmla z24.s, p3/M, z4.s, z9.s\n"
    "movprfx z23, z27\n fmla z23.s, p3/M, z3.s, z9.s\n"
    "whilelt p1.s, x11, %x[n_channels]\n"
    "incw x21\n"
    "movprfx z22, z27\n fmla z22.s, p3/M, z1.s, z9.s\n"
    "movprfx z21, z27\n fmla z21.s, p3/M, z0.s, z9.s\n"
    "ld1w { z18.s }, p2/Z, [x23]\n"
    "incw x11\n"
    "fmla z24.s, p3/M, z0.s, z10.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x23, x24, LSL #2]\n"
    "ld1w { z20.s }, p2/Z, [x25, x27, LSL #2]\n"
    "fmla z22.s, p3/M, z2.s, z12.s\n"
    "fmla z21.s, p3/M, z1.s, z12.s\n"
    "mov p0.b, p2.b\n"
    "ld1w { z27.s }, p3/Z, [x10]\n"
    "fmla z24.s, p3/M, z5.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x9, x13, LSL #2]\n"
    "incw x20\n"
    "fmla z22.s, p3/M, z6.s, z18.s\n"
    "fmla z21.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x9, x27, LSL #2]\n"
    "addvl x9, x9, #1\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "fmla z22.s, p3/M, z4.s, z13.s\n"
    "fmla z21.s, p3/M, z8.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "fmla z24.s, p3/M, z1.s, z16.s\n"
    "fmla z23.s, p3/M, z0.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x24, LSL #2]\n"
    "addvl x26, x26, #1\n"
    "fmla z22.s, p3/M, z5.s, z20.s\n"
    "fmla z21.s, p3/M, z4.s, z20.s\n"
    "ld1w { z4.s }, p3/Z, [x10, #5, MUL VL]\n"
    "fmla z24.s, p3/M, z2.s, z18.s\n"
    "fmla z23.s, p3/M, z1.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x25]\n"
    "ld1w { z1.s }, p3/Z, [x10, #2, MUL VL]\n"
    "fmla z22.s, p3/M, z0.s, z17.s\n"
    "fmla z21.s, p3/M, z2.s, z16.s\n"
    "ld1w { z0.s }, p3/Z, [x10, #1, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x10, #3, MUL VL]\n"
    "fmla z24.s, p3/M, z8.s, z20.s\n"
    "fmla z23.s, p3/M, z7.s, z20.s\n"
    "ld1w { z18.s }, p2/Z, [x25, x24, LSL #2]\n"
    "addvl x25, x25, #1\n"
    "fmla z22.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z5.s, z18.s\n"
    "ld1w { z13.s }, p1/Z, [x25, x13, LSL #2]\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x23, x13, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x23, x27, LSL #2]\n"
    "fmla z22.s, p3/M, z7.s, z17.s\n"
    "fmla z21.s, p3/M, z6.s, z17.s\n"
    "ld1w { z3.s }, p3/Z, [x10, #4, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x10, #6, MUL VL]\n"
    "fmla z24.s, p3/M, z6.s, z19.s\n"
    "fmla z23.s, p3/M, z8.s, z18.s\n"
    "fmax z24.s, p3/M, z24.s, z26.s\n"
    "fmax z23.s, p3/M, z23.s, z26.s\n"
    "fmla z22.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "fmax z22.s, p3/M, z22.s, z26.s\n"
    "fmax z21.s, p3/M, z21.s, z26.s\n"
    "ld1w { z6.s }, p3/Z, [x10, #7, MUL VL]\n"
    "addvl x10, x10, #16\n"
    "whilelt p2.s, x21, %x[n_channels]\n"
    "ld1w { z9.s }, p1/Z, [x26, x13, LSL #2]\n"
    "cmp x11, %x[n_channels]\n"
    "fmin z24.s, p3/M, z24.s, z25.s\n"
    "ld1w { z10.s }, p1/Z, [x9]\n"
    "ld1w { z11.s }, p1/Z, [x9, x24, LSL #2]\n"
    "fmin z23.s, p3/M, z23.s, z25.s\n"
    "fmin z22.s, p3/M, z22.s, z25.s\n"
    "ld1w { z12.s }, p1/Z, [x26, x27, LSL #2]\n"
    "st1w { z24.s }, p0, [x28]\n"
    "fmin z21.s, p3/M, z21.s, z25.s\n"
    "addvl x23, x23, #1\n"
    "st1w { z23.s }, p0, [x28, x12, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x10, #-8, MUL VL]\n"
    "st1w { z22.s }, p0, [x22]\n"
    "addvl x28, x28, #1\n"
    "ld1w { z8.s }, p3/Z, [x10, #-7, MUL VL]\n"
    "addvl x10, x10, #-6\n"
    "st1w { z21.s }, p0, [x22, x12, LSL #2]\n"
    "addvl x22, x22, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z24, z27\n fmla z24.s, p3/M, z4.s, z9.s\n"
    "movprfx z23, z27\n fmla z23.s, p3/M, z3.s, z9.s\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x10, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z22, z27\n fmla z22.s, p3/M, z1.s, z9.s\n"
    "movprfx z21, z27\n fmla z21.s, p3/M, z0.s, z9.s\n"
    "ld1w { z18.s }, p2/Z, [x23]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "fmla z24.s, p3/M, z0.s, z10.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x23, x24, LSL #2]\n"
    "ld1w { z20.s }, p2/Z, [x25, x27, LSL #2]\n"
    "fmla z22.s, p3/M, z2.s, z12.s\n"
    "fmla z21.s, p3/M, z1.s, z12.s\n"
    "add x14, x14, #0x1\n"
    "cmp x14, x20\n"
    "fmla z24.s, p3/M, z5.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x9, x13, LSL #2]\n"
    "add x21, x10, #0x1\n"
    "fmla z22.s, p3/M, z6.s, z18.s\n"
    "fmla z21.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x9, x27, LSL #2]\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "csel x10, x10, x21, LT\n"
    "mov p0.b, p2.b\n"
    "fmla z22.s, p3/M, z4.s, z13.s\n"
    "fmla z21.s, p3/M, z8.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "csel x14, x14, XZR, LT\n"
    "fmla z24.s, p3/M, z1.s, z16.s\n"
    "fmla z23.s, p3/M, z0.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x24, LSL #2]\n"
    "cmp x10, x20\n"
    "fmla z22.s, p3/M, z5.s, z20.s\n"
    "fmla z21.s, p3/M, z4.s, z20.s\n"
    "fmla z24.s, p3/M, z2.s, z18.s\n"
    "fmla z23.s, p3/M, z1.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x25]\n"
    "fmla z22.s, p3/M, z0.s, z17.s\n"
    "fmla z21.s, p3/M, z2.s, z16.s\n"
    "fmla z24.s, p3/M, z8.s, z20.s\n"
    "fmla z23.s, p3/M, z7.s, z20.s\n"
    "ld1w { z18.s }, p2/Z, [x25, x24, LSL #2]\n"
    "fmla z22.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z5.s, z18.s\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x23, x13, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x23, x27, LSL #2]\n"
    "fmla z22.s, p3/M, z7.s, z17.s\n"
    "fmla z21.s, p3/M, z6.s, z17.s\n"
    "fmla z24.s, p3/M, z6.s, z19.s\n"
    "fmla z23.s, p3/M, z8.s, z18.s\n"
    "fmax z24.s, p3/M, z24.s, z26.s\n"
    "fmax z23.s, p3/M, z23.s, z26.s\n"
    "fmla z22.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "fmax z22.s, p3/M, z22.s, z26.s\n"
    "fmax z21.s, p3/M, z21.s, z26.s\n"
    "fmin z24.s, p3/M, z24.s, z25.s\n"
    "fmin z23.s, p3/M, z23.s, z25.s\n"
    "st1w { z24.s }, p0, [x28]\n"
    "fmin z22.s, p3/M, z22.s, z25.s\n"
    "fmin z21.s, p3/M, z21.s, z25.s\n"
    "st1w { z23.s }, p0, [x28, x12, LSL #2]\n"
    "st1w { z22.s }, p0, [x22]\n"
    "st1w { z21.s }, p0, [x22, x12, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
