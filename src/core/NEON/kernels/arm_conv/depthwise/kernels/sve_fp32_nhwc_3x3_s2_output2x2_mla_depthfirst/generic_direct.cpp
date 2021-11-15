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

#if __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
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
    "mov x7, #0x0\n"
    "mov x8, #0x0\n"
    "1:"  // Tile loop
    "str x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x23, #0x4\n"
    "str x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov x17, #0x2\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_params]]\n"
    "mov x15, #0x0\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cntw x14\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "sub x12, XZR, x14\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x19, x7, x22\n" // offset = tile_i * ld_input_row
    "ldr x20, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x19, x8, x13, x19\n" // offset += tile_j * ld_input_col
    "ldr x11, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "mul x19, x19, x23\n" // offset *= kernel_stride * output_size
    "ldr x10, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x21, x21, x19, LSL #2\n" // inptr[0] += offset * sizeof(float)
    "ld1rw { z19.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x9, x21, x22, LSL #2\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "add x28, x9, x22, LSL #2\n"
    "ld1w { z17.s }, p3/Z, [x16]\n"
    "add x27, x28, x22, LSL #2\n"
    "ld1w { z0.s }, p3/Z, [x16, #1, MUL VL]\n"
    "add x26, x27, x22, LSL #2\n"
    "ld1w { z1.s }, p3/Z, [x16, #2, MUL VL]\n"
    "add x25, x13, x13\n"
    "ld1w { z2.s }, p3/Z, [x16, #3, MUL VL]\n"
    "add x24, x25, x13\n"
    "ld1w { z3.s }, p3/Z, [x16, #4, MUL VL]\n"
    "add x23, x24, x13\n"
    "ld1w { z4.s }, p3/Z, [x16, #5, MUL VL]\n"
    "mul x19, x7, x20\n" // offset = tile_i * ld_output_row
    "ld1w { z5.s }, p3/Z, [x16, #6, MUL VL]\n"
    "madd x19, x8, x11, x19\n" // offset += tile_j * ld_output_col
    "ld1w { z6.s }, p3/Z, [x16, #7, MUL VL]\n"
    "mul x19, x19, x17\n" // offset *= output_tile_size
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1w { z9.s }, p2/Z, [x28, x25, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x21]\n"
    "add x10, x10, x19, LSL #2\n" // outptrs[0] += offset * sizeof(float)
    "ld1w { z11.s }, p2/Z, [x21, x13, LSL #2]\n"
    "add x22, x10, x20, LSL #2\n"
    "ld1w { z12.s }, p2/Z, [x21, x24, LSL #2]\n"
    "addvl x16, x16, #16\n"
    "ld1w { z13.s }, p2/Z, [x21, x23, LSL #2]\n"
    "cmp x14, %x[n_channels]\n"
    "ld1w { z7.s }, p3/Z, [x16, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x16, #-7, MUL VL]\n"
    "addvl x16, x16, #-6\n"
    "ld1w { z14.s }, p2/Z, [x9]\n"
    "ld1w { z15.s }, p2/Z, [x9, x13, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x21, x25, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z31, z17\n fmla z31.s, p3/M, z8.s, z9.s\n"
    "whilelt p1.s, x14, %x[n_channels]\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z6.s, z9.s\n"
    "incw x12\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "mov p0.b, p2.b\n"
    "movprfx z28, z17\n fmla z28.s, p3/M, z0.s, z9.s\n"
    "ld1w { z17.s }, p3/Z, [x16]\n"
    "incw x15\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "addvl x21, x21, #1\n"
    "ld1w { z10.s }, p1/Z, [x21]\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "incw x14\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x24, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x9, x23, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x9, x25, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x27]\n"
    "addvl x9, x9, #1\n"
    "fmla z30.s, p3/M, z0.s, z16.s\n"
    "fmla z29.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x27, x23, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x28]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z15.s\n"
    "ld1w { z0.s }, p3/Z, [x16, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x23, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "ld1w { z9.s }, p1/Z, [x28, x25, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z16.s\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26]\n"
    "fmla z28.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x26, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z12.s\n"
    "ld1w { z4.s }, p3/Z, [x16, #5, MUL VL]\n"
    "fmla z29.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26, x25, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x27, x25, LSL #2]\n"
    "addvl x27, x27, #1\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p1/Z, [x21, x24, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z11.s\n"
    "ld1w { z1.s }, p3/Z, [x16, #2, MUL VL]\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p1/Z, [x21, x23, LSL #2]\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmla z28.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x26, x24, LSL #2]\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmla z29.s, p3/M, z5.s, z16.s\n"
    "ld1w { z5.s }, p3/Z, [x16, #6, MUL VL]\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x10]\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "fmla z29.s, p3/M, z8.s, z15.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x23, LSL #2]\n"
    "whilelt p2.s, x15, %x[n_channels]\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "ld1w { z2.s }, p3/Z, [x16, #3, MUL VL]\n"
    "addvl x26, x26, #1\n"
    "fmla z28.s, p3/M, z3.s, z16.s\n"
    "ld1w { z16.s }, p1/Z, [x21, x25, LSL #2]\n"
    "cmp x14, %x[n_channels]\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "ld1w { z3.s }, p3/Z, [x16, #4, MUL VL]\n"
    "st1w { z30.s }, p0, [x10, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z14.s\n"
    "ld1w { z14.s }, p1/Z, [x9]\n"
    "addvl x10, x10, #1\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z29.s }, p0, [x22]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p1/Z, [x9, x13, LSL #2]\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x21, x13, LSL #2]\n"
    "ld1w { z6.s }, p3/Z, [x16, #7, MUL VL]\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "addvl x16, x16, #16\n"
    "ld1w { z7.s }, p3/Z, [x16, #-8, MUL VL]\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "ld1w { z8.s }, p3/Z, [x16, #-7, MUL VL]\n"
    "addvl x16, x16, #-6\n"
    "st1w { z28.s }, p0, [x22, x11, LSL #2]\n"
    "addvl x22, x22, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z31, z17\n fmla z31.s, p3/M, z8.s, z9.s\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z6.s, z9.s\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "add x21, x7, #0x1\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "movprfx z28, z17\n fmla z28.s, p3/M, z0.s, z9.s\n"
    "ldr x19, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "add x8, x8, #0x1\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "cmp x8, x19\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x9, x23, LSL #2]\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x24, LSL #2]\n"
    "csel x8, x8, XZR, LT\n"
    "fmla z30.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x9, x25, LSL #2]\n"
    "csel x7, x7, x21, LT\n"
    "fmla z31.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x27]\n"
    "cmp x7, x20\n"
    "fmla z30.s, p3/M, z0.s, z16.s\n"
    "fmla z29.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x27, x23, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x28]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z15.s\n"
    "fmla z31.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x28, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x23, LSL #2]\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x24, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z16.s\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26]\n"
    "fmla z28.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x26, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z12.s\n"
    "fmla z29.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26, x25, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x27, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z30.s, p3/M, z8.s, z11.s\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmla z28.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x26, x24, LSL #2]\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmla z29.s, p3/M, z5.s, z16.s\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x10]\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "fmla z29.s, p3/M, z8.s, z15.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x23, LSL #2]\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "st1w { z30.s }, p0, [x10, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z16.s\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "fmla z28.s, p3/M, z7.s, z14.s\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z29.s }, p0, [x22]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "st1w { z28.s }, p0, [x22, x11, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE)
