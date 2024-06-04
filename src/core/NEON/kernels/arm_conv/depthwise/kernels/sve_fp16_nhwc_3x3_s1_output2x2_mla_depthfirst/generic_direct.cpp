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

void sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x17, #0x0\n"
    "mov x16, #0x0\n"
    "1:"  // Tile loop
    "str x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x2\n"
    "mov x25, #0x2\n"
    "str x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "cnth x15\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "mov x12, #0x0\n"
    "ldr x11, [%x[params_struct], %[offsetof_args_params]]\n"
    "ldr x10, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x22, x17, x24\n"  // offset = tile_i * ld_input_row
    "mul x21, x17, x23\n"  // offset = tile_i * ld_output_row
    "ldr x9, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "cmp x15, %x[n_channels]\n"
    "ld1rh { z27.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "add x28, x14, x14\n"
    "ld1rh { z26.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x20, XZR, x15\n"
    "madd x22, x16, x14, x22\n"  // offset += tile_j * ld_input_col
    "ld1h { z25.h }, p3/Z, [x11]\n"
    "ld1h { z0.h }, p3/Z, [x11, #1, MUL VL]\n"
    "add x27, x28, x14\n"
    "madd x21, x16, x13, x21\n"  // offset += tile_j * ld_output_col
    "ld1h { z1.h }, p3/Z, [x11, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x11, #3, MUL VL]\n"
    "mul x22, x22, x26\n"  // offset *= kernel_stride * output_size
    "ld1h { z3.h }, p3/Z, [x11, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x11, #5, MUL VL]\n"
    "mul x21, x21, x25\n"  // offset *= output_tile_size
    "ld1h { z5.h }, p3/Z, [x11, #6, MUL VL]\n"
    "ld1h { z6.h }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "add x10, x10, x22, LSL #1\n"  // inptr[0] += offset * sizeof(__fp16)
    "add x26, x10, x24, LSL #1\n"
    "ld1h { z10.h }, p2/Z, [x10]\n"
    "ld1h { z11.h }, p2/Z, [x10, x27, LSL #1]\n"
    "add x25, x26, x24, LSL #1\n"
    "add x9, x9, x21, LSL #1\n"  // outptrs[0] += offset * sizeof(__fp16)
    "add x24, x25, x24, LSL #1\n"
    "ld1h { z9.h }, p2/Z, [x26, x14, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x26, x28, LSL #1]\n"
    "add x23, x9, x23, LSL #1\n"
    "ld1h { z7.h }, p3/Z, [x11, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x11, #-7, MUL VL]\n"
    "addvl x11, x11, #-6\n"
    "ld1h { z13.h }, p2/Z, [x25, x14, LSL #1]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z24, z25\n fmla z24.h, p3/M, z4.h, z9.h\n"
    "movprfx z23, z25\n fmla z23.h, p3/M, z3.h, z9.h\n"
    "whilelt p1.h, x15, %x[n_channels]\n"
    "inch x12\n"
    "movprfx z22, z25\n fmla z22.h, p3/M, z1.h, z9.h\n"
    "movprfx z21, z25\n fmla z21.h, p3/M, z0.h, z9.h\n"
    "ld1h { z18.h }, p2/Z, [x24]\n"
    "inch x15\n"
    "mov p0.b, p2.b\n"
    "ld1h { z25.h }, p3/Z, [x11]\n"
    "inch x20\n"
    "fmla z24.h, p3/M, z0.h, z10.h\n"
    "ld1h { z20.h }, p2/Z, [x25, x28, LSL #1]\n"
    "fmla z23.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x27, LSL #1]\n"
    "fmla z22.h, p3/M, z2.h, z12.h\n"
    "fmla z21.h, p3/M, z1.h, z12.h\n"
    "fmla z24.h, p3/M, z5.h, z12.h\n"
    "fmla z23.h, p3/M, z4.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x14, LSL #1]\n"
    "fmla z22.h, p3/M, z6.h, z18.h\n"
    "ld1h { z18.h }, p2/Z, [x10, x28, LSL #1]\n"
    "addvl x10, x10, #1\n"
    "fmla z21.h, p3/M, z3.h, z13.h\n"
    "fmla z24.h, p3/M, z7.h, z13.h\n"
    "fmla z23.h, p3/M, z6.h, z13.h\n"
    "fmla z22.h, p3/M, z4.h, z13.h\n"
    "fmla z21.h, p3/M, z8.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x26]\n"
    "fmla z24.h, p3/M, z1.h, z16.h\n"
    "fmla z23.h, p3/M, z0.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x27, LSL #1]\n"
    "addvl x26, x26, #1\n"
    "fmla z22.h, p3/M, z5.h, z20.h\n"
    "fmla z21.h, p3/M, z4.h, z20.h\n"
    "ld1h { z4.h }, p3/Z, [x11, #5, MUL VL]\n"
    "fmla z24.h, p3/M, z2.h, z18.h\n"
    "fmla z23.h, p3/M, z1.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x25]\n"
    "ld1h { z1.h }, p3/Z, [x11, #2, MUL VL]\n"
    "fmla z22.h, p3/M, z0.h, z17.h\n"
    "ld1h { z0.h }, p3/Z, [x11, #1, MUL VL]\n"
    "fmla z21.h, p3/M, z2.h, z16.h\n"
    "ld1h { z2.h }, p3/Z, [x11, #3, MUL VL]\n"
    "fmla z24.h, p3/M, z8.h, z20.h\n"
    "fmla z23.h, p3/M, z7.h, z20.h\n"
    "ld1h { z18.h }, p2/Z, [x25, x27, LSL #1]\n"
    "addvl x25, x25, #1\n"
    "fmla z22.h, p3/M, z3.h, z19.h\n"
    "fmla z21.h, p3/M, z5.h, z18.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x14, LSL #1]\n"
    "ld1h { z3.h }, p3/Z, [x11, #4, MUL VL]\n"
    "ld1h { z13.h }, p1/Z, [x25, x14, LSL #1]\n"
    "fmla z23.h, p3/M, z5.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x28, LSL #1]\n"
    "whilelt p2.h, x12, %x[n_channels]\n"
    "ld1h { z5.h }, p3/Z, [x11, #6, MUL VL]\n"
    "cmp x15, %x[n_channels]\n"
    "addvl x24, x24, #1\n"
    "fmla z22.h, p3/M, z7.h, z17.h\n"
    "fmla z21.h, p3/M, z6.h, z17.h\n"
    "ld1h { z11.h }, p1/Z, [x10, x27, LSL #1]\n"
    "fmla z24.h, p3/M, z6.h, z19.h\n"
    "ld1h { z6.h }, p3/Z, [x11, #7, MUL VL]\n"
    "addvl x11, x11, #16\n"
    "ld1h { z9.h }, p1/Z, [x26, x14, LSL #1]\n"
    "fmla z23.h, p3/M, z8.h, z18.h\n"
    "ld1h { z10.h }, p1/Z, [x10]\n"
    "fmla z22.h, p3/M, z8.h, z16.h\n"
    "fmla z21.h, p3/M, z7.h, z16.h\n"
    "ld1h { z12.h }, p1/Z, [x26, x28, LSL #1]\n"
    "fmax z24.h, p3/M, z24.h, z27.h\n"
    "ld1h { z7.h }, p3/Z, [x11, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x11, #-7, MUL VL]\n"
    "addvl x11, x11, #-6\n"
    "fmax z23.h, p3/M, z23.h, z27.h\n"
    "fmin z24.h, p3/M, z24.h, z26.h\n"
    "fmax z22.h, p3/M, z22.h, z27.h\n"
    "fmax z21.h, p3/M, z21.h, z27.h\n"
    "fmin z23.h, p3/M, z23.h, z26.h\n"
    "fmin z22.h, p3/M, z22.h, z26.h\n"
    "st1h { z24.h }, p0, [x9]\n"
    "fmin z21.h, p3/M, z21.h, z26.h\n"
    "st1h { z23.h }, p0, [x9, x13, LSL #1]\n"
    "addvl x9, x9, #1\n"
    "st1h { z22.h }, p0, [x23]\n"
    "st1h { z21.h }, p0, [x23, x13, LSL #1]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z24, z25\n fmla z24.h, p3/M, z4.h, z9.h\n"
    "movprfx z23, z25\n fmla z23.h, p3/M, z3.h, z9.h\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z22, z25\n fmla z22.h, p3/M, z1.h, z9.h\n"
    "movprfx z21, z25\n fmla z21.h, p3/M, z0.h, z9.h\n"
    "ld1h { z18.h }, p2/Z, [x24]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "mov p0.b, p2.b\n"
    "add x16, x16, #0x1\n"
    "add x20, x17, #0x1\n"
    "fmla z24.h, p3/M, z0.h, z10.h\n"
    "ld1h { z20.h }, p2/Z, [x25, x28, LSL #1]\n"
    "fmla z23.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x27, LSL #1]\n"
    "cmp x16, x22\n"
    "fmla z22.h, p3/M, z2.h, z12.h\n"
    "fmla z21.h, p3/M, z1.h, z12.h\n"
    "csel x17, x17, x20, LT\n"
    "csel x16, x16, XZR, LT\n"
    "fmla z24.h, p3/M, z5.h, z12.h\n"
    "fmla z23.h, p3/M, z4.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x10, x14, LSL #1]\n"
    "fmla z22.h, p3/M, z6.h, z18.h\n"
    "ld1h { z18.h }, p2/Z, [x10, x28, LSL #1]\n"
    "fmla z21.h, p3/M, z3.h, z13.h\n"
    "cmp x17, x21\n"
    "fmla z24.h, p3/M, z7.h, z13.h\n"
    "fmla z23.h, p3/M, z6.h, z13.h\n"
    "fmla z22.h, p3/M, z4.h, z13.h\n"
    "fmla z21.h, p3/M, z8.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x26]\n"
    "fmla z24.h, p3/M, z1.h, z16.h\n"
    "fmla z23.h, p3/M, z0.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x26, x27, LSL #1]\n"
    "fmla z22.h, p3/M, z5.h, z20.h\n"
    "fmla z21.h, p3/M, z4.h, z20.h\n"
    "fmla z24.h, p3/M, z2.h, z18.h\n"
    "fmla z23.h, p3/M, z1.h, z18.h\n"
    "ld1h { z19.h }, p2/Z, [x25]\n"
    "fmla z22.h, p3/M, z0.h, z17.h\n"
    "fmla z21.h, p3/M, z2.h, z16.h\n"
    "fmla z24.h, p3/M, z8.h, z20.h\n"
    "fmla z23.h, p3/M, z7.h, z20.h\n"
    "ld1h { z18.h }, p2/Z, [x25, x27, LSL #1]\n"
    "fmla z22.h, p3/M, z3.h, z19.h\n"
    "fmla z21.h, p3/M, z5.h, z18.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x24, x14, LSL #1]\n"
    "fmla z23.h, p3/M, z5.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x24, x28, LSL #1]\n"
    "fmla z22.h, p3/M, z7.h, z17.h\n"
    "fmla z21.h, p3/M, z6.h, z17.h\n"
    "fmla z24.h, p3/M, z6.h, z19.h\n"
    "fmla z23.h, p3/M, z8.h, z18.h\n"
    "fmla z22.h, p3/M, z8.h, z16.h\n"
    "fmla z21.h, p3/M, z7.h, z16.h\n"
    "fmax z24.h, p3/M, z24.h, z27.h\n"
    "fmax z23.h, p3/M, z23.h, z27.h\n"
    "fmin z24.h, p3/M, z24.h, z26.h\n"
    "fmin z23.h, p3/M, z23.h, z26.h\n"
    "fmax z22.h, p3/M, z22.h, z27.h\n"
    "fmax z21.h, p3/M, z21.h, z27.h\n"
    "st1h { z24.h }, p0, [x9]\n"
    "st1h { z23.h }, p0, [x9, x13, LSL #1]\n"
    "fmin z22.h, p3/M, z22.h, z26.h\n"
    "fmin z21.h, p3/M, z21.h, z26.h\n"
    "st1h { z22.h }, p0, [x23]\n"
    "st1h { z21.h }, p0, [x23, x13, LSL #1]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
