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

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_strided_impl(
  const float *const inptr,
  const size_t in_row_stride,
  const size_t in_col_stride,
  float *const outptr,
  const size_t out_row_stride,
  const size_t out_col_stride,
  const void *params,
  unsigned long n_channels,
  const float activation_min,
  const float activation_max
)
{
  const float minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ptrue p2.b\n"
    "ld1w { z15.s }, p2/Z, [%x[params]]\n"
    "mov z14.d, z15.d\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "whilelt p1.s, XZR, %x[n_channels]\n"
    "mov z12.d, z15.d\n"
    "ld1w { z11.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "mov x26, %x[inptr]\n"
    "mov z10.d, z15.d\n"
    "ld1w { z9.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "add x25, x26, %x[in_row_stride], LSL #2\n"
    "mov z8.d, z15.d\n"
    "ld1w { z7.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "add x24, x25, %x[in_row_stride], LSL #2\n"
    "ld1w { z6.s }, p2/Z, [%x[params], #5, MUL VL]\n"
    "add x23, x24, %x[in_row_stride], LSL #2\n"
    "ld1w { z5.s }, p2/Z, [%x[params], #6, MUL VL]\n"
    "mov x22, %x[outptr]\n"
    "ld1w { z4.s }, p2/Z, [%x[params], #7, MUL VL]\n"
    "add x21, x22, %x[out_row_stride], LSL #2\n"
    "ld1w { z3.s }, p1/Z, [x26]\n"
    "add x20, %x[in_col_stride], %x[in_col_stride]\n"
    "ld1w { z2.s }, p1/Z, [x26, %x[in_col_stride], LSL #2]\n"
    "add x19, x20, %x[in_col_stride]\n"
    "ld1w { z1.s }, p1/Z, [x25]\n"
    "addvl %x[params], %x[params], #16\n"
    "ld1w { z0.s }, p1/Z, [x25, %x[in_col_stride], LSL #2]\n"
    "decw %x[n_channels]\n"
    "ld1w { z31.s }, p2/Z, [%x[params], #-8, MUL VL]\n"
    "cmp %x[n_channels], XZR\n"
    "ld1w { z30.s }, p2/Z, [%x[params], #-7, MUL VL]\n"
    "addvl %x[params], %x[params], #-6\n"
    "ld1w { z29.s }, p1/Z, [x26, x20, LSL #2]\n"
    "ld1w { z28.s }, p1/Z, [x25, x20, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x26, x19, LSL #2]\n"
    "ld1w { z26.s }, p1/Z, [x25, x19, LSL #2]\n"
    "ld1w { z25.s }, p1/Z, [x24]\n"
    "ld1w { z24.s }, p1/Z, [x24, %x[in_col_stride], LSL #2]\n"
    "ld1w { z23.s }, p1/Z, [x24, x20, LSL #2]\n"
    "ld1w { z22.s }, p1/Z, [x24, x19, LSL #2]\n"
    "ld1w { z21.s }, p1/Z, [x23]\n"
    "ld1w { z20.s }, p1/Z, [x23, %x[in_col_stride], LSL #2]\n"
    "ld1w { z19.s }, p1/Z, [x23, x20, LSL #2]\n"
    "ld1w { z18.s }, p1/Z, [x23, x19, LSL #2]\n"
    "ld1rw { z17.s }, p2/Z, [%x[minmax_vals]]\n"
    "ld1rw { z16.s }, p2/Z, [%x[minmax_vals], #4]\n"
    "ble 2f\n"
    "1:"  // Loop
    "fmla z14.s, p2/M, z13.s, z3.s\n"
    "ld1w { z15.s }, p2/Z, [%x[params]]\n"
    "addvl x26, x26, #1\n"
    "fmla z12.s, p2/M, z13.s, z2.s\n"
    "addvl x25, x25, #1\n"
    "fmla z10.s, p2/M, z13.s, z1.s\n"
    "addvl x24, x24, #1\n"
    "fmla z8.s, p2/M, z13.s, z0.s\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "addvl x23, x23, #1\n"
    "fmla z14.s, p2/M, z11.s, z2.s\n"
    "decw %x[n_channels]\n"
    "mov p0.b, p1.b\n"
    "fmla z12.s, p2/M, z11.s, z29.s\n"
    "fmla z10.s, p2/M, z11.s, z0.s\n"
    "whilelt p1.s, XZR, %x[n_channels]\n"
    "ld1w { z3.s }, p1/Z, [x26]\n"
    "fmla z8.s, p2/M, z11.s, z28.s\n"
    "cmp %x[n_channels], XZR\n"
    "fmla z14.s, p2/M, z9.s, z29.s\n"
    "ld1w { z11.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "ld1w { z2.s }, p1/Z, [x26, %x[in_col_stride], LSL #2]\n"
    "fmla z12.s, p2/M, z9.s, z27.s\n"
    "fmla z10.s, p2/M, z9.s, z28.s\n"
    "ld1w { z29.s }, p1/Z, [x26, x20, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x26, x19, LSL #2]\n"
    "fmla z8.s, p2/M, z9.s, z26.s\n"
    "ld1w { z9.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "fmla z14.s, p2/M, z7.s, z1.s\n"
    "ld1w { z1.s }, p1/Z, [x25]\n"
    "fmla z12.s, p2/M, z7.s, z0.s\n"
    "fmla z10.s, p2/M, z7.s, z25.s\n"
    "fmla z8.s, p2/M, z7.s, z24.s\n"
    "ld1w { z7.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "fmla z14.s, p2/M, z6.s, z0.s\n"
    "ld1w { z0.s }, p1/Z, [x25, %x[in_col_stride], LSL #2]\n"
    "fmla z12.s, p2/M, z6.s, z28.s\n"
    "fmla z10.s, p2/M, z6.s, z24.s\n"
    "fmla z8.s, p2/M, z6.s, z23.s\n"
    "ld1w { z6.s }, p2/Z, [%x[params], #5, MUL VL]\n"
    "fmla z14.s, p2/M, z5.s, z28.s\n"
    "ld1w { z28.s }, p1/Z, [x25, x20, LSL #2]\n"
    "fmla z12.s, p2/M, z5.s, z26.s\n"
    "ld1w { z26.s }, p1/Z, [x25, x19, LSL #2]\n"
    "fmla z10.s, p2/M, z5.s, z23.s\n"
    "fmla z8.s, p2/M, z5.s, z22.s\n"
    "ld1w { z5.s }, p2/Z, [%x[params], #6, MUL VL]\n"
    "fmla z14.s, p2/M, z4.s, z25.s\n"
    "ld1w { z25.s }, p1/Z, [x24]\n"
    "fmla z12.s, p2/M, z4.s, z24.s\n"
    "fmla z10.s, p2/M, z4.s, z21.s\n"
    "ld1w { z21.s }, p1/Z, [x23]\n"
    "fmla z8.s, p2/M, z4.s, z20.s\n"
    "ld1w { z4.s }, p2/Z, [%x[params], #7, MUL VL]\n"
    "addvl %x[params], %x[params], #16\n"
    "fmla z14.s, p2/M, z31.s, z24.s\n"
    "ld1w { z24.s }, p1/Z, [x24, %x[in_col_stride], LSL #2]\n"
    "fmla z12.s, p2/M, z31.s, z23.s\n"
    "fmla z10.s, p2/M, z31.s, z20.s\n"
    "ld1w { z20.s }, p1/Z, [x23, %x[in_col_stride], LSL #2]\n"
    "fmla z8.s, p2/M, z31.s, z19.s\n"
    "ld1w { z31.s }, p2/Z, [%x[params], #-8, MUL VL]\n"
    "fmla z14.s, p2/M, z30.s, z23.s\n"
    "ld1w { z23.s }, p1/Z, [x24, x20, LSL #2]\n"
    "fmla z12.s, p2/M, z30.s, z22.s\n"
    "ld1w { z22.s }, p1/Z, [x24, x19, LSL #2]\n"
    "fmla z10.s, p2/M, z30.s, z19.s\n"
    "ld1w { z19.s }, p1/Z, [x23, x20, LSL #2]\n"
    "fmla z8.s, p2/M, z30.s, z18.s\n"
    "ld1w { z30.s }, p2/Z, [%x[params], #-7, MUL VL]\n"
    "addvl %x[params], %x[params], #-6\n"
    "fmax z14.s, p2/M, z14.s, z17.s\n"
    "ld1w { z18.s }, p1/Z, [x23, x19, LSL #2]\n"
    "fmax z12.s, p2/M, z12.s, z17.s\n"
    "fmax z10.s, p2/M, z10.s, z17.s\n"
    "fmax z8.s, p2/M, z8.s, z17.s\n"
    "fmin z14.s, p2/M, z14.s, z16.s\n"
    "st1w { z14.s }, p0, [x22]\n"
    "mov z14.d, z15.d\n"
    "fmin z12.s, p2/M, z12.s, z16.s\n"
    "st1w { z12.s }, p0, [x22, %x[out_col_stride], LSL #2]\n"
    "mov z12.d, z15.d\n"
    "addvl x22, x22, #1\n"
    "fmin z10.s, p2/M, z10.s, z16.s\n"
    "st1w { z10.s }, p0, [x21]\n"
    "mov z10.d, z15.d\n"
    "fmin z8.s, p2/M, z8.s, z16.s\n"
    "st1w { z8.s }, p0, [x21, %x[out_col_stride], LSL #2]\n"
    "mov z8.d, z15.d\n"
    "addvl x21, x21, #1\n"
    "bgt 1b\n"
    "2:"  // Tail
    "fmla z14.s, p2/M, z13.s, z3.s\n"
    "mov p0.b, p1.b\n"
    "fmla z12.s, p2/M, z13.s, z2.s\n"
    "fmla z10.s, p2/M, z13.s, z1.s\n"
    "fmla z8.s, p2/M, z13.s, z0.s\n"
    "fmla z14.s, p2/M, z11.s, z2.s\n"
    "fmla z12.s, p2/M, z11.s, z29.s\n"
    "fmla z10.s, p2/M, z11.s, z0.s\n"
    "fmla z8.s, p2/M, z11.s, z28.s\n"
    "fmla z14.s, p2/M, z9.s, z29.s\n"
    "fmla z12.s, p2/M, z9.s, z27.s\n"
    "fmla z10.s, p2/M, z9.s, z28.s\n"
    "fmla z8.s, p2/M, z9.s, z26.s\n"
    "fmla z14.s, p2/M, z7.s, z1.s\n"
    "fmla z12.s, p2/M, z7.s, z0.s\n"
    "fmla z10.s, p2/M, z7.s, z25.s\n"
    "fmla z8.s, p2/M, z7.s, z24.s\n"
    "fmla z14.s, p2/M, z6.s, z0.s\n"
    "fmla z12.s, p2/M, z6.s, z28.s\n"
    "fmla z10.s, p2/M, z6.s, z24.s\n"
    "fmla z8.s, p2/M, z6.s, z23.s\n"
    "fmla z14.s, p2/M, z5.s, z28.s\n"
    "fmla z12.s, p2/M, z5.s, z26.s\n"
    "fmla z10.s, p2/M, z5.s, z23.s\n"
    "fmla z8.s, p2/M, z5.s, z22.s\n"
    "fmla z14.s, p2/M, z4.s, z25.s\n"
    "fmla z12.s, p2/M, z4.s, z24.s\n"
    "fmla z10.s, p2/M, z4.s, z21.s\n"
    "fmla z8.s, p2/M, z4.s, z20.s\n"
    "fmla z14.s, p2/M, z31.s, z24.s\n"
    "fmla z12.s, p2/M, z31.s, z23.s\n"
    "fmla z10.s, p2/M, z31.s, z20.s\n"
    "fmla z8.s, p2/M, z31.s, z19.s\n"
    "fmla z14.s, p2/M, z30.s, z23.s\n"
    "fmla z12.s, p2/M, z30.s, z22.s\n"
    "fmla z10.s, p2/M, z30.s, z19.s\n"
    "fmla z8.s, p2/M, z30.s, z18.s\n"
    "fmax z14.s, p2/M, z14.s, z17.s\n"
    "fmax z12.s, p2/M, z12.s, z17.s\n"
    "fmax z10.s, p2/M, z10.s, z17.s\n"
    "fmax z8.s, p2/M, z8.s, z17.s\n"
    "fmin z14.s, p2/M, z14.s, z16.s\n"
    "st1w { z14.s }, p0, [x22]\n"
    "fmin z12.s, p2/M, z12.s, z16.s\n"
    "fmin z10.s, p2/M, z10.s, z16.s\n"
    "st1w { z12.s }, p0, [x22, %x[out_col_stride], LSL #2]\n"
    "fmin z8.s, p2/M, z8.s, z16.s\n"
    "st1w { z10.s }, p0, [x21]\n"
    "st1w { z8.s }, p0, [x21, %x[out_col_stride], LSL #2]\n"
    : [n_channels] "+r" (n_channels), [params] "+r" (params)
    : [in_col_stride] "r" (in_col_stride), [in_row_stride] "r" (in_row_stride), [inptr] "r" (inptr), [minmax_vals] "r" (minmax_vals), [out_col_stride] "r" (out_col_stride), [out_row_stride] "r" (out_row_stride), [outptr] "r" (outptr)
    : "cc", "memory", "p0", "p1", "p2", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
