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

void sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const float *const *const input_ptrs,
  float *const *const outptrs,
  const void *params,
  unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  const float *const inptrs[16] = {
    input_ptrs[0], input_ptrs[1], input_ptrs[4], input_ptrs[5], input_ptrs[2], input_ptrs[6], input_ptrs[3], input_ptrs[7], input_ptrs[8], input_ptrs[9], input_ptrs[10], input_ptrs[11], input_ptrs[12], input_ptrs[13], input_ptrs[14], input_ptrs[15],
  };
  const float minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ldp x26, x23, [%x[inptrs], #0x0]\n"
    "ptrue p2.b\n"
    "ldp x25, x16, [%x[inptrs], #0x10]\n"
    "mov x15, #0x0\n"
    "ld1w { z15.s }, p2/Z, [%x[params]]\n"
    "mov z14.d, z15.d\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "cntw x14\n"
    "mov z12.d, z15.d\n"
    "ld1w { z11.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "sub x13, XZR, x14\n"
    "mov z10.d, z15.d\n"
    "ld1w { z9.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "whilelt p1.s, XZR, %x[n_channels]\n"
    "mov z8.d, z15.d\n"
    "ld1w { z7.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "cmp x14, %x[n_channels]\n"
    "ld1w { z6.s }, p2/Z, [%x[params], #5, MUL VL]\n"
    "ld1w { z5.s }, p2/Z, [%x[params], #6, MUL VL]\n"
    "ld1w { z4.s }, p2/Z, [%x[params], #7, MUL VL]\n"
    "addvl %x[params], %x[params], #16\n"
    "ld1w { z3.s }, p1/Z, [x26, x15, LSL #2]\n"
    "ld1w { z2.s }, p2/Z, [%x[params], #-8, MUL VL]\n"
    "ld1w { z1.s }, p2/Z, [%x[params], #-7, MUL VL]\n"
    "addvl %x[params], %x[params], #-6\n"
    "ld1w { z0.s }, p1/Z, [x23, x15, LSL #2]\n"
    "ld1w { z31.s }, p1/Z, [x25, x15, LSL #2]\n"
    "ld1w { z30.s }, p1/Z, [x16, x15, LSL #2]\n"
    "ldp x24, x12, [%x[inptrs], #0x20]\n"
    "ldp x23, x11, [%x[inptrs], #0x30]\n"
    "ldp x10, x9, [%x[inptrs], #0x40]\n"
    "ld1w { z29.s }, p1/Z, [x24, x15, LSL #2]\n"
    "ld1w { z28.s }, p1/Z, [x12, x15, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x23, x15, LSL #2]\n"
    "ld1w { z26.s }, p1/Z, [x11, x15, LSL #2]\n"
    "ld1w { z25.s }, p1/Z, [x10, x15, LSL #2]\n"
    "ld1w { z24.s }, p1/Z, [x9, x15, LSL #2]\n"
    "ldp x28, x27, [%x[inptrs], #0x50]\n"
    "ldp x26, x25, [%x[inptrs], #0x60]\n"
    "ldp x24, x23, [%x[inptrs], #0x70]\n"
    "ld1w { z23.s }, p1/Z, [x28, x15, LSL #2]\n"
    "ld1w { z22.s }, p1/Z, [x27, x15, LSL #2]\n"
    "ld1w { z21.s }, p1/Z, [x26, x15, LSL #2]\n"
    "ld1w { z20.s }, p1/Z, [x25, x15, LSL #2]\n"
    "ld1w { z19.s }, p1/Z, [x24, x15, LSL #2]\n"
    "ld1w { z18.s }, p1/Z, [x23, x15, LSL #2]\n"
    "ldp x22, x21, [%x[outptrs], #0x0]\n"
    "ldp x20, x19, [%x[outptrs], #0x10]\n"
    "ld1rw { z17.s }, p2/Z, [%x[minmax_vals]]\n"
    "ld1rw { z16.s }, p2/Z, [%x[minmax_vals], #4]\n"
    "bge 1f\n"
    "1:"  // Loop
    "fmla z14.s, p2/M, z13.s, z3.s\n"
    "ld1w { z15.s }, p2/Z, [%x[params]]\n"
    "incw x13\n"
    "fmla z12.s, p2/M, z13.s, z0.s\n"
    "ldp x26, x23, [%x[inptrs], #0x0]\n"
    "mov p0.b, p1.b\n"
    "fmla z10.s, p2/M, z13.s, z31.s\n"
    "ldp x25, x16, [%x[inptrs], #0x10]\n"
    "mov x15, x14\n"
    "fmla z8.s, p2/M, z13.s, z30.s\n"
    "ld1w { z13.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "incw x14\n"
    "fmla z14.s, p2/M, z11.s, z0.s\n"
    "ldp x24, x12, [%x[inptrs], #0x20]\n"
    "whilelt p1.s, x15, %x[n_channels]\n"
    "fmla z12.s, p2/M, z11.s, z29.s\n"
    "ld1w { z3.s }, p1/Z, [x26, x15, LSL #2]\n"
    "cmp x14, %x[n_channels]\n"
    "fmla z10.s, p2/M, z11.s, z30.s\n"
    "ld1w { z0.s }, p1/Z, [x23, x15, LSL #2]\n"
    "ldp x23, x11, [%x[inptrs], #0x30]\n"
    "fmla z8.s, p2/M, z11.s, z28.s\n"
    "ld1w { z11.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "fmla z14.s, p2/M, z9.s, z29.s\n"
    "ld1w { z29.s }, p1/Z, [x24, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z9.s, z27.s\n"
    "ld1w { z27.s }, p1/Z, [x23, x15, LSL #2]\n"
    "fmla z10.s, p2/M, z9.s, z28.s\n"
    "ldp x10, x9, [%x[inptrs], #0x40]\n"
    "fmla z8.s, p2/M, z9.s, z26.s\n"
    "ld1w { z9.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "fmla z14.s, p2/M, z7.s, z31.s\n"
    "ld1w { z31.s }, p1/Z, [x25, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z7.s, z30.s\n"
    "ldp x28, x27, [%x[inptrs], #0x50]\n"
    "fmla z10.s, p2/M, z7.s, z25.s\n"
    "ldp x26, x25, [%x[inptrs], #0x60]\n"
    "fmla z8.s, p2/M, z7.s, z24.s\n"
    "ld1w { z7.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "fmla z14.s, p2/M, z6.s, z30.s\n"
    "ld1w { z30.s }, p1/Z, [x16, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z6.s, z28.s\n"
    "ldp x24, x23, [%x[inptrs], #0x70]\n"
    "fmla z10.s, p2/M, z6.s, z24.s\n"
    "fmla z8.s, p2/M, z6.s, z23.s\n"
    "ld1w { z6.s }, p2/Z, [%x[params], #5, MUL VL]\n"
    "fmla z14.s, p2/M, z5.s, z28.s\n"
    "ld1w { z28.s }, p1/Z, [x12, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z5.s, z26.s\n"
    "ld1w { z26.s }, p1/Z, [x11, x15, LSL #2]\n"
    "fmla z10.s, p2/M, z5.s, z23.s\n"
    "fmla z8.s, p2/M, z5.s, z22.s\n"
    "ld1w { z5.s }, p2/Z, [%x[params], #6, MUL VL]\n"
    "fmla z14.s, p2/M, z4.s, z25.s\n"
    "ld1w { z25.s }, p1/Z, [x10, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z4.s, z24.s\n"
    "fmla z10.s, p2/M, z4.s, z21.s\n"
    "ld1w { z21.s }, p1/Z, [x26, x15, LSL #2]\n"
    "fmla z8.s, p2/M, z4.s, z20.s\n"
    "ld1w { z4.s }, p2/Z, [%x[params], #7, MUL VL]\n"
    "addvl %x[params], %x[params], #16\n"
    "fmla z14.s, p2/M, z2.s, z24.s\n"
    "ld1w { z24.s }, p1/Z, [x9, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z2.s, z23.s\n"
    "fmla z10.s, p2/M, z2.s, z20.s\n"
    "ld1w { z20.s }, p1/Z, [x25, x15, LSL #2]\n"
    "fmla z8.s, p2/M, z2.s, z19.s\n"
    "ld1w { z2.s }, p2/Z, [%x[params], #-8, MUL VL]\n"
    "fmla z14.s, p2/M, z1.s, z23.s\n"
    "ld1w { z23.s }, p1/Z, [x28, x15, LSL #2]\n"
    "fmla z12.s, p2/M, z1.s, z22.s\n"
    "ld1w { z22.s }, p1/Z, [x27, x15, LSL #2]\n"
    "fmla z10.s, p2/M, z1.s, z19.s\n"
    "ld1w { z19.s }, p1/Z, [x24, x15, LSL #2]\n"
    "fmla z8.s, p2/M, z1.s, z18.s\n"
    "ld1w { z1.s }, p2/Z, [%x[params], #-7, MUL VL]\n"
    "addvl %x[params], %x[params], #-6\n"
    "fmax z14.s, p2/M, z14.s, z17.s\n"
    "ld1w { z18.s }, p1/Z, [x23, x15, LSL #2]\n"
    "fmax z12.s, p2/M, z12.s, z17.s\n"
    "fmax z10.s, p2/M, z10.s, z17.s\n"
    "fmax z8.s, p2/M, z8.s, z17.s\n"
    "fmin z14.s, p2/M, z14.s, z16.s\n"
    "st1w { z14.s }, p0, [x22, x13, LSL #2]\n"
    "mov z14.d, z15.d\n"
    "fmin z12.s, p2/M, z12.s, z16.s\n"
    "st1w { z12.s }, p0, [x21, x13, LSL #2]\n"
    "mov z12.d, z15.d\n"
    "fmin z10.s, p2/M, z10.s, z16.s\n"
    "st1w { z10.s }, p0, [x20, x13, LSL #2]\n"
    "mov z10.d, z15.d\n"
    "fmin z8.s, p2/M, z8.s, z16.s\n"
    "st1w { z8.s }, p0, [x19, x13, LSL #2]\n"
    "mov z8.d, z15.d\n"
    "blt 1b\n"
    "2:"  // Tail
    "fmla z14.s, p2/M, z13.s, z3.s\n"
    "incw x13\n"
    "fmla z12.s, p2/M, z13.s, z0.s\n"
    "mov p0.b, p1.b\n"
    "fmla z10.s, p2/M, z13.s, z31.s\n"
    "fmla z8.s, p2/M, z13.s, z30.s\n"
    "fmla z14.s, p2/M, z11.s, z0.s\n"
    "fmla z12.s, p2/M, z11.s, z29.s\n"
    "fmla z10.s, p2/M, z11.s, z30.s\n"
    "fmla z8.s, p2/M, z11.s, z28.s\n"
    "fmla z14.s, p2/M, z9.s, z29.s\n"
    "fmla z12.s, p2/M, z9.s, z27.s\n"
    "fmla z10.s, p2/M, z9.s, z28.s\n"
    "fmla z8.s, p2/M, z9.s, z26.s\n"
    "fmla z14.s, p2/M, z7.s, z31.s\n"
    "fmla z12.s, p2/M, z7.s, z30.s\n"
    "fmla z10.s, p2/M, z7.s, z25.s\n"
    "fmla z8.s, p2/M, z7.s, z24.s\n"
    "fmla z14.s, p2/M, z6.s, z30.s\n"
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
    "fmla z14.s, p2/M, z2.s, z24.s\n"
    "fmla z12.s, p2/M, z2.s, z23.s\n"
    "fmla z10.s, p2/M, z2.s, z20.s\n"
    "fmla z8.s, p2/M, z2.s, z19.s\n"
    "fmla z14.s, p2/M, z1.s, z23.s\n"
    "fmla z12.s, p2/M, z1.s, z22.s\n"
    "fmla z10.s, p2/M, z1.s, z19.s\n"
    "fmla z8.s, p2/M, z1.s, z18.s\n"
    "fmax z14.s, p2/M, z14.s, z17.s\n"
    "fmax z12.s, p2/M, z12.s, z17.s\n"
    "fmax z10.s, p2/M, z10.s, z17.s\n"
    "fmax z8.s, p2/M, z8.s, z17.s\n"
    "fmin z14.s, p2/M, z14.s, z16.s\n"
    "st1w { z14.s }, p0, [x22, x13, LSL #2]\n"
    "fmin z12.s, p2/M, z12.s, z16.s\n"
    "fmin z10.s, p2/M, z10.s, z16.s\n"
    "st1w { z12.s }, p0, [x21, x13, LSL #2]\n"
    "fmin z8.s, p2/M, z8.s, z16.s\n"
    "st1w { z10.s }, p0, [x20, x13, LSL #2]\n"
    "st1w { z8.s }, p0, [x19, x13, LSL #2]\n"
    : [params] "+r" (params)
    : [inptrs] "r" (inptrs), [minmax_vals] "r" (minmax_vals), [n_channels] "r" ((unsigned long) n_channels), [outptrs] "r" (outptrs)
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
