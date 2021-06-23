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

void sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst_impl(
  const float *const *const inptrs,
  float *const *const outptrs,
  const void *params,
  const unsigned int n_output_channels,
  const float activation_min,
  const float activation_max
)
{
  const float minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ldp x12, x11, [%x[outptrs], #0x0]\n"
    "ptrue p2.b\n"
    "ldp x10, x9, [%x[outptrs], #0x10]\n"
    "mov x28, #0x0\n"
    "ldp x27, x26, [%x[outptrs], #0x20]\n"
    "mov x25, #0x0\n"
    "ldp x24, x23, [%x[outptrs], #0x30]\n"
    "whilelt p1.s, x28, %x[channel_multiplier]\n"
    "ldr x22, [%x[outptrs], #0x40]\n"
    "ldr x21, [%x[inptrs], #0x0]\n"
    "ldr x20, [%x[inptrs], #0x8]\n"
    "ldr x19, [%x[inptrs], #0x10]\n"
    "ld1rqw { z2.s }, p2/Z, [x21]\n"
    "ld1rqw { z3.s }, p2/Z, [x21, #16]\n"
    "ld1rqw { z4.s }, p2/Z, [x20]\n"
    "ld1rqw { z5.s }, p2/Z, [x20, #16]\n"
    "ld1rqw { z6.s }, p2/Z, [x19]\n"
    "ld1rqw { z7.s }, p2/Z, [x19, #16]\n"
    "ldr x21, [%x[inptrs], #0x18]\n"
    "ldr x20, [%x[inptrs], #0x20]\n"
    "ldr x19, [%x[inptrs], #0x28]\n"
    "ld1rqw { z8.s }, p2/Z, [x21]\n"
    "ld1rqw { z9.s }, p2/Z, [x21, #16]\n"
    "ld1rqw { z10.s }, p2/Z, [x20]\n"
    "ld1rqw { z11.s }, p2/Z, [x20, #16]\n"
    "ld1rqw { z12.s }, p2/Z, [x19]\n"
    "ld1rqw { z13.s }, p2/Z, [x19, #16]\n"
    "ldr x19, [%x[inptrs], #0x30]\n"
    "ld1rw { z26.s }, p2/Z, [%x[clamps]]\n"
    "ld1rw { z25.s }, p2/Z, [%x[clamps], #4]\n"
    "ld1rqw { z14.s }, p2/Z, [x19]\n"
    "ld1rqw { z15.s }, p2/Z, [x19, #16]\n"
    "ld1w { z24.s }, p1/Z, [%x[params]]\n"
    "mov z23.d, z24.d\n"
    "ld1w { z31.s }, p1/Z, [%x[params], #1, MUL VL]\n"
    "mov z22.d, z24.d\n"
    "ld1w { z30.s }, p1/Z, [%x[params], #2, MUL VL]\n"
    "mov z21.d, z24.d\n"
    "ld1w { z29.s }, p1/Z, [%x[params], #3, MUL VL]\n"
    "addvl %x[params], %x[params], #4\n"
    "mov z20.d, z24.d\n"
    "mov z19.d, z24.d\n"
    "mov z18.d, z24.d\n"
    "mov z17.d, z24.d\n"
    "mov z16.d, z24.d\n"
    "1:"  // Output channel complete vector loop
    "mov z0.d, z10.d\n"
    "mov p0.b, p1.b\n"
    "mov z1.d, z11.d\n"
    "incw x28\n"
    "fmla z24.s, z31.s, z2.s[0]\n"
    "whilelt p1.s, x28, %x[channel_multiplier]\n"
    "fmla z23.s, z31.s, z2.s[2]\n"
    "fmla z22.s, z31.s, z3.s[0]\n"
    "fmla z21.s, z31.s, z6.s[0]\n"
    "fmla z20.s, z31.s, z6.s[2]\n"
    "fmla z19.s, z31.s, z7.s[0]\n"
    "fmla z18.s, z31.s, z0.s[0]\n"
    "fmla z17.s, z31.s, z0.s[2]\n"
    "fmla z16.s, z31.s, z1.s[0]\n"
    "ld1w { z31.s }, p2/Z, [%x[params]]\n"
    "fmla z24.s, z30.s, z2.s[1]\n"
    "fmla z23.s, z30.s, z2.s[3]\n"
    "fmla z22.s, z30.s, z3.s[1]\n"
    "fmla z21.s, z30.s, z6.s[1]\n"
    "fmla z20.s, z30.s, z6.s[3]\n"
    "fmla z19.s, z30.s, z7.s[1]\n"
    "fmla z18.s, z30.s, z0.s[1]\n"
    "fmla z17.s, z30.s, z0.s[3]\n"
    "fmla z16.s, z30.s, z1.s[1]\n"
    "ld1w { z30.s }, p2/Z, [%x[params], #1, MUL VL]\n"
    "fmla z24.s, z29.s, z2.s[2]\n"
    "fmla z23.s, z29.s, z3.s[0]\n"
    "fmla z22.s, z29.s, z3.s[2]\n"
    "fmla z21.s, z29.s, z6.s[2]\n"
    "fmla z20.s, z29.s, z7.s[0]\n"
    "fmla z19.s, z29.s, z7.s[2]\n"
    "fmla z18.s, z29.s, z0.s[2]\n"
    "mov z0.d, z8.d\n"
    "fmla z17.s, z29.s, z1.s[0]\n"
    "fmla z16.s, z29.s, z1.s[2]\n"
    "ld1w { z29.s }, p2/Z, [%x[params], #2, MUL VL]\n"
    "mov z1.d, z9.d\n"
    "fmla z24.s, z31.s, z4.s[0]\n"
    "fmla z23.s, z31.s, z4.s[2]\n"
    "fmla z22.s, z31.s, z5.s[0]\n"
    "fmla z21.s, z31.s, z0.s[0]\n"
    "fmla z20.s, z31.s, z0.s[2]\n"
    "mov z0.d, z12.d\n"
    "fmla z19.s, z31.s, z1.s[0]\n"
    "mov z1.d, z13.d\n"
    "fmla z18.s, z31.s, z0.s[0]\n"
    "fmla z17.s, z31.s, z0.s[2]\n"
    "mov z0.d, z8.d\n"
    "fmla z16.s, z31.s, z1.s[0]\n"
    "ld1w { z31.s }, p2/Z, [%x[params], #3, MUL VL]\n"
    "mov z1.d, z9.d\n"
    "fmla z24.s, z30.s, z4.s[1]\n"
    "fmla z23.s, z30.s, z4.s[3]\n"
    "fmla z22.s, z30.s, z5.s[1]\n"
    "fmla z21.s, z30.s, z0.s[1]\n"
    "fmla z20.s, z30.s, z0.s[3]\n"
    "mov z0.d, z12.d\n"
    "fmla z19.s, z30.s, z1.s[1]\n"
    "mov z1.d, z13.d\n"
    "fmla z18.s, z30.s, z0.s[1]\n"
    "fmla z17.s, z30.s, z0.s[3]\n"
    "mov z0.d, z8.d\n"
    "fmla z16.s, z30.s, z1.s[1]\n"
    "ld1w { z30.s }, p2/Z, [%x[params], #4, MUL VL]\n"
    "mov z1.d, z9.d\n"
    "fmla z24.s, z29.s, z4.s[2]\n"
    "fmla z23.s, z29.s, z5.s[0]\n"
    "fmla z22.s, z29.s, z5.s[2]\n"
    "fmla z21.s, z29.s, z0.s[2]\n"
    "mov z0.d, z12.d\n"
    "fmla z20.s, z29.s, z1.s[0]\n"
    "fmla z19.s, z29.s, z1.s[2]\n"
    "mov z1.d, z13.d\n"
    "fmla z18.s, z29.s, z0.s[2]\n"
    "mov z0.d, z10.d\n"
    "fmla z17.s, z29.s, z1.s[0]\n"
    "fmla z16.s, z29.s, z1.s[2]\n"
    "ld1w { z29.s }, p2/Z, [%x[params], #5, MUL VL]\n"
    "mov z1.d, z11.d\n"
    "fmla z24.s, z31.s, z6.s[0]\n"
    "fmla z23.s, z31.s, z6.s[2]\n"
    "fmla z22.s, z31.s, z7.s[0]\n"
    "fmla z21.s, z31.s, z0.s[0]\n"
    "fmla z20.s, z31.s, z0.s[2]\n"
    "mov z0.d, z14.d\n"
    "fmla z19.s, z31.s, z1.s[0]\n"
    "mov z1.d, z15.d\n"
    "fmla z18.s, z31.s, z0.s[0]\n"
    "fmla z17.s, z31.s, z0.s[2]\n"
    "mov z0.d, z10.d\n"
    "fmla z16.s, z31.s, z1.s[0]\n"
    "ld1w { z31.s }, p1/Z, [%x[params], #7, MUL VL]\n"
    "mov z1.d, z11.d\n"
    "fmla z24.s, z30.s, z6.s[1]\n"
    "fmla z23.s, z30.s, z6.s[3]\n"
    "fmla z22.s, z30.s, z7.s[1]\n"
    "fmla z21.s, z30.s, z0.s[1]\n"
    "fmla z20.s, z30.s, z0.s[3]\n"
    "mov z0.d, z14.d\n"
    "fmla z19.s, z30.s, z1.s[1]\n"
    "mov z1.d, z15.d\n"
    "fmla z18.s, z30.s, z0.s[1]\n"
    "fmla z17.s, z30.s, z0.s[3]\n"
    "mov z0.d, z10.d\n"
    "fmla z16.s, z30.s, z1.s[1]\n"
    "mov z1.d, z11.d\n"
    "fmla z24.s, z29.s, z6.s[2]\n"
    "fmla z23.s, z29.s, z7.s[0]\n"
    "fmla z22.s, z29.s, z7.s[2]\n"
    "fmla z21.s, z29.s, z0.s[2]\n"
    "mov z0.d, z14.d\n"
    "fmla z20.s, z29.s, z1.s[0]\n"
    "fmla z19.s, z29.s, z1.s[2]\n"
    "mov z1.d, z15.d\n"
    "fmla z18.s, z29.s, z0.s[2]\n"
    "fmla z17.s, z29.s, z1.s[0]\n"
    "fmla z16.s, z29.s, z1.s[2]\n"
    "fmin z24.s, p2/M, z24.s, z25.s\n"
    "fmin z23.s, p2/M, z23.s, z25.s\n"
    "fmin z22.s, p2/M, z22.s, z25.s\n"
    "fmin z21.s, p2/M, z21.s, z25.s\n"
    "fmax z24.s, p2/M, z24.s, z26.s\n"
    "st1w { z24.s }, p0, [x12, x25, LSL #2]\n"
    "fmax z23.s, p2/M, z23.s, z26.s\n"
    "fmax z22.s, p2/M, z22.s, z26.s\n"
    "ld1w { z24.s }, p1/Z, [%x[params], #6, MUL VL]\n"
    "addvl %x[params], %x[params], #16\n"
    "fmax z21.s, p2/M, z21.s, z26.s\n"
    "ld1w { z30.s }, p1/Z, [%x[params], #-8, MUL VL]\n"
    "fmin z20.s, p2/M, z20.s, z25.s\n"
    "ld1w { z29.s }, p1/Z, [%x[params], #-7, MUL VL]\n"
    "addvl %x[params], %x[params], #-6\n"
    "fmin z19.s, p2/M, z19.s, z25.s\n"
    "st1w { z23.s }, p0, [x11, x25, LSL #2]\n"
    "mov z23.d, z24.d\n"
    "st1w { z22.s }, p0, [x10, x25, LSL #2]\n"
    "mov z22.d, z24.d\n"
    "st1w { z21.s }, p0, [x9, x25, LSL #2]\n"
    "mov z21.d, z24.d\n"
    "fmax z20.s, p2/M, z20.s, z26.s\n"
    "st1w { z20.s }, p0, [x27, x25, LSL #2]\n"
    "mov z20.d, z24.d\n"
    "fmax z19.s, p2/M, z19.s, z26.s\n"
    "st1w { z19.s }, p0, [x26, x25, LSL #2]\n"
    "mov z19.d, z24.d\n"
    "fmin z18.s, p2/M, z18.s, z25.s\n"
    "fmin z17.s, p2/M, z17.s, z25.s\n"
    "fmin z16.s, p2/M, z16.s, z25.s\n"
    "fmax z18.s, p2/M, z18.s, z26.s\n"
    "st1w { z18.s }, p0, [x24, x25, LSL #2]\n"
    "mov z18.d, z24.d\n"
    "fmax z17.s, p2/M, z17.s, z26.s\n"
    "st1w { z17.s }, p0, [x23, x25, LSL #2]\n"
    "mov z17.d, z24.d\n"
    "fmax z16.s, p2/M, z16.s, z26.s\n"
    "st1w { z16.s }, p0, [x22, x25, LSL #2]\n"
    "mov z16.d, z24.d\n"
    "incw x25\n"
    "b.any 1b\n"
    : [params] "+&r" (params)
    : [channel_multiplier] "r" (n_output_channels), [clamps] "r" (minmax_vals), [inptrs] "r" (inptrs), [outptrs] "r" (outptrs)
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
