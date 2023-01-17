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

void sve_fp32_nhwc_generic_output9_mla_depthfirst_impl(
  const float *const *const inptrs,
  float *const *const outptrs,
  const void *params,
  const void *bias,
  const unsigned int n_points,
  const unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  const float minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ptrue p1.b\n"
    "ld1rw { z4.s }, p1/Z, [%x[minmax_vals]]\n"
    "mov x28, #0x0\n"
    "ld1rw { z3.s }, p1/Z, [%x[minmax_vals], #4]\n"
    "whilelt p0.s, x28, %x[n_channels]\n"
    "1:"  // Channel loop
    "mov z2.b, #0x0\n"
    "cbz %x[bias], 2f\n"
    "ld1w { z2.s }, p0/Z, [%x[bias], x28, LSL #2]\n"
    "2:"  // Channel loop: Load bias: Done
    "mov z1.d, z2.d\n"
    "ld1w { z0.s }, p1/Z, [%x[params]]\n"
    "mov x22, %x[inptrs]\n"
    "mov z31.d, z2.d\n"
    "ldp x20, x19, [x22], #0x10\n"
    "subs x21, %x[n_points], #0x1\n"
    "mov z30.d, z2.d\n"
    "ld1w { z29.s }, p0/Z, [x20, x28, LSL #2]\n"
    "mov z28.d, z2.d\n"
    "addvl %x[params], %x[params], #1\n"
    "mov z27.d, z2.d\n"
    "ld1w { z26.s }, p0/Z, [x19, x28, LSL #2]\n"
    "mov z25.d, z2.d\n"
    "ldp x20, x19, [x22], #0x10\n"
    "mov z24.d, z2.d\n"
    "ld1w { z23.s }, p0/Z, [x20, x28, LSL #2]\n"
    "mov z22.d, z2.d\n"
    "ld1w { z21.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ldp x20, x19, [x22], #0x10\n"
    "ld1w { z20.s }, p0/Z, [x20, x28, LSL #2]\n"
    "ld1w { z19.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ldp x20, x19, [x22], #0x10\n"
    "ld1w { z18.s }, p0/Z, [x20, x28, LSL #2]\n"
    "ld1w { z17.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ldr x19, [x22], #0x8\n"
    "ld1w { z16.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ble 4f\n"
    "3:"  // Channel loop: Planar loop
    "fmla z2.s, p1/M, z29.s, z0.s\n"
    "ldp x20, x19, [x22], #0x10\n"
    "subs x21, x21, #0x1\n"
    "fmla z1.s, p1/M, z26.s, z0.s\n"
    "ld1w { z29.s }, p0/Z, [x20, x28, LSL #2]\n"
    "fmla z31.s, p1/M, z23.s, z0.s\n"
    "fmla z30.s, p1/M, z21.s, z0.s\n"
    "ld1w { z26.s }, p0/Z, [x19, x28, LSL #2]\n"
    "fmla z28.s, p1/M, z20.s, z0.s\n"
    "ldp x20, x19, [x22], #0x10\n"
    "fmla z27.s, p1/M, z19.s, z0.s\n"
    "ld1w { z23.s }, p0/Z, [x20, x28, LSL #2]\n"
    "fmla z25.s, p1/M, z18.s, z0.s\n"
    "fmla z24.s, p1/M, z17.s, z0.s\n"
    "ld1w { z21.s }, p0/Z, [x19, x28, LSL #2]\n"
    "fmla z22.s, p1/M, z16.s, z0.s\n"
    "ld1w { z0.s }, p1/Z, [%x[params]]\n"
    "addvl %x[params], %x[params], #1\n"
    "ldp x20, x19, [x22], #0x10\n"
    "ld1w { z20.s }, p0/Z, [x20, x28, LSL #2]\n"
    "ld1w { z19.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ldp x20, x19, [x22], #0x10\n"
    "ld1w { z18.s }, p0/Z, [x20, x28, LSL #2]\n"
    "ld1w { z17.s }, p0/Z, [x19, x28, LSL #2]\n"
    "ldr x19, [x22], #0x8\n"
    "ld1w { z16.s }, p0/Z, [x19, x28, LSL #2]\n"
    "bgt 3b\n"
    "4:"  // Channel loop: Planar tail
    "fmla z2.s, p1/M, z29.s, z0.s\n"
    "ldp x27, x26, [%x[outptrs], #0x0]\n"
    "fmla z1.s, p1/M, z26.s, z0.s\n"
    "ldp x25, x24, [%x[outptrs], #0x10]\n"
    "fmla z31.s, p1/M, z23.s, z0.s\n"
    "ldp x23, x22, [%x[outptrs], #0x20]\n"
    "fmla z30.s, p1/M, z21.s, z0.s\n"
    "ldp x21, x20, [%x[outptrs], #0x30]\n"
    "fmla z28.s, p1/M, z20.s, z0.s\n"
    "ldr x19, [%x[outptrs], #0x40]\n"
    "fmla z27.s, p1/M, z19.s, z0.s\n"
    "fmla z25.s, p1/M, z18.s, z0.s\n"
    "fmla z24.s, p1/M, z17.s, z0.s\n"
    "fmla z22.s, p1/M, z16.s, z0.s\n"
    "fmax z2.s, p1/M, z2.s, z4.s\n"
    "fmax z1.s, p1/M, z1.s, z4.s\n"
    "fmax z31.s, p1/M, z31.s, z4.s\n"
    "fmax z30.s, p1/M, z30.s, z4.s\n"
    "fmin z2.s, p1/M, z2.s, z3.s\n"
    "st1w { z2.s }, p0, [x27, x28, LSL #2]\n"
    "fmin z1.s, p1/M, z1.s, z3.s\n"
    "fmin z31.s, p1/M, z31.s, z3.s\n"
    "st1w { z1.s }, p0, [x26, x28, LSL #2]\n"
    "fmin z30.s, p1/M, z30.s, z3.s\n"
    "fmax z28.s, p1/M, z28.s, z4.s\n"
    "st1w { z31.s }, p0, [x25, x28, LSL #2]\n"
    "fmax z27.s, p1/M, z27.s, z4.s\n"
    "st1w { z30.s }, p0, [x24, x28, LSL #2]\n"
    "fmin z28.s, p1/M, z28.s, z3.s\n"
    "fmax z25.s, p1/M, z25.s, z4.s\n"
    "st1w { z28.s }, p0, [x23, x28, LSL #2]\n"
    "fmin z27.s, p1/M, z27.s, z3.s\n"
    "fmin z25.s, p1/M, z25.s, z3.s\n"
    "st1w { z27.s }, p0, [x22, x28, LSL #2]\n"
    "fmax z24.s, p1/M, z24.s, z4.s\n"
    "fmax z22.s, p1/M, z22.s, z4.s\n"
    "st1w { z25.s }, p0, [x21, x28, LSL #2]\n"
    "fmin z24.s, p1/M, z24.s, z3.s\n"
    "st1w { z24.s }, p0, [x20, x28, LSL #2]\n"
    "fmin z22.s, p1/M, z22.s, z3.s\n"
    "st1w { z22.s }, p0, [x19, x28, LSL #2]\n"
    "incw x28\n"
    "whilelt p0.s, x28, %x[n_channels]\n"
    "b.any 1b\n"
    : [params] "+&r" (params)
    : [bias] "r" (bias), [inptrs] "r" (inptrs), [minmax_vals] "r" (minmax_vals), [n_channels] "r" ((uint64_t) n_channels), [n_points] "r" ((uint64_t) n_points), [outptrs] "r" (outptrs)
    : "cc", "memory", "p0", "p1", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
