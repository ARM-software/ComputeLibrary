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
    "mov x11, #0x0\n"
    "ld1rw { z2.s }, p1/Z, [%x[minmax_vals]]\n"
    "ld1rw { z1.s }, p1/Z, [%x[minmax_vals], #4]\n"
    "whilelt p0.s, x11, %x[n_channels]\n"
    "1:"  // Channel loop
    "mov z23.b, #0x0\n"
    "cbz %x[bias], 2f\n"
    "ld1w { z23.s }, p0/Z, [%x[bias], x11, LSL #2]\n"
    "2:"  // Channel loop: Load bias: Done
    "mov x10, %x[inptrs]\n"
    "ldp x9, x28, [x10], #0x10\n"
    "ldp x27, x26, [x10], #0x10\n"
    "subs x25, %x[n_points], #0x1\n"
    "ldp x24, x23, [x10], #0x10\n"
    "ldp x22, x21, [x10], #0x10\n"
    "mov z24.d, z23.d\n"
    "mov z25.d, z23.d\n"
    "ldr x20, [x10], #0x8\n"
    "mov z26.d, z23.d\n"
    "mov z27.d, z23.d\n"
    "ld1w { z0.s }, p1/Z, [%x[params]]\n"
    "mov z28.d, z23.d\n"
    "mov z29.d, z23.d\n"
    "ld1w { z14.s }, p0/Z, [x9, x11, LSL #2]\n"
    "ld1w { z15.s }, p0/Z, [x28, x11, LSL #2]\n"
    "mov z30.d, z23.d\n"
    "mov z31.d, z23.d\n"
    "ld1w { z16.s }, p0/Z, [x27, x11, LSL #2]\n"
    "ld1w { z17.s }, p0/Z, [x26, x11, LSL #2]\n"
    "ld1w { z18.s }, p0/Z, [x24, x11, LSL #2]\n"
    "ld1w { z19.s }, p0/Z, [x23, x11, LSL #2]\n"
    "addvl %x[params], %x[params], #1\n"
    "ld1w { z20.s }, p0/Z, [x22, x11, LSL #2]\n"
    "ld1w { z21.s }, p0/Z, [x21, x11, LSL #2]\n"
    "ld1w { z22.s }, p0/Z, [x20, x11, LSL #2]\n"
    "ble 4f\n"
    "3:"  // Channel loop: Planar loop
    "ldp x9, x28, [x10], #0x10\n"
    "ldp x27, x26, [x10], #0x10\n"
    "subs x25, x25, #0x1\n"
    "fmla z23.s, p1/M, z14.s, z0.s\n"
    "ldp x24, x23, [x10], #0x10\n"
    "ldp x22, x21, [x10], #0x10\n"
    "fmla z24.s, p1/M, z15.s, z0.s\n"
    "fmla z25.s, p1/M, z16.s, z0.s\n"
    "ldr x20, [x10], #0x8\n"
    "fmla z26.s, p1/M, z17.s, z0.s\n"
    "fmla z27.s, p1/M, z18.s, z0.s\n"
    "ld1w { z14.s }, p0/Z, [x9, x11, LSL #2]\n"
    "fmla z28.s, p1/M, z19.s, z0.s\n"
    "fmla z29.s, p1/M, z20.s, z0.s\n"
    "ld1w { z15.s }, p0/Z, [x28, x11, LSL #2]\n"
    "ld1w { z16.s }, p0/Z, [x27, x11, LSL #2]\n"
    "fmla z30.s, p1/M, z21.s, z0.s\n"
    "fmla z31.s, p1/M, z22.s, z0.s\n"
    "ld1w { z0.s }, p1/Z, [%x[params]]\n"
    "ld1w { z17.s }, p0/Z, [x26, x11, LSL #2]\n"
    "ld1w { z18.s }, p0/Z, [x24, x11, LSL #2]\n"
    "ld1w { z19.s }, p0/Z, [x23, x11, LSL #2]\n"
    "addvl %x[params], %x[params], #1\n"
    "ld1w { z20.s }, p0/Z, [x22, x11, LSL #2]\n"
    "ld1w { z21.s }, p0/Z, [x21, x11, LSL #2]\n"
    "ld1w { z22.s }, p0/Z, [x20, x11, LSL #2]\n"
    "bgt 3b\n"
    "4:"  // Channel loop: Planar tail
    "fmla z23.s, p1/M, z14.s, z0.s\n"
    "fmla z24.s, p1/M, z15.s, z0.s\n"
    "fmax z23.s, p1/M, z23.s, z2.s\n"
    "fmax z24.s, p1/M, z24.s, z2.s\n"
    "fmla z25.s, p1/M, z16.s, z0.s\n"
    "fmla z26.s, p1/M, z17.s, z0.s\n"
    "fmax z25.s, p1/M, z25.s, z2.s\n"
    "fmax z26.s, p1/M, z26.s, z2.s\n"
    "fmla z27.s, p1/M, z18.s, z0.s\n"
    "fmla z28.s, p1/M, z19.s, z0.s\n"
    "fmax z27.s, p1/M, z27.s, z2.s\n"
    "fmax z28.s, p1/M, z28.s, z2.s\n"
    "fmla z29.s, p1/M, z20.s, z0.s\n"
    "fmla z30.s, p1/M, z21.s, z0.s\n"
    "fmax z29.s, p1/M, z29.s, z2.s\n"
    "fmax z30.s, p1/M, z30.s, z2.s\n"
    "fmla z31.s, p1/M, z22.s, z0.s\n"
    "fmax z31.s, p1/M, z31.s, z2.s\n"
    "ldp x28, x27, [%x[outptrs], #0x0]\n"
    "ldp x26, x25, [%x[outptrs], #0x10]\n"
    "ldp x24, x23, [%x[outptrs], #0x20]\n"
    "ldp x22, x21, [%x[outptrs], #0x30]\n"
    "fmin z23.s, p1/M, z23.s, z1.s\n"
    "fmin z24.s, p1/M, z24.s, z1.s\n"
    "ldr x20, [%x[outptrs], #0x40]\n"
    "fmin z25.s, p1/M, z25.s, z1.s\n"
    "fmin z26.s, p1/M, z26.s, z1.s\n"
    "st1w { z23.s }, p0, [x28, x11, LSL #2]\n"
    "fmin z27.s, p1/M, z27.s, z1.s\n"
    "fmin z28.s, p1/M, z28.s, z1.s\n"
    "st1w { z24.s }, p0, [x27, x11, LSL #2]\n"
    "fmin z29.s, p1/M, z29.s, z1.s\n"
    "fmin z30.s, p1/M, z30.s, z1.s\n"
    "st1w { z25.s }, p0, [x26, x11, LSL #2]\n"
    "fmin z31.s, p1/M, z31.s, z1.s\n"
    "st1w { z26.s }, p0, [x25, x11, LSL #2]\n"
    "st1w { z27.s }, p0, [x24, x11, LSL #2]\n"
    "st1w { z28.s }, p0, [x23, x11, LSL #2]\n"
    "st1w { z29.s }, p0, [x22, x11, LSL #2]\n"
    "st1w { z30.s }, p0, [x21, x11, LSL #2]\n"
    "st1w { z31.s }, p0, [x20, x11, LSL #2]\n"
    "incw x11\n"
    "whilelt p0.s, x11, %x[n_channels]\n"
    "b.any 1b\n"
    : [params] "+&r" (params)
    : [bias] "r" (bias), [inptrs] "r" (inptrs), [minmax_vals] "r" (minmax_vals), [n_channels] "r" ((uint64_t) n_channels), [n_points] "r" ((uint64_t) n_points), [outptrs] "r" (outptrs)
    : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
