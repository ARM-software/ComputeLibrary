/*
 * Copyright (c) 2022 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SME2)

#include <cstddef>
#include <cstdint>

namespace arm_conv {
namespace depthwise {

void sme2_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_indirect_impl(
  const float *const *const input_ptrs,
  float *const *const outptrs,
  const void *params,
  unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  struct Args
  {
    float *const *outptrs;
    const void *params;
    const float min, max;
    const float *inptrs[16];

    Args(
      const float *const *const input_ptrs,
      float *const *const outptrs,
      const void *const params,
      const float min,
      const float max
    ) : outptrs(outptrs), params(params), min(min), max(max)
    {
      inptrs[0] = input_ptrs[5];
      inptrs[1] = input_ptrs[0];
      inptrs[2] = input_ptrs[3];
      inptrs[3] = input_ptrs[6];
      inptrs[4] = input_ptrs[9];
      inptrs[5] = input_ptrs[12];
      inptrs[6] = input_ptrs[15];
      inptrs[7] = input_ptrs[1];
      inptrs[8] = input_ptrs[2];
      inptrs[9] = input_ptrs[10];
      inptrs[10] = input_ptrs[4];
      inptrs[11] = input_ptrs[7];
      inptrs[12] = input_ptrs[8];
      inptrs[13] = input_ptrs[11];
      inptrs[14] = input_ptrs[13];
      inptrs[15] = input_ptrs[14];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ldr x19, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "add x14, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ptrue p3.b\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "ld1w { z18.s }, p3/Z, [x13]\n"
    "addvl x13, x13, #1\n"
    "ldp x12, x11, [x19, #0x0]\n"
    "cntw x10\n"
    ".inst 0xa040c1a0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x13]\n"
    "addvl x13, x13, #4\n"
    "ldp x9, x28, [x19, #0x10]\n"
    "mov x27, #0x0\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    ".inst 0xa040c1a4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x13]\n"
    "ldp x26, x25, [x14, #0x0]\n"
    "addvl x13, x13, #4\n"
    "cmp x10, %x[n_channels]\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ldp x24, x21, [x14, #0x10]\n"
    "ld1rw { z16.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x23, XZR, x10\n"
    "ldr x22, [x14, #0x20]\n"
    "ld1w { z8.s }, p3/Z, [x13]\n"
    "addvl x13, x13, #1\n"
    "ld1w { z9.s }, p2/Z, [x26, x27, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x25, x27, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x24, x27, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x22, x27, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z28, z18\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "ldr x21, [x14, #0x28]\n"
    "whilelt p1.s, x10, %x[n_channels]\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ldr x20, [x14, #0x30]\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ldr x19, [x14, #0x38]\n"
    "ld1w { z11.s }, p2/Z, [x20, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ldr x25, [x14, #0x48]\n"
    "ld1w { z10.s }, p2/Z, [x25, x27, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x19, x27, LSL #2]\n"
    "ldr x26, [x14, #0x40]\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x26, x27, LSL #2]\n"
    "ldr x24, [x14, #0x50]\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "ldr x21, [x14, #0x58]\n"
    "ld1w { z18.s }, p3/Z, [x13]\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x27, LSL #2]\n"
    "ldr x22, [x14, #0x60]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ldr x21, [x14, #0x68]\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "ldr x20, [x14, #0x70]\n"
    "addvl x13, x13, #1\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x22, x27, LSL #2]\n"
    "ldr x19, [x14, #0x78]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "ldp x26, x25, [x14, #0x0]\n"
    "incw x23\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ldp x24, x21, [x14, #0x10]\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "ldr x22, [x14, #0x20]\n"
    "ld1w { z13.s }, p1/Z, [x22, x10, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x20, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x19, x27, LSL #2]\n"
    "incw x27\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    "ld1w { z9.s }, p1/Z, [x26, x10, LSL #2]\n"
    "whilelt p2.s, x27, %x[n_channels]\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    "ld1w { z10.s }, p1/Z, [x25, x10, LSL #2]\n"
    "ld1w { z11.s }, p1/Z, [x24, x10, LSL #2]\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "st1w { z28.s }, p0, [x12, x23, LSL #2]\n"
    "ld1w { z12.s }, p1/Z, [x21, x10, LSL #2]\n"
    "incw x10\n"
    "cmp x10, %x[n_channels]\n"
    "st1w { z29.s }, p0, [x11, x23, LSL #2]\n"
    ".inst 0xa040c1a0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x13]\n"
    "addvl x13, x13, #4\n"
    "st1w { z30.s }, p0, [x9, x23, LSL #2]\n"
    ".inst 0xa040c1a4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x13]\n"
    "addvl x13, x13, #4\n"
    "st1w { z31.s }, p0, [x28, x23, LSL #2]\n"
    "ld1w { z8.s }, p3/Z, [x13]\n"
    "addvl x13, x13, #1\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z28, z18\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "ldr x21, [x14, #0x28]\n"
    "incw x23\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ldr x20, [x14, #0x30]\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ldr x19, [x14, #0x38]\n"
    "ld1w { z11.s }, p2/Z, [x20, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ldr x25, [x14, #0x48]\n"
    "ld1w { z10.s }, p2/Z, [x25, x27, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x19, x27, LSL #2]\n"
    "ldr x26, [x14, #0x40]\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x26, x27, LSL #2]\n"
    "ldr x24, [x14, #0x50]\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "ldr x21, [x14, #0x58]\n"
    "mov p0.b, p2.b\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x27, LSL #2]\n"
    "ldr x22, [x14, #0x60]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ldr x21, [x14, #0x68]\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "ldr x20, [x14, #0x70]\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x22, x27, LSL #2]\n"
    "ldr x19, [x14, #0x78]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x21, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x20, x27, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x19, x27, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "st1w { z28.s }, p0, [x12, x23, LSL #2]\n"
    "st1w { z29.s }, p0, [x11, x23, LSL #2]\n"
    "st1w { z30.s }, p0, [x9, x23, LSL #2]\n"
    "st1w { z31.s }, p0, [x28, x23, LSL #2]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
