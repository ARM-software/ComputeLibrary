/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SME2)

namespace arm_conv {
namespace depthwise {

void sme2_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(
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
    const float *inptrs[25];

    Args(
      const float *const *const input_ptrs,
      float *const *const outptrs,
      const void *const params,
      const float min,
      const float max
    ) : outptrs(outptrs), params(params), min(min), max(max)
    {
      inptrs[0] = input_ptrs[12];
      inptrs[1] = input_ptrs[0];
      inptrs[2] = input_ptrs[1];
      inptrs[3] = input_ptrs[3];
      inptrs[4] = input_ptrs[4];
      inptrs[5] = input_ptrs[5];
      inptrs[6] = input_ptrs[6];
      inptrs[7] = input_ptrs[2];
      inptrs[8] = input_ptrs[8];
      inptrs[9] = input_ptrs[9];
      inptrs[10] = input_ptrs[7];
      inptrs[11] = input_ptrs[15];
      inptrs[12] = input_ptrs[10];
      inptrs[13] = input_ptrs[16];
      inptrs[14] = input_ptrs[11];
      inptrs[15] = input_ptrs[18];
      inptrs[16] = input_ptrs[13];
      inptrs[17] = input_ptrs[19];
      inptrs[18] = input_ptrs[20];
      inptrs[19] = input_ptrs[14];
      inptrs[20] = input_ptrs[21];
      inptrs[21] = input_ptrs[17];
      inptrs[22] = input_ptrs[23];
      inptrs[23] = input_ptrs[22];
      inptrs[24] = input_ptrs[24];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ldr x20, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "add x16, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ptrue p3.b\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "ld1w { z26.s }, p3/Z, [x15]\n"
    "addvl x15, x15, #1\n"
    "ldp x14, x13, [x20, #0x0]\n"
    "cntw x12\n"
    ".inst 0xa040c1e0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x15]\n"
    "addvl x15, x15, #4\n"
    "ldp x11, x10, [x20, #0x10]\n"
    "mov x9, #0x0\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    ".inst 0xa040c1e4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x15]\n"
    "ldp x28, x26, [x16, #0x0]\n"
    "addvl x15, x15, #4\n"
    "cmp x12, %x[n_channels]\n"
    "ld1rw { z25.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ldp x25, x24, [x16, #0x10]\n"
    "ld1rw { z24.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x27, XZR, x12\n"
    "ldp x23, x22, [x16, #0x20]\n"
    "ld1w { z8.s }, p3/Z, [x15]\n"
    "addvl x15, x15, #1\n"
    "ldp x21, x20, [x16, #0x30]\n"
    "ld1w { z9.s }, p2/Z, [x28, x9, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x26, x9, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x25, x9, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x24, x9, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x23, x9, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x22, x9, LSL #2]\n"
    "ld1w { z15.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z28, z26\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z26\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "ldr x21, [x16, #0x40]\n"
    "whilelt p1.s, x12, %x[n_channels]\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ldr x20, [x16, #0x48]\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z22.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x20, [x16, #0x50]\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0x58]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "fmla z29.s, p3/M, z4.s, z22.s\n"
    "ldr x21, [x16, #0x78]\n"
    "ld1w { z23.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "fmla z29.s, p3/M, z5.s, z18.s\n"
    "ldr x20, [x16, #0x60]\n"
    "ld1w { z13.s }, p2/Z, [x20, x9, LSL #2]\n"
    "movprfx z30, z26\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z26\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ldr x20, [x16, #0x80]\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z17.s\n"
    "fmla z29.s, p3/M, z3.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x21, [x16, #0x68]\n"
    "fmla z30.s, p3/M, z3.s, z23.s\n"
    "fmla z31.s, p3/M, z4.s, z16.s\n"
    "ldr x20, [x16, #0x88]\n"
    "ld1w { z17.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z13.s\n"
    "fmla z31.s, p3/M, z1.s, z18.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0x70]\n"
    "ldr x20, [x16, #0x98]\n"
    "fmla z30.s, p3/M, z4.s, z17.s\n"
    "fmla z31.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z13.s\n"
    "ld1w { z4.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0x90]\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ldr x20, [x16, #0xa8]\n"
    "fmla z31.s, p3/M, z2.s, z4.s\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0xa0]\n"
    "fmla z30.s, p3/M, z6.s, z16.s\n"
    "fmla z31.s, p3/M, z3.s, z17.s\n"
    "ldr x20, [x16, #0xb0]\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z16.s\n"
    "fmla z29.s, p3/M, z7.s, z18.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0xb8]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "fmla z30.s, p3/M, z5.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0xc0]\n"
    "fmla z31.s, p3/M, z6.s, z17.s\n"
    "fmla z29.s, p3/M, z8.s, z4.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldp x20, x26, [x16, #0x0]\n"
    "fmla z30.s, p3/M, z8.s, z17.s\n"
    "fmla z31.s, p3/M, z8.s, z16.s\n"
    "ldp x25, x24, [x16, #0x10]\n"
    "ld1w { z26.s }, p3/Z, [x15]\n"
    "addvl x15, x15, #1\n"
    "incw x9\n"
    "ldp x23, x22, [x16, #0x20]\n"
    "ld1w { z9.s }, p1/Z, [x20, x12, LSL #2]\n"
    "incw x27\n"
    "mov p0.b, p2.b\n"
    "ldp x21, x20, [x16, #0x30]\n"
    "ld1w { z10.s }, p1/Z, [x26, x12, LSL #2]\n"
    "whilelt p2.s, x9, %x[n_channels]\n"
    ".inst 0xc1b8cb3c  // fclamp { z28.s-z31.s }, z25.s, z24.s\n"
    "ld1w { z11.s }, p1/Z, [x25, x12, LSL #2]\n"
    "st1w { z28.s }, p0, [x14, x27, LSL #2]\n"
    "ld1w { z12.s }, p1/Z, [x24, x12, LSL #2]\n"
    "st1w { z29.s }, p0, [x13, x27, LSL #2]\n"
    "ld1w { z13.s }, p1/Z, [x23, x12, LSL #2]\n"
    "st1w { z30.s }, p0, [x11, x27, LSL #2]\n"
    "ld1w { z14.s }, p1/Z, [x22, x12, LSL #2]\n"
    "st1w { z31.s }, p0, [x10, x27, LSL #2]\n"
    "ld1w { z15.s }, p1/Z, [x21, x12, LSL #2]\n"
    "ld1w { z16.s }, p1/Z, [x20, x12, LSL #2]\n"
    "incw x12\n"
    "cmp x12, %x[n_channels]\n"
    ".inst 0xa040c1e0  // ld1w { z0.s-z3.s }, pn8.b/Z, [x15]\n"
    "addvl x15, x15, #4\n"
    ".inst 0xa040c1e4  // ld1w { z4.s-z7.s }, pn8.b/Z, [x15]\n"
    "addvl x15, x15, #4\n"
    "ld1w { z8.s }, p3/Z, [x15]\n"
    "addvl x15, x15, #1\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z28, z26\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z26\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "ldr x21, [x16, #0x40]\n"
    "incw x27\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ldr x20, [x16, #0x48]\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z17.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x20, [x16, #0x50]\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "ld1w { z20.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0x58]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "fmla z29.s, p3/M, z4.s, z17.s\n"
    "ldr x21, [x16, #0x78]\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "fmla z29.s, p3/M, z5.s, z18.s\n"
    "ldr x20, [x16, #0x60]\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "movprfx z30, z26\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z26\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ldr x20, [x16, #0x80]\n"
    "ld1w { z19.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z20.s\n"
    "fmla z29.s, p3/M, z3.s, z20.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x21, [x16, #0x68]\n"
    "fmla z30.s, p3/M, z3.s, z17.s\n"
    "fmla z31.s, p3/M, z4.s, z16.s\n"
    "ldr x20, [x16, #0x88]\n"
    "ld1w { z17.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z18.s\n"
    "fmla z31.s, p3/M, z1.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0x70]\n"
    "ldr x20, [x16, #0x98]\n"
    "fmla z30.s, p3/M, z4.s, z17.s\n"
    "fmla z31.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z18.s\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0x90]\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ldr x20, [x16, #0xa8]\n"
    "fmla z31.s, p3/M, z2.s, z18.s\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x16, #0xa0]\n"
    "fmla z30.s, p3/M, z6.s, z16.s\n"
    "fmla z31.s, p3/M, z3.s, z17.s\n"
    "ldr x20, [x16, #0xb0]\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z16.s\n"
    "fmla z29.s, p3/M, z7.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0xb8]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "fmla z30.s, p3/M, z5.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x16, #0xc0]\n"
    "fmla z31.s, p3/M, z6.s, z17.s\n"
    "fmla z29.s, p3/M, z8.s, z18.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z17.s\n"
    "fmla z31.s, p3/M, z8.s, z16.s\n"
    "mov p0.b, p2.b\n"
    ".inst 0xc1b8cb3c  // fclamp { z28.s-z31.s }, z25.s, z24.s\n"
    "st1w { z28.s }, p0, [x14, x27, LSL #2]\n"
    "st1w { z29.s }, p0, [x13, x27, LSL #2]\n"
    "st1w { z30.s }, p0, [x11, x27, LSL #2]\n"
    "st1w { z31.s }, p0, [x10, x27, LSL #2]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
