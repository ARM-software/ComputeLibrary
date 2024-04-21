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

void sme2_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl(
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
      inptrs[2] = input_ptrs[4];
      inptrs[3] = input_ptrs[20];
      inptrs[4] = input_ptrs[7];
      inptrs[5] = input_ptrs[24];
      inptrs[6] = input_ptrs[11];
      inptrs[7] = input_ptrs[1];
      inptrs[8] = input_ptrs[3];
      inptrs[9] = input_ptrs[13];
      inptrs[10] = input_ptrs[5];
      inptrs[11] = input_ptrs[9];
      inptrs[12] = input_ptrs[15];
      inptrs[13] = input_ptrs[17];
      inptrs[14] = input_ptrs[19];
      inptrs[15] = input_ptrs[21];
      inptrs[16] = input_ptrs[6];
      inptrs[17] = input_ptrs[8];
      inptrs[18] = input_ptrs[23];
      inptrs[19] = input_ptrs[16];
      inptrs[20] = input_ptrs[2];
      inptrs[21] = input_ptrs[18];
      inptrs[22] = input_ptrs[10];
      inptrs[23] = input_ptrs[14];
      inptrs[24] = input_ptrs[22];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ldr x8, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "add x17, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ptrue p3.b\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "ld1w { z20.s }, p3/Z, [x8]\n"
    "addvl x8, x8, #1\n"
    "ldp x24, x23, [x17, #0x0]\n"
    "ldp x22, x21, [x17, #0x10]\n"
    "cntw x16\n"
    ".inst 0xa040c100  // ld1w { z0.s-z3.s }, pn8.b/Z, [x8]\n"
    "addvl x8, x8, #4\n"
    "ldr x20, [x17, #0x20]\n"
    "mov x15, #0x0\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    ".inst 0xa040c104  // ld1w { z4.s-z7.s }, pn8.b/Z, [x8]\n"
    "addvl x8, x8, #4\n"
    "cmp x16, %x[n_channels]\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ld1rw { z22.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ld1rw { z14.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x13, XZR, x16\n"
    "ld1w { z8.s }, p3/Z, [x8]\n"
    "addvl x8, x8, #1\n"
    "ld1w { z9.s }, p2/Z, [x24, x15, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x23, x15, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x22, x15, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x21, x15, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x20, x15, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z21, z20\n fmla z21.s, p3/M, z8.s, z9.s\n"
    "movprfx z24, z20\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "ldr x22, [x17, #0x30]\n"
    "incw x13\n"
    "movprfx z25, z20\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "fmla z21.s, p3/M, z0.s, z10.s\n"
    "ldr x25, [x17, #0x38]\n"
    "mov p1.b, p2.b\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "movprfx z26, z20\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "ldr x21, [x17, #0x28]\n"
    "whilelt p0.s, x16, %x[n_channels]\n"
    "movprfx z27, z20\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z20\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "ldr x20, [x17, #0x48]\n"
    "ld1w { z19.s }, p2/Z, [x20, x15, LSL #2]\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z23.s }, p2/Z, [x22, x15, LSL #2]\n"
    "movprfx z29, z20\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x20, [x17, #0x40]\n"
    "fmla z21.s, p3/M, z5.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z23.s\n"
    "ldr x24, [x17, #0x50]\n"
    "movprfx z31, z20\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ldr x23, [x17, #0x58]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "ldr x22, [x17, #0x60]\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z17.s }, p2/Z, [x25, x15, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ldr x12, [x17, #0x70]\n"
    "ld1w { z16.s }, p2/Z, [x21, x15, LSL #2]\n"
    "movprfx z30, z20\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "fmla z21.s, p3/M, z7.s, z23.s\n"
    "ldr x21, [x17, #0x68]\n"
    "fmla z24.s, p3/M, z0.s, z17.s\n"
    "fmla z31.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x15, LSL #2]\n"
    "ldr x27, [x17, #0x78]\n"
    "fmla z26.s, p3/M, z4.s, z23.s\n"
    "fmla z27.s, p3/M, z3.s, z23.s\n"
    "ldr x20, [x17, #0x80]\n"
    "ld1w { z20.s }, p3/Z, [x8]\n"
    "fmla z30.s, p3/M, z0.s, z23.s\n"
    "fmla z28.s, p3/M, z4.s, z19.s\n"
    "ldr x11, [x17, #0x88]\n"
    "addvl x8, x8, #1\n"
    "fmla z29.s, p3/M, z1.s, z23.s\n"
    "fmla z21.s, p3/M, z1.s, z17.s\n"
    "ld1w { z18.s }, p2/Z, [x24, x15, LSL #2]\n"
    "ldr x26, [x17, #0x90]\n"
    "fmla z24.s, p3/M, z2.s, z16.s\n"
    "fmla z25.s, p3/M, z1.s, z16.s\n"
    "ld1w { z11.s }, p2/Z, [x23, x15, LSL #2]\n"
    "ldr x25, [x17, #0x98]\n"
    "ld1w { z17.s }, p2/Z, [x22, x15, LSL #2]\n"
    "fmla z27.s, p3/M, z5.s, z19.s\n"
    "fmla z30.s, p3/M, z2.s, z19.s\n"
    "ldr x24, [x17, #0xa0]\n"
    "fmla z26.s, p3/M, z0.s, z18.s\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ldr x10, [x14, #0x0]\n"
    "fmla z24.s, p3/M, z8.s, z19.s\n"
    "fmla z25.s, p3/M, z7.s, z19.s\n"
    "ldr x9, [x14, #0x8]\n"
    "fmla z31.s, p3/M, z1.s, z19.s\n"
    "fmla z29.s, p3/M, z3.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x15, LSL #2]\n"
    "ldr x23, [x17, #0xa8]\n"
    "fmla z26.s, p3/M, z6.s, z17.s\n"
    "fmla z27.s, p3/M, z7.s, z16.s\n"
    "ld1w { z23.s }, p2/Z, [x20, x15, LSL #2]\n"
    "ldr x22, [x17, #0xc0]\n"
    "fmla z28.s, p3/M, z6.s, z16.s\n"
    "fmla z30.s, p3/M, z4.s, z16.s\n"
    "ldr x28, [x14, #0x10]\n"
    "fmla z21.s, p3/M, z3.s, z18.s\n"
    "fmla z25.s, p3/M, z5.s, z11.s\n"
    "ld1w { z15.s }, p2/Z, [x12, x15, LSL #2]\n"
    "ldr x21, [x17, #0xb0]\n"
    "fmla z29.s, p3/M, z5.s, z16.s\n"
    "fmla z31.s, p3/M, z3.s, z16.s\n"
    "ld1w { z19.s }, p2/Z, [x27, x15, LSL #2]\n"
    "ldr x20, [x17, #0xb8]\n"
    "fmla z26.s, p3/M, z8.s, z16.s\n"
    "fmla z28.s, p3/M, z8.s, z15.s\n"
    "ldr x27, [x14, #0x18]\n"
    "fmla z30.s, p3/M, z6.s, z19.s\n"
    "fmla z24.s, p3/M, z3.s, z23.s\n"
    "fmla z27.s, p3/M, z0.s, z23.s\n"
    "fmla z31.s, p3/M, z5.s, z15.s\n"
    "ld1w { z17.s }, p2/Z, [x11, x15, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z19.s\n"
    "ld1w { z19.s }, p2/Z, [x26, x15, LSL #2]\n"
    "fmla z21.s, p3/M, z4.s, z23.s\n"
    "fmla z26.s, p3/M, z1.s, z23.s\n"
    "fmla z24.s, p3/M, z5.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x15, LSL #2]\n"
    "fmla z25.s, p3/M, z4.s, z17.s\n"
    "fmla z27.s, p3/M, z2.s, z17.s\n"
    "fmla z28.s, p3/M, z1.s, z17.s\n"
    "fmla z30.s, p3/M, z8.s, z19.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x15, LSL #2]\n"
    "ldr x26, [x17, #0x20]\n"
    "fmla z21.s, p3/M, z2.s, z17.s\n"
    "fmla z26.s, p3/M, z7.s, z16.s\n"
    "fmla z27.s, p3/M, z6.s, z16.s\n"
    "fmla z29.s, p3/M, z4.s, z16.s\n"
    "fmla z30.s, p3/M, z3.s, z16.s\n"
    "ld1w { z18.s }, p2/Z, [x21, x15, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x23, x15, LSL #2]\n"
    "fmla z21.s, p3/M, z6.s, z18.s\n"
    "fmla z31.s, p3/M, z4.s, z16.s\n"
    "fmla z24.s, p3/M, z1.s, z17.s\n"
    "fmla z25.s, p3/M, z0.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x15, LSL #2]\n"
    "fmax z21.s, p3/M, z21.s, z22.s\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "fmla z29.s, p3/M, z0.s, z18.s\n"
    "fmla z31.s, p3/M, z2.s, z17.s\n"
    "fmla z27.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x22, x15, LSL #2]\n"
    "ldp x22, x21, [x17, #0x0]\n"
    "fmla z26.s, p3/M, z3.s, z18.s\n"
    "fmla z25.s, p3/M, z8.s, z17.s\n"
    "ldp x25, x24, [x17, #0x10]\n"
    "incw x15\n"
    "fmin z21.s, p3/M, z21.s, z14.s\n"
    "st1w { z21.s }, p1, [x10, x13, LSL #2]\n"
    "ldr x20, [x14, #0x20]\n"
    "fmla z28.s, p3/M, z5.s, z17.s\n"
    "fmla z29.s, p3/M, z8.s, z16.s\n"
    "fmla z30.s, p3/M, z7.s, z16.s\n"
    "ld1w { z9.s }, p0/Z, [x22, x16, LSL #2]\n"
    "whilelt p2.s, x15, %x[n_channels]\n"
    "fmla z31.s, p3/M, z6.s, z16.s\n"
    ".inst 0xc1aecad8  // fclamp { z24.s-z27.s }, z22.s, z14.s\n"
    "st1w { z24.s }, p1, [x9, x13, LSL #2]\n"
    "ldr x23, [x14, #0x28]\n"
    "st1w { z25.s }, p1, [x28, x13, LSL #2]\n"
    "ldr x22, [x14, #0x30]\n"
    "ld1w { z10.s }, p0/Z, [x21, x16, LSL #2]\n"
    ".inst 0xc1aecadc  // fclamp { z28.s-z31.s }, z22.s, z14.s\n"
    "st1w { z26.s }, p1, [x27, x13, LSL #2]\n"
    "ldr x21, [x14, #0x38]\n"
    "ld1w { z11.s }, p0/Z, [x25, x16, LSL #2]\n"
    "st1w { z27.s }, p1, [x20, x13, LSL #2]\n"
    "ldr x20, [x14, #0x40]\n"
    "ld1w { z12.s }, p0/Z, [x24, x16, LSL #2]\n"
    "ld1w { z13.s }, p0/Z, [x26, x16, LSL #2]\n"
    "incw x16\n"
    "cmp x16, %x[n_channels]\n"
    "st1w { z28.s }, p1, [x23, x13, LSL #2]\n"
    ".inst 0xa040c100  // ld1w { z0.s-z3.s }, pn8.b/Z, [x8]\n"
    "addvl x8, x8, #4\n"
    "st1w { z29.s }, p1, [x22, x13, LSL #2]\n"
    ".inst 0xa040c104  // ld1w { z4.s-z7.s }, pn8.b/Z, [x8]\n"
    "addvl x8, x8, #4\n"
    "st1w { z30.s }, p1, [x21, x13, LSL #2]\n"
    "st1w { z31.s }, p1, [x20, x13, LSL #2]\n"
    "ld1w { z8.s }, p3/Z, [x8]\n"
    "addvl x8, x8, #1\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z21, z20\n fmla z21.s, p3/M, z8.s, z9.s\n"
    "movprfx z24, z20\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "ldr x23, [x17, #0x30]\n"
    "incw x13\n"
    "movprfx z25, z20\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "fmla z21.s, p3/M, z0.s, z10.s\n"
    "ldr x22, [x17, #0x38]\n"
    "mov p0.b, p2.b\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "movprfx z26, z20\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "ldr x21, [x17, #0x28]\n"
    "movprfx z27, z20\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z20\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "ldr x20, [x17, #0x48]\n"
    "ld1w { z19.s }, p2/Z, [x20, x15, LSL #2]\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z18.s }, p2/Z, [x23, x15, LSL #2]\n"
    "movprfx z29, z20\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x20, [x17, #0x40]\n"
    "fmla z21.s, p3/M, z5.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z18.s\n"
    "ldr x25, [x17, #0x50]\n"
    "movprfx z31, z20\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ldr x24, [x17, #0x58]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "ldr x23, [x17, #0x60]\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z17.s }, p2/Z, [x22, x15, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ldr x12, [x17, #0x70]\n"
    "ld1w { z16.s }, p2/Z, [x21, x15, LSL #2]\n"
    "movprfx z30, z20\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "fmla z21.s, p3/M, z7.s, z18.s\n"
    "ldr x22, [x17, #0x68]\n"
    "fmla z24.s, p3/M, z0.s, z17.s\n"
    "fmla z31.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x15, LSL #2]\n"
    "ldr x21, [x17, #0x78]\n"
    "fmla z26.s, p3/M, z4.s, z18.s\n"
    "fmla z27.s, p3/M, z3.s, z18.s\n"
    "ldr x20, [x17, #0x80]\n"
    "fmla z30.s, p3/M, z0.s, z18.s\n"
    "fmla z28.s, p3/M, z4.s, z19.s\n"
    "ldr x11, [x17, #0x88]\n"
    "fmla z29.s, p3/M, z1.s, z18.s\n"
    "fmla z21.s, p3/M, z1.s, z17.s\n"
    "ld1w { z20.s }, p2/Z, [x25, x15, LSL #2]\n"
    "ldr x10, [x17, #0x90]\n"
    "fmla z24.s, p3/M, z2.s, z16.s\n"
    "fmla z25.s, p3/M, z1.s, z16.s\n"
    "ld1w { z17.s }, p2/Z, [x24, x15, LSL #2]\n"
    "ldr x9, [x17, #0x98]\n"
    "ld1w { z16.s }, p2/Z, [x23, x15, LSL #2]\n"
    "fmla z27.s, p3/M, z5.s, z19.s\n"
    "fmla z30.s, p3/M, z2.s, z19.s\n"
    "ldr x28, [x17, #0xa0]\n"
    "fmla z26.s, p3/M, z0.s, z20.s\n"
    "fmla z28.s, p3/M, z2.s, z17.s\n"
    "ldr x27, [x14, #0x0]\n"
    "fmla z24.s, p3/M, z8.s, z19.s\n"
    "fmla z25.s, p3/M, z7.s, z19.s\n"
    "ldr x26, [x14, #0x8]\n"
    "fmla z31.s, p3/M, z1.s, z19.s\n"
    "fmla z29.s, p3/M, z3.s, z16.s\n"
    "ld1w { z19.s }, p2/Z, [x22, x15, LSL #2]\n"
    "ldr x25, [x17, #0xa8]\n"
    "fmla z26.s, p3/M, z6.s, z16.s\n"
    "fmla z27.s, p3/M, z7.s, z19.s\n"
    "ld1w { z18.s }, p2/Z, [x20, x15, LSL #2]\n"
    "ldr x23, [x17, #0xc0]\n"
    "fmla z28.s, p3/M, z6.s, z19.s\n"
    "fmla z30.s, p3/M, z4.s, z19.s\n"
    "ldr x24, [x14, #0x10]\n"
    "fmla z21.s, p3/M, z3.s, z20.s\n"
    "fmla z25.s, p3/M, z5.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x12, x15, LSL #2]\n"
    "ldr x22, [x17, #0xb0]\n"
    "fmla z29.s, p3/M, z5.s, z19.s\n"
    "fmla z31.s, p3/M, z3.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x15, LSL #2]\n"
    "ldr x20, [x17, #0xb8]\n"
    "fmla z26.s, p3/M, z8.s, z19.s\n"
    "fmla z28.s, p3/M, z8.s, z17.s\n"
    "ldr x21, [x14, #0x18]\n"
    "fmla z30.s, p3/M, z6.s, z16.s\n"
    "fmla z24.s, p3/M, z3.s, z18.s\n"
    "fmla z27.s, p3/M, z0.s, z18.s\n"
    "fmla z31.s, p3/M, z5.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x11, x15, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z16.s\n"
    "ld1w { z19.s }, p2/Z, [x10, x15, LSL #2]\n"
    "fmla z21.s, p3/M, z4.s, z18.s\n"
    "fmla z26.s, p3/M, z1.s, z18.s\n"
    "fmla z24.s, p3/M, z5.s, z17.s\n"
    "ld1w { z16.s }, p2/Z, [x9, x15, LSL #2]\n"
    "fmla z25.s, p3/M, z4.s, z17.s\n"
    "fmla z27.s, p3/M, z2.s, z17.s\n"
    "fmla z28.s, p3/M, z1.s, z17.s\n"
    "fmla z30.s, p3/M, z8.s, z19.s\n"
    "ld1w { z17.s }, p2/Z, [x28, x15, LSL #2]\n"
    "fmla z21.s, p3/M, z2.s, z17.s\n"
    "fmla z26.s, p3/M, z7.s, z16.s\n"
    "fmla z27.s, p3/M, z6.s, z16.s\n"
    "fmla z29.s, p3/M, z4.s, z16.s\n"
    "fmla z30.s, p3/M, z3.s, z16.s\n"
    "ld1w { z18.s }, p2/Z, [x22, x15, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z19.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x15, LSL #2]\n"
    "fmla z21.s, p3/M, z6.s, z18.s\n"
    "fmla z31.s, p3/M, z4.s, z16.s\n"
    "fmla z24.s, p3/M, z1.s, z17.s\n"
    "fmla z25.s, p3/M, z0.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x15, LSL #2]\n"
    "fmax z21.s, p3/M, z21.s, z22.s\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "fmla z29.s, p3/M, z0.s, z18.s\n"
    "fmla z31.s, p3/M, z2.s, z17.s\n"
    "fmla z27.s, p3/M, z8.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x23, x15, LSL #2]\n"
    "fmla z26.s, p3/M, z3.s, z18.s\n"
    "fmla z25.s, p3/M, z8.s, z17.s\n"
    "fmin z21.s, p3/M, z21.s, z14.s\n"
    "st1w { z21.s }, p0, [x27, x13, LSL #2]\n"
    "ldr x20, [x14, #0x20]\n"
    "fmla z28.s, p3/M, z5.s, z17.s\n"
    "fmla z29.s, p3/M, z8.s, z16.s\n"
    "fmla z30.s, p3/M, z7.s, z16.s\n"
    "fmla z31.s, p3/M, z6.s, z16.s\n"
    ".inst 0xc1aecad8  // fclamp { z24.s-z27.s }, z22.s, z14.s\n"
    "st1w { z24.s }, p0, [x26, x13, LSL #2]\n"
    "ldr x23, [x14, #0x28]\n"
    "st1w { z25.s }, p0, [x24, x13, LSL #2]\n"
    "ldr x22, [x14, #0x30]\n"
    ".inst 0xc1aecadc  // fclamp { z28.s-z31.s }, z22.s, z14.s\n"
    "st1w { z26.s }, p0, [x21, x13, LSL #2]\n"
    "ldr x21, [x14, #0x38]\n"
    "st1w { z27.s }, p0, [x20, x13, LSL #2]\n"
    "ldr x20, [x14, #0x40]\n"
    "st1w { z28.s }, p0, [x23, x13, LSL #2]\n"
    "st1w { z29.s }, p0, [x22, x13, LSL #2]\n"
    "st1w { z30.s }, p0, [x21, x13, LSL #2]\n"
    "st1w { z31.s }, p0, [x20, x13, LSL #2]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
