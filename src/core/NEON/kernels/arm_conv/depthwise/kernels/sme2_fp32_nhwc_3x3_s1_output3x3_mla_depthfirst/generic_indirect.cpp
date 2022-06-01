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
    "ldr x16, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "add x15, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ptrue p3.b\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "ld1w { z18.s }, p3/Z, [x16]\n"
    "addvl x16, x16, #1\n"
    "ldp x14, x13, [x15, #0x0]\n"
    "ldp x12, x11, [x15, #0x10]\n"
    "cntw x10\n"
    ".inst 0xa040c200  // ld1w { z0.s-z3.s }, pn8.b/Z, [x16]\n"
    "addvl x16, x16, #4\n"
    "ldr x9, [x15, #0x20]\n"
    "mov x28, #0x0\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    ".inst 0xa040c204  // ld1w { z4.s-z7.s }, pn8.b/Z, [x16]\n"
    "addvl x16, x16, #4\n"
    "cmp x10, %x[n_channels]\n"
    "ldr x27, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ld1rw { z16.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x26, XZR, x10\n"
    "ld1w { z8.s }, p3/Z, [x16]\n"
    "addvl x16, x16, #1\n"
    "ld1w { z9.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x13, x28, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x12, x28, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x11, x28, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x9, x28, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z23, z18\n fmla z23.s, p3/M, z8.s, z9.s\n"
    "movprfx z24, z18\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "ldr x25, [x15, #0x30]\n"
    "incw x26\n"
    "movprfx z25, z18\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "fmla z23.s, p3/M, z0.s, z10.s\n"
    "ldr x24, [x15, #0x38]\n"
    "mov p1.b, p2.b\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "movprfx z26, z18\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "ldr x23, [x15, #0x28]\n"
    "whilelt p0.s, x10, %x[n_channels]\n"
    "movprfx z27, z18\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z18\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "ldr x13, [x15, #0x48]\n"
    "ld1w { z10.s }, p2/Z, [x13, x28, LSL #2]\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x28, LSL #2]\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x14, [x15, #0x40]\n"
    "fmla z23.s, p3/M, z5.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z11.s\n"
    "ldr x12, [x15, #0x50]\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ldr x11, [x15, #0x58]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "ldr x9, [x15, #0x60]\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ldr x25, [x15, #0x70]\n"
    "ld1w { z12.s }, p2/Z, [x23, x28, LSL #2]\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "fmla z23.s, p3/M, z7.s, z11.s\n"
    "ldr x23, [x15, #0x68]\n"
    "fmla z24.s, p3/M, z0.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ldr x24, [x15, #0x78]\n"
    "fmla z26.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "ldr x14, [x15, #0x80]\n"
    "ld1w { z18.s }, p3/Z, [x16]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ldr x13, [x15, #0x88]\n"
    "addvl x16, x16, #1\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z23.s, p3/M, z1.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x28, LSL #2]\n"
    "ldr x12, [x15, #0x90]\n"
    "fmla z24.s, p3/M, z2.s, z12.s\n"
    "fmla z25.s, p3/M, z1.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x11, x28, LSL #2]\n"
    "ldr x11, [x15, #0x98]\n"
    "ld1w { z12.s }, p2/Z, [x9, x28, LSL #2]\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "fmla z30.s, p3/M, z2.s, z10.s\n"
    "ldr x9, [x15, #0xa0]\n"
    "fmla z26.s, p3/M, z0.s, z11.s\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "ldr x22, [x27, #0x0]\n"
    "fmla z24.s, p3/M, z8.s, z10.s\n"
    "fmla z25.s, p3/M, z7.s, z10.s\n"
    "ldr x21, [x27, #0x8]\n"
    "fmla z31.s, p3/M, z1.s, z10.s\n"
    "fmla z29.s, p3/M, z3.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x23, x28, LSL #2]\n"
    "ldr x23, [x15, #0xa8]\n"
    "fmla z26.s, p3/M, z6.s, z12.s\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "ld1w { z12.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ldr x14, [x15, #0xc0]\n"
    "fmla z28.s, p3/M, z6.s, z10.s\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "ldr x20, [x27, #0x10]\n"
    "fmla z23.s, p3/M, z3.s, z11.s\n"
    "fmla z25.s, p3/M, z5.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x28, LSL #2]\n"
    "ldr x25, [x15, #0xb0]\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z3.s, z10.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x28, LSL #2]\n"
    "ldr x24, [x15, #0xb8]\n"
    "fmla z26.s, p3/M, z8.s, z10.s\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ldr x19, [x27, #0x18]\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "fmla z31.s, p3/M, z5.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x12, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "fmla z26.s, p3/M, z1.s, z12.s\n"
    "fmla z24.s, p3/M, z5.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x28, LSL #2]\n"
    "fmla z25.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z30.s, p3/M, z8.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x28, LSL #2]\n"
    "ldr x9, [x15, #0x20]\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "fmla z26.s, p3/M, z7.s, z12.s\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x25, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x23, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z6.s, z12.s\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "fmla z24.s, p3/M, z1.s, z11.s\n"
    "fmla z25.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmax z23.s, p3/M, z23.s, z17.s\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z30.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ldp x14, x13, [x15, #0x0]\n"
    "fmla z26.s, p3/M, z3.s, z12.s\n"
    "fmla z25.s, p3/M, z8.s, z11.s\n"
    "ldp x12, x11, [x15, #0x10]\n"
    "incw x28\n"
    "fmin z23.s, p3/M, z23.s, z16.s\n"
    "st1w { z23.s }, p1, [x22, x26, LSL #2]\n"
    "ldr x22, [x27, #0x20]\n"
    "fmla z28.s, p3/M, z5.s, z11.s\n"
    "fmla z29.s, p3/M, z8.s, z13.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "ld1w { z9.s }, p0/Z, [x14, x10, LSL #2]\n"
    "whilelt p2.s, x28, %x[n_channels]\n"
    "fmla z31.s, p3/M, z6.s, z13.s\n"
    ".inst 0xc1b0ca38  // fclamp { z24.s-z27.s }, z17.s, z16.s\n"
    "st1w { z24.s }, p1, [x21, x26, LSL #2]\n"
    "ldr x21, [x27, #0x28]\n"
    "st1w { z25.s }, p1, [x20, x26, LSL #2]\n"
    "ldr x20, [x27, #0x30]\n"
    "ld1w { z10.s }, p0/Z, [x13, x10, LSL #2]\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "st1w { z26.s }, p1, [x19, x26, LSL #2]\n"
    "ldr x19, [x27, #0x38]\n"
    "ld1w { z11.s }, p0/Z, [x12, x10, LSL #2]\n"
    "st1w { z27.s }, p1, [x22, x26, LSL #2]\n"
    "ldr x22, [x27, #0x40]\n"
    "ld1w { z12.s }, p0/Z, [x11, x10, LSL #2]\n"
    "ld1w { z13.s }, p0/Z, [x9, x10, LSL #2]\n"
    "incw x10\n"
    "cmp x10, %x[n_channels]\n"
    "st1w { z28.s }, p1, [x21, x26, LSL #2]\n"
    ".inst 0xa040c200  // ld1w { z0.s-z3.s }, pn8.b/Z, [x16]\n"
    "addvl x16, x16, #4\n"
    "st1w { z29.s }, p1, [x20, x26, LSL #2]\n"
    ".inst 0xa040c204  // ld1w { z4.s-z7.s }, pn8.b/Z, [x16]\n"
    "addvl x16, x16, #4\n"
    "st1w { z30.s }, p1, [x19, x26, LSL #2]\n"
    "st1w { z31.s }, p1, [x22, x26, LSL #2]\n"
    "ld1w { z8.s }, p3/Z, [x16]\n"
    "addvl x16, x16, #1\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z23, z18\n fmla z23.s, p3/M, z8.s, z9.s\n"
    "movprfx z24, z18\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "ldr x25, [x15, #0x30]\n"
    "incw x26\n"
    "movprfx z25, z18\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "fmla z23.s, p3/M, z0.s, z10.s\n"
    "ldr x24, [x15, #0x38]\n"
    "mov p1.b, p2.b\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "movprfx z26, z18\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "ldr x23, [x15, #0x28]\n"
    "movprfx z27, z18\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z18\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "ldr x13, [x15, #0x48]\n"
    "ld1w { z10.s }, p2/Z, [x13, x28, LSL #2]\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x28, LSL #2]\n"
    "movprfx z29, z18\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x14, [x15, #0x40]\n"
    "fmla z23.s, p3/M, z5.s, z13.s\n"
    "fmla z24.s, p3/M, z6.s, z11.s\n"
    "ldr x12, [x15, #0x50]\n"
    "movprfx z31, z18\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ldr x11, [x15, #0x58]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "ldr x9, [x15, #0x60]\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ldr x25, [x15, #0x70]\n"
    "ld1w { z12.s }, p2/Z, [x23, x28, LSL #2]\n"
    "movprfx z30, z18\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "fmla z23.s, p3/M, z7.s, z11.s\n"
    "ldr x23, [x15, #0x68]\n"
    "fmla z24.s, p3/M, z0.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ldr x24, [x15, #0x78]\n"
    "fmla z26.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "ldr x14, [x15, #0x80]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "ldr x13, [x15, #0x88]\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z23.s, p3/M, z1.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x12, x28, LSL #2]\n"
    "ldr x12, [x15, #0x90]\n"
    "fmla z24.s, p3/M, z2.s, z12.s\n"
    "fmla z25.s, p3/M, z1.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x11, x28, LSL #2]\n"
    "ldr x11, [x15, #0x98]\n"
    "ld1w { z12.s }, p2/Z, [x9, x28, LSL #2]\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "fmla z30.s, p3/M, z2.s, z10.s\n"
    "ldr x9, [x15, #0xa0]\n"
    "fmla z26.s, p3/M, z0.s, z11.s\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "ldr x22, [x27, #0x0]\n"
    "fmla z24.s, p3/M, z8.s, z10.s\n"
    "fmla z25.s, p3/M, z7.s, z10.s\n"
    "ldr x21, [x27, #0x8]\n"
    "fmla z31.s, p3/M, z1.s, z10.s\n"
    "fmla z29.s, p3/M, z3.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x23, x28, LSL #2]\n"
    "ldr x23, [x15, #0xa8]\n"
    "fmla z26.s, p3/M, z6.s, z12.s\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "ld1w { z12.s }, p2/Z, [x14, x28, LSL #2]\n"
    "ldr x14, [x15, #0xc0]\n"
    "fmla z28.s, p3/M, z6.s, z10.s\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "ldr x20, [x27, #0x10]\n"
    "fmla z23.s, p3/M, z3.s, z11.s\n"
    "fmla z25.s, p3/M, z5.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x28, LSL #2]\n"
    "ldr x25, [x15, #0xb0]\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z3.s, z10.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x28, LSL #2]\n"
    "ldr x24, [x15, #0xb8]\n"
    "fmla z26.s, p3/M, z8.s, z10.s\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ldr x19, [x27, #0x18]\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "fmla z31.s, p3/M, z5.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x13, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x12, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "fmla z26.s, p3/M, z1.s, z12.s\n"
    "fmla z24.s, p3/M, z5.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x28, LSL #2]\n"
    "fmla z25.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z30.s, p3/M, z8.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "fmla z26.s, p3/M, z7.s, z12.s\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x25, x28, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x23, x28, LSL #2]\n"
    "fmla z23.s, p3/M, z6.s, z12.s\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "fmla z24.s, p3/M, z1.s, z11.s\n"
    "fmla z25.s, p3/M, z0.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmax z23.s, p3/M, z23.s, z17.s\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z30.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x14, x28, LSL #2]\n"
    "fmla z26.s, p3/M, z3.s, z12.s\n"
    "fmla z25.s, p3/M, z8.s, z11.s\n"
    "fmin z23.s, p3/M, z23.s, z16.s\n"
    "st1w { z23.s }, p1, [x22, x26, LSL #2]\n"
    "ldr x22, [x27, #0x20]\n"
    "fmla z28.s, p3/M, z5.s, z11.s\n"
    "fmla z29.s, p3/M, z8.s, z13.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "fmla z31.s, p3/M, z6.s, z13.s\n"
    ".inst 0xc1b0ca38  // fclamp { z24.s-z27.s }, z17.s, z16.s\n"
    "st1w { z24.s }, p1, [x21, x26, LSL #2]\n"
    "ldr x21, [x27, #0x28]\n"
    "st1w { z25.s }, p1, [x20, x26, LSL #2]\n"
    "ldr x20, [x27, #0x30]\n"
    ".inst 0xc1b0ca3c  // fclamp { z28.s-z31.s }, z17.s, z16.s\n"
    "st1w { z26.s }, p1, [x19, x26, LSL #2]\n"
    "ldr x19, [x27, #0x38]\n"
    "st1w { z27.s }, p1, [x22, x26, LSL #2]\n"
    "ldr x22, [x27, #0x40]\n"
    "st1w { z28.s }, p1, [x21, x26, LSL #2]\n"
    "st1w { z29.s }, p1, [x20, x26, LSL #2]\n"
    "st1w { z30.s }, p1, [x19, x26, LSL #2]\n"
    "st1w { z31.s }, p1, [x22, x26, LSL #2]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2)
