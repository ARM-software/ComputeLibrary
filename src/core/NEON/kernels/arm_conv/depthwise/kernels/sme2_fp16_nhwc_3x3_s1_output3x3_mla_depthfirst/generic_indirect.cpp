/*
 * Copyright (c) 2023-2024 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SME2) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace depthwise {

void sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst_indirect_impl(
  const __fp16 *const *const input_ptrs,
  __fp16 *const *const outptrs,
  const void *params,
  unsigned int n_channels,
  const __fp16 activation_min,
  const __fp16 activation_max
)
{
  struct Args
  {
    __fp16 *const *outptrs;
    const void *params;
    const __fp16 min, max;
    const __fp16 *inptrs[25];

    Args(
      const __fp16 *const *const input_ptrs,
      __fp16 *const *const outptrs,
      const void *const params,
      const __fp16 min,
      const __fp16 max
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
    "ldr x17, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "add x16, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "mov x15, #0x0\n"
    "ptrue p3.b\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "ldp x24, x23, [x16, #0x0]\n"
    "ldp x22, x21, [x16, #0x10]\n"
    "cnth x14\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1rh { z15.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ld1h { z30.h }, p3/Z, [x17]\n"
    "addvl x17, x17, #1\n"
    "ldr x20, [x16, #0x20]\n"
    "cmp x14, %x[n_channels]\n"
    ".inst 0xa040a220  // ld1h { z0.h-z3.h }, pn8.b/Z, [x17]\n"
    "addvl x17, x17, #4\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "sub x12, XZR, x14\n"
    ".inst 0xa040a224  // ld1h { z4.h-z7.h }, pn8.b/Z, [x17]\n"
    "addvl x17, x17, #4\n"
    "ld1rh { z14.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1h { z8.h }, p3/Z, [x17]\n"
    "addvl x17, x17, #1\n"
    "ld1h { z9.h }, p2/Z, [x24, x15, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x23, x15, LSL #1]\n"
    "ld1h { z11.h }, p2/Z, [x22, x15, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x21, x15, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x20, x15, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z31, z30\n fmla z31.h, p3/M, z8.h, z9.h\n"
    "movprfx z24, z30\n fmla z24.h, p3/M, z7.h, z9.h\n"
    "ldr x23, [x16, #0x30]\n"
    "inch x12\n"
    "movprfx z25, z30\n fmla z25.h, p3/M, z6.h, z9.h\n"
    "movprfx z26, z30\n fmla z26.h, p3/M, z5.h, z9.h\n"
    "ldr x27, [x16, #0x38]\n"
    "mov p1.b, p2.b\n"
    "movprfx z27, z30\n fmla z27.h, p3/M, z4.h, z9.h\n"
    "movprfx z20, z30\n fmla z20.h, p3/M, z3.h, z9.h\n"
    "ldr x22, [x16, #0x28]\n"
    "whilelt p0.h, x14, %x[n_channels]\n"
    "movprfx z21, z30\n fmla z21.h, p3/M, z2.h, z9.h\n"
    "movprfx z23, z30\n fmla z23.h, p3/M, z0.h, z9.h\n"
    "ldr x21, [x16, #0x48]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "fmla z24.h, p3/M, z4.h, z13.h\n"
    "ldr x20, [x16, #0x40]\n"
    "fmla z25.h, p3/M, z2.h, z11.h\n"
    "ld1h { z17.h }, p2/Z, [x23, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "ldr x26, [x16, #0x50]\n"
    "fmla z27.h, p3/M, z1.h, z13.h\n"
    "fmla z20.h, p3/M, z0.h, z13.h\n"
    "ld1h { z19.h }, p2/Z, [x21, x15, LSL #1]\n"
    "ldr x25, [x16, #0x58]\n"
    "fmla z21.h, p3/M, z6.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x22, x15, LSL #1]\n"
    "movprfx z22, z30\n fmla z22.h, p3/M, z1.h, z9.h\n"
    "ldr x24, [x16, #0x60]\n"
    "fmla z31.h, p3/M, z5.h, z13.h\n"
    "fmla z24.h, p3/M, z6.h, z17.h\n"
    "ldr x23, [x16, #0x68]\n"
    "ld1h { z30.h }, p3/Z, [x17]\n"
    "fmla z25.h, p3/M, z3.h, z13.h\n"
    "ld1h { z18.h }, p2/Z, [x27, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z4.h, z17.h\n"
    "ldr x22, [x16, #0x70]\n"
    "fmla z23.h, p3/M, z8.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x15, LSL #1]\n"
    "fmla z27.h, p3/M, z3.h, z17.h\n"
    "ldr x21, [x16, #0x78]\n"
    "fmla z22.h, p3/M, z0.h, z17.h\n"
    "fmla z20.h, p3/M, z4.h, z19.h\n"
    "ldr x20, [x16, #0x80]\n"
    "addvl x17, x17, #1\n"
    "fmla z31.h, p3/M, z7.h, z17.h\n"
    "fmla z24.h, p3/M, z0.h, z18.h\n"
    "ldr x11, [x16, #0x88]\n"
    "fmla z21.h, p3/M, z1.h, z17.h\n"
    "fmla z25.h, p3/M, z1.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x15, LSL #1]\n"
    "ldr x10, [x16, #0x90]\n"
    "fmla z27.h, p3/M, z5.h, z19.h\n"
    "fmla z23.h, p3/M, z1.h, z19.h\n"
    "ldr x9, [x13, #0x0]\n"
    "fmla z22.h, p3/M, z2.h, z19.h\n"
    "ldr x28, [x13, #0x8]\n"
    "fmla z31.h, p3/M, z1.h, z18.h\n"
    "fmla z24.h, p3/M, z2.h, z16.h\n"
    "ld1h { z9.h }, p2/Z, [x25, x15, LSL #1]\n"
    "ldr x27, [x16, #0x98]\n"
    "ld1h { z16.h }, p2/Z, [x24, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z0.h, z17.h\n"
    "fmla z25.h, p3/M, z7.h, z19.h\n"
    "ldr x24, [x16, #0xa0]\n"
    "ldr x26, [x13, #0x10]\n"
    "fmla z20.h, p3/M, z2.h, z9.h\n"
    "ldr x25, [x13, #0x18]\n"
    "fmla z24.h, p3/M, z8.h, z19.h\n"
    "fmla z21.h, p3/M, z3.h, z16.h\n"
    "ld1h { z29.h }, p2/Z, [x23, x15, LSL #1]\n"
    "ldr x23, [x16, #0xa8]\n"
    "fmla z26.h, p3/M, z6.h, z16.h\n"
    "fmla z31.h, p3/M, z3.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x22, x15, LSL #1]\n"
    "ldr x22, [x16, #0xb0]\n"
    "fmla z25.h, p3/M, z5.h, z9.h\n"
    "ld1h { z16.h }, p2/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xb8]\n"
    "fmla z27.h, p3/M, z7.h, z29.h\n"
    "fmla z20.h, p3/M, z6.h, z29.h\n"
    "ld1h { z17.h }, p2/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xc0]\n"
    "fmla z22.h, p3/M, z4.h, z29.h\n"
    "fmla z21.h, p3/M, z5.h, z29.h\n"
    "fmla z23.h, p3/M, z3.h, z29.h\n"
    "fmla z26.h, p3/M, z8.h, z29.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "fmla z31.h, p3/M, z4.h, z17.h\n"
    "fmla z20.h, p3/M, z8.h, z18.h\n"
    "fmla z27.h, p3/M, z0.h, z17.h\n"
    "fmla z22.h, p3/M, z6.h, z16.h\n"
    "fmla z21.h, p3/M, z7.h, z16.h\n"
    "ld1h { z13.h }, p2/Z, [x10, x15, LSL #1]\n"
    "fmla z23.h, p3/M, z5.h, z18.h\n"
    "ld1h { z16.h }, p2/Z, [x11, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z1.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x15, LSL #1]\n"
    "fmla z24.h, p3/M, z5.h, z16.h\n"
    "fmla z25.h, p3/M, z4.h, z16.h\n"
    "fmla z27.h, p3/M, z2.h, z16.h\n"
    "fmla z20.h, p3/M, z1.h, z16.h\n"
    "ld1h { z28.h }, p2/Z, [x24, x15, LSL #1]\n"
    "ldr x24, [x16, #0x20]\n"
    "fmla z22.h, p3/M, z8.h, z13.h\n"
    "fmla z26.h, p3/M, z7.h, z17.h\n"
    "fmla z21.h, p3/M, z4.h, z17.h\n"
    "fmla z23.h, p3/M, z7.h, z13.h\n"
    "ld1h { z16.h }, p2/Z, [x23, x15, LSL #1]\n"
    "fmla z31.h, p3/M, z2.h, z28.h\n"
    "fmla z24.h, p3/M, z1.h, z28.h\n"
    "fmla z27.h, p3/M, z6.h, z17.h\n"
    "fmla z25.h, p3/M, z0.h, z28.h\n"
    "ld1h { z18.h }, p2/Z, [x21, x15, LSL #1]\n"
    "fmla z22.h, p3/M, z3.h, z17.h\n"
    "ld1h { z17.h }, p2/Z, [x22, x15, LSL #1]\n"
    "fmla z20.h, p3/M, z7.h, z16.h\n"
    "fmla z23.h, p3/M, z4.h, z16.h\n"
    "fmla z31.h, p3/M, z6.h, z17.h\n"
    "fmla z21.h, p3/M, z0.h, z17.h\n"
    "fmla z22.h, p3/M, z5.h, z16.h\n"
    "fmla z27.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x15, LSL #1]\n"
    "ldp x23, x22, [x16, #0x0]\n"
    "fmla z23.h, p3/M, z2.h, z18.h\n"
    "fmla z26.h, p3/M, z3.h, z17.h\n"
    "ldp x21, x20, [x16, #0x10]\n"
    "inch x15\n"
    "fmla z25.h, p3/M, z8.h, z18.h\n"
    "fmla z20.h, p3/M, z5.h, z18.h\n"
    ".inst 0xa040a220  // ld1h { z0.h-z3.h }, pn8.b/Z, [x17]\n"
    "addvl x17, x17, #4\n"
    "fmax z31.h, p3/M, z31.h, z15.h\n"
    "fmla z21.h, p3/M, z8.h, z16.h\n"
    "ld1h { z9.h }, p0/Z, [x23, x14, LSL #1]\n"
    "whilelt p2.h, x15, %x[n_channels]\n"
    "fmla z22.h, p3/M, z7.h, z16.h\n"
    "ld1h { z10.h }, p0/Z, [x22, x14, LSL #1]\n"
    "fmla z23.h, p3/M, z6.h, z16.h\n"
    "ld1h { z11.h }, p0/Z, [x21, x14, LSL #1]\n"
    ".inst 0xc16ec9f8  // fclamp { z24.h-z27.h }, z15.h, z14.h\n"
    "ld1h { z12.h }, p0/Z, [x20, x14, LSL #1]\n"
    "fmin z31.h, p3/M, z31.h, z14.h\n"
    "ld1h { z13.h }, p0/Z, [x24, x14, LSL #1]\n"
    "inch x14\n"
    ".inst 0xa040a224  // ld1h { z4.h-z7.h }, pn8.b/Z, [x17]\n"
    "addvl x17, x17, #4\n"
    "cmp x14, %x[n_channels]\n"
    ".inst 0xc16ec9f4  // fclamp { z20.h-z23.h }, z15.h, z14.h\n"
    "ld1h { z8.h }, p3/Z, [x17]\n"
    "addvl x17, x17, #1\n"
    "st1h { z24.h }, p1, [x28, x12, LSL #1]\n"
    "ldr x23, [x13, #0x28]\n"
    "st1h { z31.h }, p1, [x9, x12, LSL #1]\n"
    "ldr x20, [x13, #0x20]\n"
    "st1h { z25.h }, p1, [x26, x12, LSL #1]\n"
    "ldr x22, [x13, #0x30]\n"
    "st1h { z26.h }, p1, [x25, x12, LSL #1]\n"
    "ldr x21, [x13, #0x38]\n"
    "st1h { z27.h }, p1, [x20, x12, LSL #1]\n"
    "ldr x20, [x13, #0x40]\n"
    "st1h { z20.h }, p1, [x23, x12, LSL #1]\n"
    "st1h { z21.h }, p1, [x22, x12, LSL #1]\n"
    "st1h { z22.h }, p1, [x21, x12, LSL #1]\n"
    "st1h { z23.h }, p1, [x20, x12, LSL #1]\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z20, z30\n fmla z20.h, p3/M, z8.h, z9.h\n"
    "movprfx z24, z30\n fmla z24.h, p3/M, z7.h, z9.h\n"
    "ldr x23, [x16, #0x30]\n"
    "inch x12\n"
    "movprfx z25, z30\n fmla z25.h, p3/M, z6.h, z9.h\n"
    "movprfx z26, z30\n fmla z26.h, p3/M, z5.h, z9.h\n"
    "ldr x27, [x16, #0x38]\n"
    "mov p0.b, p2.b\n"
    "movprfx z27, z30\n fmla z27.h, p3/M, z4.h, z9.h\n"
    "movprfx z28, z30\n fmla z28.h, p3/M, z3.h, z9.h\n"
    "ldr x22, [x16, #0x28]\n"
    "movprfx z29, z30\n fmla z29.h, p3/M, z2.h, z9.h\n"
    "movprfx z31, z30\n fmla z31.h, p3/M, z0.h, z9.h\n"
    "ldr x21, [x16, #0x48]\n"
    "fmla z20.h, p3/M, z0.h, z10.h\n"
    "fmla z24.h, p3/M, z4.h, z13.h\n"
    "ldr x20, [x16, #0x40]\n"
    "fmla z25.h, p3/M, z2.h, z11.h\n"
    "ld1h { z19.h }, p2/Z, [x23, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z2.h, z13.h\n"
    "ldr x26, [x16, #0x50]\n"
    "fmla z27.h, p3/M, z1.h, z13.h\n"
    "fmla z28.h, p3/M, z0.h, z13.h\n"
    "ld1h { z18.h }, p2/Z, [x21, x15, LSL #1]\n"
    "ldr x25, [x16, #0x58]\n"
    "fmla z29.h, p3/M, z6.h, z12.h\n"
    "ld1h { z16.h }, p2/Z, [x22, x15, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z9.h\n"
    "ldr x24, [x16, #0x60]\n"
    "fmla z20.h, p3/M, z5.h, z13.h\n"
    "fmla z24.h, p3/M, z6.h, z19.h\n"
    "ldr x23, [x16, #0x68]\n"
    "fmla z25.h, p3/M, z3.h, z13.h\n"
    "ld1h { z17.h }, p2/Z, [x27, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z4.h, z19.h\n"
    "ldr x22, [x16, #0x70]\n"
    "fmla z31.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x15, LSL #1]\n"
    "fmla z27.h, p3/M, z3.h, z19.h\n"
    "ldr x21, [x16, #0x78]\n"
    "fmla z30.h, p3/M, z0.h, z19.h\n"
    "fmla z28.h, p3/M, z4.h, z18.h\n"
    "ldr x20, [x16, #0x80]\n"
    "fmla z20.h, p3/M, z7.h, z19.h\n"
    "fmla z24.h, p3/M, z0.h, z17.h\n"
    "ldr x11, [x16, #0x88]\n"
    "fmla z29.h, p3/M, z1.h, z19.h\n"
    "fmla z25.h, p3/M, z1.h, z16.h\n"
    "ld1h { z19.h }, p2/Z, [x26, x15, LSL #1]\n"
    "ldr x10, [x16, #0x90]\n"
    "fmla z27.h, p3/M, z5.h, z18.h\n"
    "fmla z31.h, p3/M, z1.h, z18.h\n"
    "ldr x9, [x13, #0x0]\n"
    "fmla z30.h, p3/M, z2.h, z18.h\n"
    "ldr x28, [x13, #0x8]\n"
    "fmla z20.h, p3/M, z1.h, z17.h\n"
    "fmla z24.h, p3/M, z2.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x25, x15, LSL #1]\n"
    "ldr x27, [x16, #0x98]\n"
    "ld1h { z16.h }, p2/Z, [x24, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z0.h, z19.h\n"
    "fmla z25.h, p3/M, z7.h, z18.h\n"
    "ldr x26, [x16, #0xa0]\n"
    "ldr x25, [x13, #0x10]\n"
    "fmla z28.h, p3/M, z2.h, z17.h\n"
    "ldr x24, [x13, #0x18]\n"
    "fmla z24.h, p3/M, z8.h, z18.h\n"
    "fmla z29.h, p3/M, z3.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x23, x15, LSL #1]\n"
    "ldr x23, [x16, #0xa8]\n"
    "fmla z26.h, p3/M, z6.h, z16.h\n"
    "fmla z20.h, p3/M, z3.h, z19.h\n"
    "ld1h { z19.h }, p2/Z, [x22, x15, LSL #1]\n"
    "ldr x22, [x16, #0xb0]\n"
    "fmla z25.h, p3/M, z5.h, z17.h\n"
    "ld1h { z16.h }, p2/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xb8]\n"
    "fmla z27.h, p3/M, z7.h, z18.h\n"
    "fmla z28.h, p3/M, z6.h, z18.h\n"
    "ld1h { z17.h }, p2/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xc0]\n"
    "fmla z30.h, p3/M, z4.h, z18.h\n"
    "fmla z29.h, p3/M, z5.h, z18.h\n"
    "fmla z31.h, p3/M, z3.h, z18.h\n"
    "fmla z26.h, p3/M, z8.h, z18.h\n"
    "fmla z24.h, p3/M, z3.h, z17.h\n"
    "fmla z20.h, p3/M, z4.h, z17.h\n"
    "fmla z28.h, p3/M, z8.h, z19.h\n"
    "fmla z27.h, p3/M, z0.h, z17.h\n"
    "fmla z30.h, p3/M, z6.h, z16.h\n"
    "fmla z29.h, p3/M, z7.h, z16.h\n"
    "ld1h { z18.h }, p2/Z, [x10, x15, LSL #1]\n"
    "fmla z31.h, p3/M, z5.h, z19.h\n"
    "ld1h { z16.h }, p2/Z, [x11, x15, LSL #1]\n"
    "fmla z26.h, p3/M, z1.h, z17.h\n"
    "ld1h { z19.h }, p2/Z, [x27, x15, LSL #1]\n"
    "fmla z24.h, p3/M, z5.h, z16.h\n"
    "fmla z25.h, p3/M, z4.h, z16.h\n"
    "fmla z27.h, p3/M, z2.h, z16.h\n"
    "fmla z28.h, p3/M, z1.h, z16.h\n"
    "ld1h { z17.h }, p2/Z, [x26, x15, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z18.h\n"
    "fmla z26.h, p3/M, z7.h, z19.h\n"
    "fmla z29.h, p3/M, z4.h, z19.h\n"
    "fmla z31.h, p3/M, z7.h, z18.h\n"
    "ld1h { z16.h }, p2/Z, [x23, x15, LSL #1]\n"
    "fmla z20.h, p3/M, z2.h, z17.h\n"
    "fmla z24.h, p3/M, z1.h, z17.h\n"
    "fmla z27.h, p3/M, z6.h, z19.h\n"
    "fmla z25.h, p3/M, z0.h, z17.h\n"
    "ld1h { z18.h }, p2/Z, [x21, x15, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z19.h\n"
    "ld1h { z17.h }, p2/Z, [x22, x15, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z16.h\n"
    "fmla z31.h, p3/M, z4.h, z16.h\n"
    "fmla z20.h, p3/M, z6.h, z17.h\n"
    "fmla z29.h, p3/M, z0.h, z17.h\n"
    "fmla z30.h, p3/M, z5.h, z16.h\n"
    "fmla z27.h, p3/M, z8.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x15, LSL #1]\n"
    "fmla z31.h, p3/M, z2.h, z18.h\n"
    "fmla z26.h, p3/M, z3.h, z17.h\n"
    "fmla z25.h, p3/M, z8.h, z18.h\n"
    "fmla z28.h, p3/M, z5.h, z18.h\n"
    "fmax z20.h, p3/M, z20.h, z15.h\n"
    "fmla z29.h, p3/M, z8.h, z16.h\n"
    "fmla z30.h, p3/M, z7.h, z16.h\n"
    "fmla z31.h, p3/M, z6.h, z16.h\n"
    ".inst 0xc16ec9f8  // fclamp { z24.h-z27.h }, z15.h, z14.h\n"
    "fmin z20.h, p3/M, z20.h, z14.h\n"
    ".inst 0xc16ec9fc  // fclamp { z28.h-z31.h }, z15.h, z14.h\n"
    "st1h { z24.h }, p0, [x28, x12, LSL #1]\n"
    "ldr x23, [x13, #0x28]\n"
    "st1h { z20.h }, p0, [x9, x12, LSL #1]\n"
    "ldr x20, [x13, #0x20]\n"
    "st1h { z25.h }, p0, [x25, x12, LSL #1]\n"
    "ldr x22, [x13, #0x30]\n"
    "st1h { z26.h }, p0, [x24, x12, LSL #1]\n"
    "ldr x21, [x13, #0x38]\n"
    "st1h { z27.h }, p0, [x20, x12, LSL #1]\n"
    "ldr x20, [x13, #0x40]\n"
    "st1h { z28.h }, p0, [x23, x12, LSL #1]\n"
    "st1h { z29.h }, p0, [x22, x12, LSL #1]\n"
    "st1h { z30.h }, p0, [x21, x12, LSL #1]\n"
    "st1h { z31.h }, p0, [x20, x12, LSL #1]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME2) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
