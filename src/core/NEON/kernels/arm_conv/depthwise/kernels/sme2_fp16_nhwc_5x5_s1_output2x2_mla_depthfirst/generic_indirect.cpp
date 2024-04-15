/*
 * Copyright (c) 2023 Arm Limited.
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

void sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl(
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
    const __fp16 *inptrs[36];

    Args(
      const __fp16 *const *const input_ptrs,
      __fp16 *const *const outptrs,
      const void *const params,
      const __fp16 min,
      const __fp16 max
    ) : outptrs(outptrs), params(params), min(min), max(max)
    {
      inptrs[0] = input_ptrs[0];
      inptrs[1] = input_ptrs[1];
      inptrs[2] = input_ptrs[6];
      inptrs[3] = input_ptrs[7];
      inptrs[4] = input_ptrs[2];
      inptrs[5] = input_ptrs[8];
      inptrs[6] = input_ptrs[3];
      inptrs[7] = input_ptrs[4];
      inptrs[8] = input_ptrs[11];
      inptrs[9] = input_ptrs[12];
      inptrs[10] = input_ptrs[9];
      inptrs[11] = input_ptrs[10];
      inptrs[12] = input_ptrs[5];
      inptrs[13] = input_ptrs[13];
      inptrs[14] = input_ptrs[14];
      inptrs[15] = input_ptrs[15];
      inptrs[16] = input_ptrs[16];
      inptrs[17] = input_ptrs[17];
      inptrs[18] = input_ptrs[18];
      inptrs[19] = input_ptrs[19];
      inptrs[20] = input_ptrs[20];
      inptrs[21] = input_ptrs[21];
      inptrs[22] = input_ptrs[22];
      inptrs[23] = input_ptrs[23];
      inptrs[24] = input_ptrs[24];
      inptrs[25] = input_ptrs[25];
      inptrs[26] = input_ptrs[26];
      inptrs[27] = input_ptrs[27];
      inptrs[28] = input_ptrs[28];
      inptrs[29] = input_ptrs[29];
      inptrs[30] = input_ptrs[30];
      inptrs[31] = input_ptrs[31];
      inptrs[32] = input_ptrs[32];
      inptrs[33] = input_ptrs[33];
      inptrs[34] = input_ptrs[34];
      inptrs[35] = input_ptrs[35];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ldr x20, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "add x16, %x[params_struct], %[offsetof_Args_inptrs]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x15, #0x0\n"
    "ldr x14, [%x[params_struct], %[offsetof_args_params]]\n"
    ".inst 0x25207810  // ptrue pn8.b\n"
    "whilelt p3.h, XZR, %x[n_channels]\n"
    "ptrue p2.b\n"
    "cnth x13\n"
    "ldp x12, x11, [x20, #0x0]\n"
    "ldp x10, x9, [x20, #0x10]\n"
    "cmp x13, %x[n_channels]\n"
    "ld1rh { z18.h }, p2/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "sub x28, XZR, x13\n"
    "ldp x27, x26, [x16, #0x0]\n"
    "ld1h { z17.h }, p2/Z, [x14]\n"
    "addvl x14, x14, #1\n"
    "ldp x25, x24, [x16, #0x10]\n"
    ".inst 0xa040a1c0  // ld1h { z0.h-z3.h }, pn8.b/Z, [x14]\n"
    "addvl x14, x14, #4\n"
    "ldp x23, x22, [x16, #0x20]\n"
    "ld1rh { z16.h }, p2/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1h { z5.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ldp x21, x20, [x16, #0x30]\n"
    "ld1h { z6.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldp x27, x26, [x16, #0x40]\n"
    "ld1h { z4.h }, p2/Z, [x14]\n"
    "addvl x14, x14, #1\n"
    "ld1h { z7.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ld1h { z8.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ld1h { z9.h }, p3/Z, [x23, x15, LSL #1]\n"
    "ld1h { z13.h }, p3/Z, [x22, x15, LSL #1]\n"
    "ld1h { z11.h }, p3/Z, [x21, x15, LSL #1]\n"
    "ld1h { z12.h }, p3/Z, [x20, x15, LSL #1]\n"
    "ld1h { z10.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ld1h { z14.h }, p3/Z, [x26, x15, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z28, z17\n fmla z28.h, p2/M, z0.h, z5.h\n"
    "movprfx z29, z17\n fmla z29.h, p2/M, z0.h, z6.h\n"
    "ldr x25, [x16, #0x50]\n"
    "whilelt p1.h, x13, %x[n_channels]\n"
    "movprfx z30, z17\n fmla z30.h, p2/M, z0.h, z7.h\n"
    "movprfx z31, z17\n fmla z31.h, p2/M, z0.h, z8.h\n"
    "ldr x24, [x16, #0x58]\n"
    "ld1h { z0.h }, p2/Z, [x14]\n"
    "ldr x23, [x16, #0x60]\n"
    "inch x28\n"
    "mov p0.b, p3.b\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x22, [x16, #0x68]\n"
    "fmla z28.h, p2/M, z1.h, z6.h\n"
    "fmla z29.h, p2/M, z1.h, z9.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x21, [x16, #0x70]\n"
    "fmla z30.h, p2/M, z1.h, z8.h\n"
    "fmla z31.h, p2/M, z1.h, z13.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #1, MUL VL]\n"
    "ldr x20, [x16, #0x78]\n"
    "ldr x27, [x16, #0x80]\n"
    "fmla z28.h, p2/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x23, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z2.h, z11.h\n"
    "ldr x26, [x16, #0x88]\n"
    "fmla z30.h, p2/M, z2.h, z13.h\n"
    "fmla z31.h, p2/M, z2.h, z5.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #2, MUL VL]\n"
    "ldr x25, [x16, #0x90]\n"
    "ldr x24, [x16, #0x98]\n"
    "fmla z28.h, p2/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x22, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z3.h, z12.h\n"
    "ldr x23, [x16, #0xa0]\n"
    "fmla z30.h, p2/M, z3.h, z5.h\n"
    "fmla z31.h, p2/M, z3.h, z6.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #3, MUL VL]\n"
    "ldr x22, [x16, #0xa8]\n"
    "fmla z28.h, p2/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xb0]\n"
    "fmla z29.h, p2/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xb8]\n"
    "fmla z30.h, p2/M, z4.h, z6.h\n"
    "fmla z31.h, p2/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #4, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z7.h\n"
    "fmla z29.h, p2/M, z0.h, z8.h\n"
    "fmla z30.h, p2/M, z0.h, z14.h\n"
    "fmla z31.h, p2/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #5, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z8.h\n"
    "ld1h { z8.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldr x26, [x16, #0xc8]\n"
    "fmla z29.h, p2/M, z1.h, z13.h\n"
    "fmla z30.h, p2/M, z1.h, z11.h\n"
    "fmla z31.h, p2/M, z1.h, z12.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #6, MUL VL]\n"
    "fmla z28.h, p2/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ldr x27, [x16, #0xc0]\n"
    "fmla z29.h, p2/M, z2.h, z5.h\n"
    "fmla z30.h, p2/M, z2.h, z12.h\n"
    "fmla z31.h, p2/M, z2.h, z9.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "ld1h { z17.h }, p2/Z, [x14, #4, MUL VL]\n"
    "fmla z28.h, p2/M, z3.h, z5.h\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x25, [x16, #0xd0]\n"
    "fmla z29.h, p2/M, z3.h, z6.h\n"
    "fmla z30.h, p2/M, z3.h, z9.h\n"
    "fmla z31.h, p2/M, z3.h, z13.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #-8, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z6.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x24, [x16, #0xd8]\n"
    "fmla z29.h, p2/M, z4.h, z10.h\n"
    "ld1h { z10.h }, p3/Z, [x23, x15, LSL #1]\n"
    "ldr x23, [x16, #0xe0]\n"
    "fmla z30.h, p2/M, z4.h, z13.h\n"
    "fmla z31.h, p2/M, z4.h, z8.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #-7, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z14.h\n"
    "ld1h { z14.h }, p3/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xf8]\n"
    "fmla z29.h, p2/M, z0.h, z11.h\n"
    "fmla z30.h, p2/M, z0.h, z5.h\n"
    "fmla z31.h, p2/M, z0.h, z6.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #-6, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x22, x15, LSL #1]\n"
    "ldr x22, [x16, #0xe8]\n"
    "fmla z29.h, p2/M, z1.h, z12.h\n"
    "fmla z30.h, p2/M, z1.h, z6.h\n"
    "fmla z31.h, p2/M, z1.h, z10.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #-5, MUL VL]\n"
    "fmla z28.h, p2/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xf0]\n"
    "fmla z29.h, p2/M, z2.h, z9.h\n"
    "fmla z30.h, p2/M, z2.h, z10.h\n"
    "fmla z31.h, p2/M, z2.h, z11.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #-4, MUL VL]\n"
    "fmla z28.h, p2/M, z3.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ldr x27, [x16, #0x100]\n"
    "fmla z29.h, p2/M, z3.h, z13.h\n"
    "fmla z30.h, p2/M, z3.h, z11.h\n"
    "fmla z31.h, p2/M, z3.h, z12.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #-3, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldr x26, [x16, #0x108]\n"
    "fmla z29.h, p2/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p3/Z, [x23, x15, LSL #1]\n"
    "fmla z30.h, p2/M, z4.h, z12.h\n"
    "fmla z31.h, p2/M, z4.h, z14.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #-2, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z5.h\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x25, [x16, #0x110]\n"
    "fmla z29.h, p2/M, z0.h, z6.h\n"
    "fmla z30.h, p2/M, z0.h, z9.h\n"
    "fmla z31.h, p2/M, z0.h, z13.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #-1, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x24, [x16, #0x118]\n"
    "fmla z29.h, p2/M, z1.h, z10.h\n"
    "fmla z30.h, p2/M, z1.h, z13.h\n"
    "fmla z31.h, p2/M, z1.h, z5.h\n"
    "ld1h { z1.h }, p2/Z, [x14]\n"
    "fmla z28.h, p2/M, z2.h, z10.h\n"
    "ld1h { z10.h }, p3/Z, [x22, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z2.h, z11.h\n"
    "fmla z30.h, p2/M, z2.h, z5.h\n"
    "fmla z31.h, p2/M, z2.h, z6.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #1, MUL VL]\n"
    "fmla z28.h, p2/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x21, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z3.h, z12.h\n"
    "fmla z30.h, p2/M, z3.h, z6.h\n"
    "fmla z31.h, p2/M, z3.h, z8.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #2, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x20, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z4.h, z14.h\n"
    "fmla z30.h, p2/M, z4.h, z8.h\n"
    "fmla z31.h, p2/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #3, MUL VL]\n"
    "addvl x14, x14, #5\n"
    "fmla z28.h, p2/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x27, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z0.h, z13.h\n"
    "fmla z30.h, p2/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldp x27, x26, [x16, #0x0]\n"
    "fmla z31.h, p2/M, z0.h, z12.h\n"
    "fmla z28.h, p2/M, z1.h, z13.h\n"
    "fmla z29.h, p2/M, z1.h, z5.h\n"
    "fmla z30.h, p2/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x25, x15, LSL #1]\n"
    "fmla z31.h, p2/M, z1.h, z9.h\n"
    "fmla z28.h, p2/M, z2.h, z5.h\n"
    "ld1h { z5.h }, p1/Z, [x27, x13, LSL #1]\n"
    "fmla z29.h, p2/M, z2.h, z6.h\n"
    "fmla z30.h, p2/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldp x25, x24, [x16, #0x10]\n"
    "ldp x23, x22, [x16, #0x20]\n"
    "inch x15\n"
    "ldp x21, x20, [x16, #0x30]\n"
    "whilelt p3.h, x15, %x[n_channels]\n"
    "fmla z31.h, p2/M, z2.h, z11.h\n"
    "fmla z28.h, p2/M, z3.h, z6.h\n"
    "ld1h { z6.h }, p1/Z, [x26, x13, LSL #1]\n"
    "ldp x27, x26, [x16, #0x40]\n"
    "fmla z29.h, p2/M, z3.h, z8.h\n"
    "fmla z30.h, p2/M, z3.h, z11.h\n"
    "ld1h { z7.h }, p1/Z, [x25, x13, LSL #1]\n"
    "ld1h { z13.h }, p1/Z, [x22, x13, LSL #1]\n"
    "fmla z31.h, p2/M, z3.h, z12.h\n"
    "fmla z28.h, p2/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p1/Z, [x24, x13, LSL #1]\n"
    "fmla z29.h, p2/M, z4.h, z10.h\n"
    "fmla z30.h, p2/M, z4.h, z12.h\n"
    "ld1h { z11.h }, p1/Z, [x21, x13, LSL #1]\n"
    "ld1h { z12.h }, p1/Z, [x20, x13, LSL #1]\n"
    "fmla z31.h, p2/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p1/Z, [x23, x13, LSL #1]\n"
    "ld1h { z10.h }, p1/Z, [x27, x13, LSL #1]\n"
    "ld1h { z14.h }, p1/Z, [x26, x13, LSL #1]\n"
    "inch x13\n"
    ".inst 0xa040a1c0  // ld1h { z0.h-z3.h }, pn8.b/Z, [x14]\n"
    "addvl x14, x14, #4\n"
    "cmp x13, %x[n_channels]\n"
    ".inst 0xc170ca5c  // fclamp { z28.h-z31.h }, z18.h, z16.h\n"
    "ld1h { z4.h }, p2/Z, [x14]\n"
    "addvl x14, x14, #1\n"
    "st1h { z28.h }, p0, [x12, x28, LSL #1]\n"
    "st1h { z29.h }, p0, [x11, x28, LSL #1]\n"
    "st1h { z30.h }, p0, [x10, x28, LSL #1]\n"
    "st1h { z31.h }, p0, [x9, x28, LSL #1]\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z28, z17\n fmla z28.h, p2/M, z0.h, z5.h\n"
    "movprfx z29, z17\n fmla z29.h, p2/M, z0.h, z6.h\n"
    "ldr x25, [x16, #0x50]\n"
    "inch x28\n"
    "movprfx z30, z17\n fmla z30.h, p2/M, z0.h, z7.h\n"
    "movprfx z31, z17\n fmla z31.h, p2/M, z0.h, z8.h\n"
    "ldr x24, [x16, #0x58]\n"
    "ld1h { z0.h }, p2/Z, [x14]\n"
    "ldr x23, [x16, #0x60]\n"
    "mov p0.b, p3.b\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x22, [x16, #0x68]\n"
    "fmla z28.h, p2/M, z1.h, z6.h\n"
    "fmla z29.h, p2/M, z1.h, z9.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x21, [x16, #0x70]\n"
    "fmla z30.h, p2/M, z1.h, z8.h\n"
    "fmla z31.h, p2/M, z1.h, z13.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #1, MUL VL]\n"
    "ldr x20, [x16, #0x78]\n"
    "ldr x27, [x16, #0x80]\n"
    "fmla z28.h, p2/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x23, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z2.h, z11.h\n"
    "ldr x26, [x16, #0x88]\n"
    "fmla z30.h, p2/M, z2.h, z13.h\n"
    "fmla z31.h, p2/M, z2.h, z5.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #2, MUL VL]\n"
    "ldr x25, [x16, #0x90]\n"
    "ldr x24, [x16, #0x98]\n"
    "fmla z28.h, p2/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x22, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z3.h, z12.h\n"
    "ldr x23, [x16, #0xa0]\n"
    "fmla z30.h, p2/M, z3.h, z5.h\n"
    "fmla z31.h, p2/M, z3.h, z6.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #3, MUL VL]\n"
    "ldr x22, [x16, #0xa8]\n"
    "fmla z28.h, p2/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xb0]\n"
    "fmla z29.h, p2/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xb8]\n"
    "fmla z30.h, p2/M, z4.h, z6.h\n"
    "fmla z31.h, p2/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #4, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z7.h\n"
    "fmla z29.h, p2/M, z0.h, z8.h\n"
    "fmla z30.h, p2/M, z0.h, z14.h\n"
    "fmla z31.h, p2/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #5, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z8.h\n"
    "ld1h { z8.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldr x26, [x16, #0xc8]\n"
    "fmla z29.h, p2/M, z1.h, z13.h\n"
    "fmla z30.h, p2/M, z1.h, z11.h\n"
    "fmla z31.h, p2/M, z1.h, z12.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #6, MUL VL]\n"
    "fmla z28.h, p2/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ldr x27, [x16, #0xc0]\n"
    "fmla z29.h, p2/M, z2.h, z5.h\n"
    "fmla z30.h, p2/M, z2.h, z12.h\n"
    "fmla z31.h, p2/M, z2.h, z9.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #7, MUL VL]\n"
    "addvl x14, x14, #16\n"
    "fmla z28.h, p2/M, z3.h, z5.h\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x25, [x16, #0xd0]\n"
    "fmla z29.h, p2/M, z3.h, z6.h\n"
    "fmla z30.h, p2/M, z3.h, z9.h\n"
    "fmla z31.h, p2/M, z3.h, z13.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #-8, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z6.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x24, [x16, #0xd8]\n"
    "fmla z29.h, p2/M, z4.h, z10.h\n"
    "ld1h { z10.h }, p3/Z, [x23, x15, LSL #1]\n"
    "ldr x23, [x16, #0xe0]\n"
    "fmla z30.h, p2/M, z4.h, z13.h\n"
    "fmla z31.h, p2/M, z4.h, z8.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #-7, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z14.h\n"
    "ld1h { z14.h }, p3/Z, [x20, x15, LSL #1]\n"
    "ldr x20, [x16, #0xf8]\n"
    "fmla z29.h, p2/M, z0.h, z11.h\n"
    "fmla z30.h, p2/M, z0.h, z5.h\n"
    "fmla z31.h, p2/M, z0.h, z6.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #-6, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x22, x15, LSL #1]\n"
    "ldr x22, [x16, #0xe8]\n"
    "fmla z29.h, p2/M, z1.h, z12.h\n"
    "fmla z30.h, p2/M, z1.h, z6.h\n"
    "fmla z31.h, p2/M, z1.h, z10.h\n"
    "ld1h { z1.h }, p2/Z, [x14, #-5, MUL VL]\n"
    "fmla z28.h, p2/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x21, x15, LSL #1]\n"
    "ldr x21, [x16, #0xf0]\n"
    "fmla z29.h, p2/M, z2.h, z9.h\n"
    "fmla z30.h, p2/M, z2.h, z10.h\n"
    "fmla z31.h, p2/M, z2.h, z11.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #-4, MUL VL]\n"
    "fmla z28.h, p2/M, z3.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x27, x15, LSL #1]\n"
    "ldr x27, [x16, #0x100]\n"
    "fmla z29.h, p2/M, z3.h, z13.h\n"
    "fmla z30.h, p2/M, z3.h, z11.h\n"
    "fmla z31.h, p2/M, z3.h, z12.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #-3, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p3/Z, [x26, x15, LSL #1]\n"
    "ldr x26, [x16, #0x108]\n"
    "fmla z29.h, p2/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p3/Z, [x23, x15, LSL #1]\n"
    "fmla z30.h, p2/M, z4.h, z12.h\n"
    "fmla z31.h, p2/M, z4.h, z14.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #-2, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z5.h\n"
    "ld1h { z5.h }, p3/Z, [x25, x15, LSL #1]\n"
    "ldr x25, [x16, #0x110]\n"
    "fmla z29.h, p2/M, z0.h, z6.h\n"
    "fmla z30.h, p2/M, z0.h, z9.h\n"
    "fmla z31.h, p2/M, z0.h, z13.h\n"
    "ld1h { z0.h }, p2/Z, [x14, #-1, MUL VL]\n"
    "fmla z28.h, p2/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p3/Z, [x24, x15, LSL #1]\n"
    "ldr x24, [x16, #0x118]\n"
    "fmla z29.h, p2/M, z1.h, z10.h\n"
    "fmla z30.h, p2/M, z1.h, z13.h\n"
    "fmla z31.h, p2/M, z1.h, z5.h\n"
    "ld1h { z1.h }, p2/Z, [x14]\n"
    "fmla z28.h, p2/M, z2.h, z10.h\n"
    "ld1h { z10.h }, p3/Z, [x22, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z2.h, z11.h\n"
    "fmla z30.h, p2/M, z2.h, z5.h\n"
    "fmla z31.h, p2/M, z2.h, z6.h\n"
    "ld1h { z2.h }, p2/Z, [x14, #1, MUL VL]\n"
    "fmla z28.h, p2/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x21, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z3.h, z12.h\n"
    "fmla z30.h, p2/M, z3.h, z6.h\n"
    "fmla z31.h, p2/M, z3.h, z8.h\n"
    "ld1h { z3.h }, p2/Z, [x14, #2, MUL VL]\n"
    "fmla z28.h, p2/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x20, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z4.h, z14.h\n"
    "fmla z30.h, p2/M, z4.h, z8.h\n"
    "fmla z31.h, p2/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p2/Z, [x14, #3, MUL VL]\n"
    "fmla z28.h, p2/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x27, x15, LSL #1]\n"
    "fmla z29.h, p2/M, z0.h, z13.h\n"
    "fmla z30.h, p2/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p3/Z, [x26, x15, LSL #1]\n"
    "fmla z31.h, p2/M, z0.h, z12.h\n"
    "fmla z28.h, p2/M, z1.h, z13.h\n"
    "fmla z29.h, p2/M, z1.h, z5.h\n"
    "fmla z30.h, p2/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p3/Z, [x25, x15, LSL #1]\n"
    "fmla z31.h, p2/M, z1.h, z9.h\n"
    "fmla z28.h, p2/M, z2.h, z5.h\n"
    "fmla z29.h, p2/M, z2.h, z6.h\n"
    "fmla z30.h, p2/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p3/Z, [x24, x15, LSL #1]\n"
    "fmla z31.h, p2/M, z2.h, z11.h\n"
    "fmla z28.h, p2/M, z3.h, z6.h\n"
    "fmla z29.h, p2/M, z3.h, z8.h\n"
    "fmla z30.h, p2/M, z3.h, z11.h\n"
    "fmla z31.h, p2/M, z3.h, z12.h\n"
    "fmla z28.h, p2/M, z4.h, z8.h\n"
    "fmla z29.h, p2/M, z4.h, z10.h\n"
    "fmla z30.h, p2/M, z4.h, z12.h\n"
    "fmla z31.h, p2/M, z4.h, z9.h\n"
    ".inst 0xc170ca5c  // fclamp { z28.h-z31.h }, z18.h, z16.h\n"
    "st1h { z28.h }, p0, [x12, x28, LSL #1]\n"
    "st1h { z29.h }, p0, [x11, x28, LSL #1]\n"
    "st1h { z30.h }, p0, [x10, x28, LSL #1]\n"
    "st1h { z31.h }, p0, [x9, x28, LSL #1]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif // defined(ARM_COMPUTE_ENABLE_SME2)
