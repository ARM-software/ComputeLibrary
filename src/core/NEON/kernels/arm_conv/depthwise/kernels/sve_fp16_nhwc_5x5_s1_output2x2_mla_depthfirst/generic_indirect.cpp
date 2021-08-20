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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl(
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
    "ldr x19, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ptrue p3.b\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x6, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ld1rh { z18.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x7, #0x0\n"
    "ld1rh { z17.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "cnth x8\n"
    "ldp x17, x16, [x19, #0x0]\n"
    "sub x15, XZR, x8\n"
    "ldp x14, x13, [x19, #0x10]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1h { z16.h }, p3/Z, [x5]\n" // Load from weights and bias
    "mov z31.d, z16.d\n"
    "ld1h { z0.h }, p3/Z, [x5, #1, MUL VL]\n" // Load from weights and bias
    "cmp x8, %x[n_channels]\n"
    "mov z30.d, z16.d\n"
    "ld1h { z1.h }, p3/Z, [x5, #2, MUL VL]\n" // Load from weights and bias
    "mov z29.d, z16.d\n"
    "ld1h { z2.h }, p3/Z, [x5, #3, MUL VL]\n" // Load from weights and bias
    "mov z28.d, z16.d\n"
    "ld1h { z3.h }, p3/Z, [x5, #4, MUL VL]\n" // Load from weights and bias
    "ld1h { z4.h }, p3/Z, [x5, #5, MUL VL]\n" // Load from weights and bias
    "addvl x5, x5, #6\n"
    "ldp x12, x11, [x6, #0x0]\n"
    "ldp x10, x9, [x6, #0x10]\n"
    "ldp x20, x28, [x6, #0x20]\n"
    "ld1h { z5.h }, p2/Z, [x12, x7, LSL #1]\n"
    "ld1h { z6.h }, p2/Z, [x11, x7, LSL #1]\n"
    "ld1h { z7.h }, p2/Z, [x10, x7, LSL #1]\n"
    "ld1h { z8.h }, p2/Z, [x9, x7, LSL #1]\n"
    "ld1h { z9.h }, p2/Z, [x20, x7, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x28, x7, LSL #1]\n"
    "ldp x27, x19, [x6, #0x30]\n"
    "ldp x26, x25, [x6, #0x40]\n"
    "ld1h { z11.h }, p2/Z, [x27, x7, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x19, x7, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x26, x7, LSL #1]\n"
    "ld1h { z14.h }, p2/Z, [x25, x7, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "fmla z31.h, p3/M, z0.h, z5.h\n"
    "ldr x24, [x6, #0x50]\n"
    "whilelt p1.h, x8, %x[n_channels]\n"
    "fmla z30.h, p3/M, z0.h, z6.h\n"
    "ldr x23, [x6, #0x58]\n"
    "inch x15\n"
    "fmla z29.h, p3/M, z0.h, z7.h\n"
    "ldr x22, [x6, #0x60]\n"
    "mov p0.b, p2.b\n"
    "fmla z28.h, p3/M, z0.h, z8.h\n"
    "ld1h { z5.h }, p2/Z, [x24, x7, LSL #1]\n"
    "ld1h { z0.h }, p3/Z, [x5]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x23, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z9.h\n"
    "ldr x21, [x6, #0x68]\n"
    "fmla z29.h, p3/M, z1.h, z8.h\n"
    "ldr x20, [x6, #0x70]\n"
    "fmla z28.h, p3/M, z1.h, z13.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #1, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x22, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "ldr x19, [x6, #0x78]\n"
    "fmla z29.h, p3/M, z2.h, z13.h\n"
    "ldr x12, [x6, #0x80]\n"
    "fmla z28.h, p3/M, z2.h, z5.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #2, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "ldr x11, [x6, #0x88]\n"
    "fmla z29.h, p3/M, z3.h, z5.h\n"
    "ldr x10, [x6, #0x90]\n"
    "fmla z28.h, p3/M, z3.h, z6.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z6.h\n"
    "ldr x9, [x6, #0x98]\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #4, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z7.h\n"
    "ldr x20, [x6, #0xa0]\n"
    "fmla z30.h, p3/M, z0.h, z8.h\n"
    "ldr x28, [x6, #0xa8]\n"
    "fmla z29.h, p3/M, z0.h, z14.h\n"
    "ldr x27, [x6, #0xb0]\n"
    "fmla z28.h, p3/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #5, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z8.h\n"
    "ld1h { z8.h }, p2/Z, [x11, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z13.h\n"
    "ldr x19, [x6, #0xb8]\n"
    "fmla z29.h, p3/M, z1.h, z11.h\n"
    "ldr x26, [x6, #0xc0]\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #6, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x12, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z5.h\n"
    "ldr x25, [x6, #0xc8]\n"
    "fmla z29.h, p3/M, z2.h, z12.h\n"
    "ldr x24, [x6, #0xd0]\n"
    "fmla z28.h, p3/M, z2.h, z9.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #7, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z3.h, z5.h\n"
    "addvl x5, x5, #16\n"
    "fmla z30.h, p3/M, z3.h, z6.h\n"
    "ld1h { z5.h }, p2/Z, [x10, x7, LSL #1]\n"
    "ldr x23, [x6, #0xd8]\n"
    "fmla z29.h, p3/M, z3.h, z9.h\n"
    "ldr x22, [x6, #0xe0]\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #-8, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x9, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "ldr x21, [x6, #0xe8]\n"
    "fmla z28.h, p3/M, z4.h, z8.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #-7, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z11.h\n"
    "ldr x20, [x6, #0xf0]\n"
    "fmla z29.h, p3/M, z0.h, z5.h\n"
    "ldr x19, [x6, #0xf8]\n"
    "fmla z28.h, p3/M, z0.h, z6.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #-6, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x28, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z12.h\n"
    "ldr x12, [x6, #0x100]\n"
    "fmla z29.h, p3/M, z1.h, z6.h\n"
    "ldr x11, [x6, #0x108]\n"
    "fmla z28.h, p3/M, z1.h, z10.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #-5, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z9.h\n"
    "ldr x10, [x6, #0x110]\n"
    "fmla z29.h, p3/M, z2.h, z10.h\n"
    "ldr x9, [x6, #0x118]\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #-4, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z3.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x26, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z13.h\n"
    "ld1h { z16.h }, p3/Z, [x5, #4, MUL VL]\n" // Load from weights and bias
    "fmla z29.h, p3/M, z3.h, z11.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #-3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x25, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p2/Z, [x22, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "fmla z28.h, p3/M, z4.h, z14.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #-2, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z5.h\n"
    "ld1h { z5.h }, p2/Z, [x24, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z6.h\n"
    "fmla z29.h, p3/M, z0.h, z9.h\n"
    "fmla z28.h, p3/M, z0.h, z13.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #-1, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x23, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z10.h\n"
    "fmla z29.h, p3/M, z1.h, z13.h\n"
    "fmla z28.h, p3/M, z1.h, z5.h\n"
    "ld1h { z1.h }, p3/Z, [x5]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x21, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "fmla z29.h, p3/M, z2.h, z5.h\n"
    "fmla z28.h, p3/M, z2.h, z6.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #1, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "fmla z29.h, p3/M, z3.h, z6.h\n"
    "fmla z28.h, p3/M, z3.h, z8.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #2, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z14.h\n"
    "fmla z29.h, p3/M, z4.h, z8.h\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x12, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z13.h\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x7, LSL #1]\n"
    "ldp x12, x11, [x6, #0x0]\n"
    "fmla z28.h, p3/M, z0.h, z12.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #5, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z13.h\n"
    "fmla z30.h, p3/M, z1.h, z5.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10, x7, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z9.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #6, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z5.h\n"
    "ld1h { z5.h }, p1/Z, [x12, x8, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z6.h\n"
    "fmla z29.h, p3/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x9, x7, LSL #1]\n"
    "inch x7\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ldp x10, x9, [x6, #0x10]\n"
    "whilelt p2.h, x7, %x[n_channels]\n"
    "fmla z31.h, p3/M, z3.h, z6.h\n"
    "ld1h { z6.h }, p1/Z, [x11, x8, LSL #1]\n"
    "ldp x20, x28, [x6, #0x20]\n"
    "fmla z30.h, p3/M, z3.h, z8.h\n"
    "ldp x27, x19, [x6, #0x30]\n"
    "fmla z29.h, p3/M, z3.h, z11.h\n"
    "ld1h { z7.h }, p1/Z, [x10, x8, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "ld1h { z13.h }, p1/Z, [x28, x8, LSL #1]\n"
    "fmla z31.h, p3/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p1/Z, [x9, x8, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "ld1h { z11.h }, p1/Z, [x27, x8, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x19, x8, LSL #1]\n"
    "fmla z28.h, p3/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p1/Z, [x20, x8, LSL #1]\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "ldp x26, x25, [x6, #0x40]\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #7, MUL VL]\n" // Load from weights and bias
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "addvl x5, x5, #16\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "ld1h { z10.h }, p1/Z, [x26, x8, LSL #1]\n"
    "ld1h { z14.h }, p1/Z, [x25, x8, LSL #1]\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "inch x8\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #-8, MUL VL]\n" // Load from weights and bias
    "cmp x8, %x[n_channels]\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x5, x5, #-6\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "st1h { z31.h }, p0, [x17, x15, LSL #1]\n"
    "mov z31.d, z16.d\n"
    "st1h { z30.h }, p0, [x16, x15, LSL #1]\n"
    "mov z30.d, z16.d\n"
    "st1h { z29.h }, p0, [x14, x15, LSL #1]\n"
    "mov z29.d, z16.d\n"
    "st1h { z28.h }, p0, [x13, x15, LSL #1]\n"
    "mov z28.d, z16.d\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "fmla z31.h, p3/M, z0.h, z5.h\n"
    "ldr x24, [x6, #0x50]\n"
    "inch x15\n"
    "fmla z30.h, p3/M, z0.h, z6.h\n"
    "ldr x23, [x6, #0x58]\n"
    "mov p0.b, p2.b\n"
    "fmla z29.h, p3/M, z0.h, z7.h\n"
    "ldr x22, [x6, #0x60]\n"
    "fmla z28.h, p3/M, z0.h, z8.h\n"
    "ld1h { z5.h }, p2/Z, [x24, x7, LSL #1]\n"
    "ld1h { z0.h }, p3/Z, [x5]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x23, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z9.h\n"
    "ldr x21, [x6, #0x68]\n"
    "fmla z29.h, p3/M, z1.h, z8.h\n"
    "fmla z28.h, p3/M, z1.h, z13.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #1, MUL VL]\n" // Load from weights and bias
    "ldr x20, [x6, #0x70]\n"
    "fmla z31.h, p3/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x22, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "ldr x19, [x6, #0x78]\n"
    "fmla z29.h, p3/M, z2.h, z13.h\n"
    "fmla z28.h, p3/M, z2.h, z5.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #2, MUL VL]\n" // Load from weights and bias
    "ldr x12, [x6, #0x80]\n"
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "ldr x11, [x6, #0x88]\n"
    "fmla z29.h, p3/M, z3.h, z5.h\n"
    "fmla z28.h, p3/M, z3.h, z6.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #3, MUL VL]\n" // Load from weights and bias
    "ldr x10, [x6, #0x90]\n"
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z6.h\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #4, MUL VL]\n" // Load from weights and bias
    "ldr x9, [x6, #0x98]\n"
    "fmla z31.h, p3/M, z0.h, z7.h\n"
    "ldr x20, [x6, #0xa0]\n"
    "fmla z30.h, p3/M, z0.h, z8.h\n"
    "ldr x28, [x6, #0xa8]\n"
    "fmla z29.h, p3/M, z0.h, z14.h\n"
    "fmla z28.h, p3/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #5, MUL VL]\n" // Load from weights and bias
    "ldr x27, [x6, #0xb0]\n"
    "fmla z31.h, p3/M, z1.h, z8.h\n"
    "ld1h { z8.h }, p2/Z, [x11, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z13.h\n"
    "ldr x19, [x6, #0xb8]\n"
    "fmla z29.h, p3/M, z1.h, z11.h\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #6, MUL VL]\n" // Load from weights and bias
    "ldr x26, [x6, #0xc0]\n"
    "fmla z31.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x12, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z5.h\n"
    "ldr x25, [x6, #0xc8]\n"
    "fmla z29.h, p3/M, z2.h, z12.h\n"
    "fmla z28.h, p3/M, z2.h, z9.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #7, MUL VL]\n" // Load from weights and bias
    "addvl x5, x5, #16\n"
    "fmla z31.h, p3/M, z3.h, z5.h\n"
    "ld1h { z5.h }, p2/Z, [x10, x7, LSL #1]\n"
    "ldr x24, [x6, #0xd0]\n"
    "fmla z30.h, p3/M, z3.h, z6.h\n"
    "ldr x23, [x6, #0xd8]\n"
    "fmla z29.h, p3/M, z3.h, z9.h\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #-8, MUL VL]\n" // Load from weights and bias
    "ldr x22, [x6, #0xe0]\n"
    "fmla z31.h, p3/M, z4.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x9, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "fmla z28.h, p3/M, z4.h, z8.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #-7, MUL VL]\n" // Load from weights and bias
    "ldr x21, [x6, #0xe8]\n"
    "fmla z31.h, p3/M, z0.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z11.h\n"
    "ldr x20, [x6, #0xf0]\n"
    "fmla z29.h, p3/M, z0.h, z5.h\n"
    "fmla z28.h, p3/M, z0.h, z6.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #-6, MUL VL]\n" // Load from weights and bias
    "ldr x19, [x6, #0xf8]\n"
    "fmla z31.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x28, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z12.h\n"
    "ldr x12, [x6, #0x100]\n"
    "fmla z29.h, p3/M, z1.h, z6.h\n"
    "fmla z28.h, p3/M, z1.h, z10.h\n"
    "ld1h { z1.h }, p3/Z, [x5, #-5, MUL VL]\n" // Load from weights and bias
    "ldr x11, [x6, #0x108]\n"
    "fmla z31.h, p3/M, z2.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z9.h\n"
    "ldr x10, [x6, #0x110]\n"
    "fmla z29.h, p3/M, z2.h, z10.h\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #-4, MUL VL]\n" // Load from weights and bias
    "ldr x9, [x6, #0x118]\n"
    "fmla z31.h, p3/M, z3.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x26, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z13.h\n"
    "fmla z29.h, p3/M, z3.h, z11.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #-3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x25, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z8.h\n"
    "ld1h { z8.h }, p2/Z, [x22, x7, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "fmla z28.h, p3/M, z4.h, z14.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #-2, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z5.h\n"
    "ld1h { z5.h }, p2/Z, [x24, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z6.h\n"
    "fmla z29.h, p3/M, z0.h, z9.h\n"
    "fmla z28.h, p3/M, z0.h, z13.h\n"
    "ld1h { z0.h }, p3/Z, [x5, #-1, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z1.h, z6.h\n"
    "ld1h { z6.h }, p2/Z, [x23, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z1.h, z10.h\n"
    "fmla z29.h, p3/M, z1.h, z13.h\n"
    "fmla z28.h, p3/M, z1.h, z5.h\n"
    "ld1h { z1.h }, p3/Z, [x5]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x21, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "fmla z29.h, p3/M, z2.h, z5.h\n"
    "fmla z28.h, p3/M, z2.h, z6.h\n"
    "ld1h { z2.h }, p3/Z, [x5, #1, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x20, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z12.h\n"
    "fmla z29.h, p3/M, z3.h, z6.h\n"
    "fmla z28.h, p3/M, z3.h, z8.h\n"
    "ld1h { z3.h }, p3/Z, [x5, #2, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x19, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z14.h\n"
    "fmla z29.h, p3/M, z4.h, z8.h\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x5, #3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x12, x7, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z13.h\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x11, x7, LSL #1]\n"
    "fmla z28.h, p3/M, z0.h, z12.h\n"
    "fmla z31.h, p3/M, z1.h, z13.h\n"
    "fmla z30.h, p3/M, z1.h, z5.h\n"
    "fmla z29.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x10, x7, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z9.h\n"
    "fmla z31.h, p3/M, z2.h, z5.h\n"
    "fmla z30.h, p3/M, z2.h, z6.h\n"
    "fmla z29.h, p3/M, z2.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x9, x7, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "fmla z31.h, p3/M, z3.h, z6.h\n"
    "fmla z30.h, p3/M, z3.h, z8.h\n"
    "fmla z29.h, p3/M, z3.h, z11.h\n"
    "fmla z28.h, p3/M, z3.h, z12.h\n"
    "fmla z31.h, p3/M, z4.h, z8.h\n"
    "fmla z30.h, p3/M, z4.h, z10.h\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "fmla z28.h, p3/M, z4.h, z9.h\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "st1h { z31.h }, p0, [x17, x15, LSL #1]\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "st1h { z30.h }, p0, [x16, x15, LSL #1]\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "st1h { z29.h }, p0, [x14, x15, LSL #1]\n"
    "st1h { z28.h }, p0, [x13, x15, LSL #1]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z16", "z17", "z18", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)
