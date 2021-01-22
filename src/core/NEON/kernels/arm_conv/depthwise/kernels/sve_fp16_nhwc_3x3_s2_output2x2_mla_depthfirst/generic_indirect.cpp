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

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace depthwise {

void sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(
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
    "ldr x19, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ptrue p3.b\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x15, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ld1rh { z19.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x14, #0x0\n"
    "ld1rh { z18.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "cnth x13\n"
    "ldp x12, x11, [x19, #0x0]\n"
    "sub x10, XZR, x13\n"
    "ldp x9, x28, [x19, #0x10]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ld1h { z17.h }, p3/Z, [x16]\n" // Load from weights and bias
    "mov z31.d, z17.d\n"
    "ld1h { z0.h }, p3/Z, [x16, #1, MUL VL]\n" // Load from weights and bias
    "cmp x13, %x[n_channels]\n"
    "mov z30.d, z17.d\n"
    "ld1h { z1.h }, p3/Z, [x16, #2, MUL VL]\n" // Load from weights and bias
    "mov z29.d, z17.d\n"
    "ld1h { z2.h }, p3/Z, [x16, #3, MUL VL]\n" // Load from weights and bias
    "mov z28.d, z17.d\n"
    "ld1h { z3.h }, p3/Z, [x16, #4, MUL VL]\n" // Load from weights and bias
    "ld1h { z4.h }, p3/Z, [x16, #5, MUL VL]\n" // Load from weights and bias
    "ld1h { z5.h }, p3/Z, [x16, #6, MUL VL]\n" // Load from weights and bias
    "ld1h { z6.h }, p3/Z, [x16, #7, MUL VL]\n" // Load from weights and bias
    "addvl x16, x16, #16\n"
    "ldp x27, x26, [x15, #0x0]\n"
    "ld1h { z7.h }, p3/Z, [x16, #-8, MUL VL]\n" // Load from weights and bias
    "ld1h { z8.h }, p3/Z, [x16, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x16, x16, #-6\n"
    "ld1h { z9.h }, p2/Z, [x27, x14, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x26, x14, LSL #1]\n"
    "ldp x25, x23, [x15, #0x10]\n"
    "ldp x22, x21, [x15, #0x20]\n"
    "ldp x20, x19, [x15, #0x30]\n"
    "ld1h { z11.h }, p2/Z, [x25, x14, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x23, x14, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x22, x14, LSL #1]\n"
    "ld1h { z14.h }, p2/Z, [x21, x14, LSL #1]\n"
    "ld1h { z15.h }, p2/Z, [x20, x14, LSL #1]\n"
    "ld1h { z16.h }, p2/Z, [x19, x14, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "fmla z31.h, p3/M, z8.h, z9.h\n"
    "ldr x24, [x15, #0x40]\n"
    "whilelt p1.h, x13, %x[n_channels]\n"
    "fmla z30.h, p3/M, z6.h, z9.h\n"
    "ldr x20, [x15, #0x48]\n"
    "inch x10\n"
    "fmla z29.h, p3/M, z2.h, z9.h\n"
    "ldr x23, [x15, #0x50]\n"
    "mov p0.b, p2.b\n"
    "fmla z28.h, p3/M, z0.h, z9.h\n"
    "ldr x19, [x15, #0x58]\n"
    "ldr x22, [x15, #0x60]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ldr x21, [x15, #0x68]\n"
    "fmla z30.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x24, x14, LSL #1]\n"
    "ldr x20, [x15, #0x70]\n"
    "fmla z30.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x23, x14, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x19, x14, LSL #1]\n"
    "ldr x19, [x15, #0x78]\n"
    "fmla z30.h, p3/M, z0.h, z16.h\n"
    "ldr x27, [x15, #0x80]\n"
    "fmla z29.h, p3/M, z3.h, z14.h\n"
    "ldr x26, [x15, #0x88]\n"
    "ldr x25, [x15, #0x90]\n"
    "fmla z31.h, p3/M, z4.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x22, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z0.h, z15.h\n"
    "ld1h { z14.h }, p2/Z, [x26, x14, LSL #1]\n"
    "ldr x23, [x15, #0x98]\n"
    "fmla z31.h, p3/M, z2.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x23, x14, LSL #1]\n"
    "ldr x22, [x15, #0xa0]\n"
    "fmla z31.h, p3/M, z5.h, z13.h\n"
    "ldr x21, [x15, #0xa8]\n"
    "fmla z30.h, p3/M, z3.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x19, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z1.h, z16.h\n"
    "ldr x20, [x15, #0xb0]\n"
    "ldr x19, [x15, #0xb8]\n"
    "fmla z31.h, p3/M, z6.h, z15.h\n"
    "fmla z28.h, p3/M, z4.h, z13.h\n"
    "ld1h { z15.h }, p2/Z, [x25, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z7.h, z12.h\n"
    "ld1h { z13.h }, p2/Z, [x22, x14, LSL #1]\n"
    "ldr x24, [x15, #0xc0]\n"
    "fmla z31.h, p3/M, z7.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x21, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "ldp x27, x26, [x15, #0x0]\n"
    "fmla z29.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x19, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z11.h\n"
    "ldp x25, x23, [x15, #0x10]\n"
    "ldp x22, x21, [x15, #0x20]\n"
    "fmla z28.h, p3/M, z5.h, z14.h\n"
    "fmax z31.h, p3/M, z31.h, z19.h\n"
    "ld1h { z14.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z7.h, z13.h\n"
    "ld1h { z9.h }, p1/Z, [x27, x13, LSL #1]\n"
    "fmax z30.h, p3/M, z30.h, z19.h\n"
    "ld1h { z10.h }, p1/Z, [x26, x13, LSL #1]\n"
    "ld1h { z12.h }, p1/Z, [x23, x13, LSL #1]\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "fmin z31.h, p3/M, z31.h, z18.h\n"
    "ld1h { z11.h }, p2/Z, [x24, x14, LSL #1]\n"
    "inch x14\n"
    "fmla z29.h, p3/M, z5.h, z16.h\n"
    "ld1h { z13.h }, p1/Z, [x22, x13, LSL #1]\n"
    "whilelt p2.h, x14, %x[n_channels]\n"
    "fmin z30.h, p3/M, z30.h, z18.h\n"
    "ldp x20, x19, [x15, #0x30]\n"
    "ld1h { z17.h }, p3/Z, [x16]\n" // Load from weights and bias
    "fmla z28.h, p3/M, z3.h, z16.h\n"
    "st1h { z31.h }, p0, [x12, x10, LSL #1]\n"
    "mov z31.d, z17.d\n"
    "ld1h { z16.h }, p1/Z, [x19, x13, LSL #1]\n"
    "fmla z29.h, p3/M, z8.h, z15.h\n"
    "st1h { z30.h }, p0, [x11, x10, LSL #1]\n"
    "mov z30.d, z17.d\n"
    "ld1h { z0.h }, p3/Z, [x16, #1, MUL VL]\n" // Load from weights and bias
    "fmla z28.h, p3/M, z7.h, z14.h\n"
    "ld1h { z14.h }, p1/Z, [x21, x13, LSL #1]\n"
    "ld1h { z1.h }, p3/Z, [x16, #2, MUL VL]\n" // Load from weights and bias
    "fmax z29.h, p3/M, z29.h, z19.h\n"
    "ld1h { z2.h }, p3/Z, [x16, #3, MUL VL]\n" // Load from weights and bias
    "fmla z28.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p1/Z, [x20, x13, LSL #1]\n"
    "ld1h { z3.h }, p3/Z, [x16, #4, MUL VL]\n" // Load from weights and bias
    "fmin z29.h, p3/M, z29.h, z18.h\n"
    "ld1h { z4.h }, p3/Z, [x16, #5, MUL VL]\n" // Load from weights and bias
    "ld1h { z5.h }, p3/Z, [x16, #6, MUL VL]\n" // Load from weights and bias
    "fmla z28.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x25, x13, LSL #1]\n"
    "inch x13\n"
    "fmax z28.h, p3/M, z28.h, z19.h\n"
    "st1h { z29.h }, p0, [x9, x10, LSL #1]\n"
    "cmp x13, %x[n_channels]\n"
    "mov z29.d, z17.d\n"
    "ld1h { z6.h }, p3/Z, [x16, #7, MUL VL]\n" // Load from weights and bias
    "addvl x16, x16, #16\n"
    "fmin z28.h, p3/M, z28.h, z18.h\n"
    "ld1h { z7.h }, p3/Z, [x16, #-8, MUL VL]\n" // Load from weights and bias
    "ld1h { z8.h }, p3/Z, [x16, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x16, x16, #-6\n"
    "st1h { z28.h }, p0, [x28, x10, LSL #1]\n"
    "mov z28.d, z17.d\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "fmla z31.h, p3/M, z8.h, z9.h\n"
    "ldr x24, [x15, #0x40]\n"
    "inch x10\n"
    "fmla z30.h, p3/M, z6.h, z9.h\n"
    "ldr x20, [x15, #0x48]\n"
    "mov p0.b, p2.b\n"
    "fmla z29.h, p3/M, z2.h, z9.h\n"
    "ldr x23, [x15, #0x50]\n"
    "fmla z28.h, p3/M, z0.h, z9.h\n"
    "ldr x19, [x15, #0x58]\n"
    "ldr x22, [x15, #0x60]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ldr x21, [x15, #0x68]\n"
    "fmla z30.h, p3/M, z1.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x24, x14, LSL #1]\n"
    "ldr x20, [x15, #0x70]\n"
    "fmla z30.h, p3/M, z2.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x23, x14, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x19, x14, LSL #1]\n"
    "ldr x19, [x15, #0x78]\n"
    "fmla z30.h, p3/M, z0.h, z16.h\n"
    "ldr x27, [x15, #0x80]\n"
    "fmla z29.h, p3/M, z3.h, z14.h\n"
    "ldr x26, [x15, #0x88]\n"
    "ldr x25, [x15, #0x90]\n"
    "fmla z31.h, p3/M, z4.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x22, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z0.h, z15.h\n"
    "ld1h { z14.h }, p2/Z, [x26, x14, LSL #1]\n"
    "ldr x23, [x15, #0x98]\n"
    "fmla z31.h, p3/M, z2.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z4.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x23, x14, LSL #1]\n"
    "ldr x22, [x15, #0xa0]\n"
    "fmla z31.h, p3/M, z5.h, z13.h\n"
    "ldr x21, [x15, #0xa8]\n"
    "fmla z30.h, p3/M, z3.h, z13.h\n"
    "ld1h { z13.h }, p2/Z, [x19, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z1.h, z16.h\n"
    "ldr x20, [x15, #0xb0]\n"
    "ldr x19, [x15, #0xb8]\n"
    "fmla z31.h, p3/M, z6.h, z15.h\n"
    "fmla z28.h, p3/M, z4.h, z13.h\n"
    "ld1h { z15.h }, p2/Z, [x25, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z7.h, z12.h\n"
    "ld1h { z13.h }, p2/Z, [x22, x14, LSL #1]\n"
    "ldr x24, [x15, #0xc0]\n"
    "fmla z31.h, p3/M, z7.h, z16.h\n"
    "ld1h { z16.h }, p2/Z, [x21, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z29.h, p3/M, z6.h, z15.h\n"
    "ld1h { z15.h }, p2/Z, [x19, x14, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z11.h\n"
    "fmla z28.h, p3/M, z5.h, z14.h\n"
    "ld1h { z14.h }, p2/Z, [x20, x14, LSL #1]\n"
    "fmax z31.h, p3/M, z31.h, z19.h\n"
    "fmla z29.h, p3/M, z7.h, z13.h\n"
    "fmax z30.h, p3/M, z30.h, z19.h\n"
    "fmla z28.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x24, x14, LSL #1]\n"
    "fmin z31.h, p3/M, z31.h, z18.h\n"
    "st1h { z31.h }, p0, [x12, x10, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z16.h\n"
    "fmla z28.h, p3/M, z3.h, z16.h\n"
    "fmin z30.h, p3/M, z30.h, z18.h\n"
    "st1h { z30.h }, p0, [x11, x10, LSL #1]\n"
    "fmla z28.h, p3/M, z7.h, z14.h\n"
    "fmla z29.h, p3/M, z8.h, z15.h\n"
    "fmla z28.h, p3/M, z6.h, z15.h\n"
    "fmax z29.h, p3/M, z29.h, z19.h\n"
    "fmla z28.h, p3/M, z8.h, z11.h\n"
    "fmin z29.h, p3/M, z29.h, z18.h\n"
    "st1h { z29.h }, p0, [x9, x10, LSL #1]\n"
    "fmax z28.h, p3/M, z28.h, z19.h\n"
    "fmin z28.h, p3/M, z28.h, z18.h\n"
    "st1h { z28.h }, p0, [x28, x10, LSL #1]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS)
