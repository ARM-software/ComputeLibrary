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

void sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst_indirect_impl(
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
    const __fp16 *inptrs[16];

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
      inptrs[2] = input_ptrs[2];
      inptrs[3] = input_ptrs[3];
      inptrs[4] = input_ptrs[4];
      inptrs[5] = input_ptrs[5];
      inptrs[6] = input_ptrs[6];
      inptrs[7] = input_ptrs[7];
      inptrs[8] = input_ptrs[8];
      inptrs[9] = input_ptrs[9];
      inptrs[10] = input_ptrs[10];
      inptrs[11] = input_ptrs[11];
      inptrs[12] = input_ptrs[12];
      inptrs[13] = input_ptrs[13];
      inptrs[14] = input_ptrs[14];
      inptrs[15] = input_ptrs[15];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ldr x3, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ptrue p3.b\n"
    "ldr x4, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x19, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ld1rh { z18.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x5, #0x0\n"
    "ldp x6, x7, [x19, #0x0]\n"
    "cnth x8\n"
    "ldp x17, x16, [x19, #0x10]\n"
    "sub x15, XZR, x8\n"
    "ldp x14, x13, [x19, #0x20]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ldp x12, x11, [x19, #0x30]\n"
    "cmp x8, %x[n_channels]\n"
    "ldp x10, x9, [x19, #0x40]\n"
    "ldp x28, x27, [x19, #0x50]\n"
    "ldp x26, x25, [x19, #0x60]\n"
    "ldp x24, x23, [x19, #0x70]\n"
    "ldp x22, x21, [x3, #0x0]\n"
    "ldp x20, x19, [x3, #0x10]\n"
    "ld1rh { z17.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1h { z16.h }, p3/Z, [x4]\n" // Load from weights and bias
    "mov z31.d, z16.d\n"
    "ld1h { z0.h }, p3/Z, [x4, #1, MUL VL]\n" // Load from weights and bias
    "mov z30.d, z16.d\n"
    "ld1h { z1.h }, p3/Z, [x4, #2, MUL VL]\n" // Load from weights and bias
    "mov z29.d, z16.d\n"
    "ld1h { z2.h }, p3/Z, [x4, #3, MUL VL]\n" // Load from weights and bias
    "mov z28.d, z16.d\n"
    "ld1h { z3.h }, p3/Z, [x4, #4, MUL VL]\n" // Load from weights and bias
    "ld1h { z4.h }, p3/Z, [x4, #5, MUL VL]\n" // Load from weights and bias
    "ld1h { z5.h }, p3/Z, [x4, #6, MUL VL]\n" // Load from weights and bias
    "ld1h { z6.h }, p3/Z, [x4, #7, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #16\n"
    "ld1h { z9.h }, p2/Z, [x13, x5, LSL #1]\n"
    "ld1h { z7.h }, p3/Z, [x4, #-8, MUL VL]\n" // Load from weights and bias
    "ld1h { z8.h }, p3/Z, [x4, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #-6\n"
    "ld1h { z10.h }, p2/Z, [x6, x5, LSL #1]\n"
    "ld1h { z11.h }, p2/Z, [x16, x5, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x12, x5, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x9, x5, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "fmla z31.h, p3/M, z4.h, z9.h\n"
    "ld1h { z16.h }, p3/Z, [x4]\n" // Load from weights and bias
    "whilelt p1.h, x8, %x[n_channels]\n"
    "fmla z30.h, p3/M, z3.h, z9.h\n"
    "inch x15\n"
    "fmla z29.h, p3/M, z1.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "fmla z28.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x26, x5, LSL #1]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x28, x5, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x23, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z12.h\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z31.h, p3/M, z5.h, z12.h\n"
    "fmla z30.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x7, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x17, x5, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "fmla z31.h, p3/M, z7.h, z13.h\n"
    "fmla z30.h, p3/M, z6.h, z13.h\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "ld1h { z13.h }, p1/Z, [x9, x8, LSL #1]\n"
    "fmla z28.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x5, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x4, #5, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z2.h, z9.h\n"
    "fmla z30.h, p3/M, z1.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x10, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x4, #1, MUL VL]\n" // Load from weights and bias
    "fmla z28.h, p3/M, z2.h, z12.h\n"
    "ld1h { z1.h }, p3/Z, [x4, #2, MUL VL]\n" // Load from weights and bias
    "ld1h { z2.h }, p3/Z, [x4, #3, MUL VL]\n" // Load from weights and bias
    "fmla z31.h, p3/M, z8.h, z10.h\n"
    "fmla z30.h, p3/M, z7.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z9.h\n"
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x25, x5, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z10.h\n"
    "ld1h { z3.h }, p3/Z, [x4, #4, MUL VL]\n" // Load from weights and bias
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x24, x5, LSL #1]\n"
    "inch x5\n"
    "fmla z29.h, p3/M, z7.h, z11.h\n"
    "ld1h { z5.h }, p3/Z, [x4, #6, MUL VL]\n" // Load from weights and bias
    "whilelt p2.h, x5, %x[n_channels]\n"
    "fmla z31.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p1/Z, [x13, x8, LSL #1]\n"
    "fmla z28.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x16, x8, LSL #1]\n"
    "fmla z30.h, p3/M, z8.h, z10.h\n"
    "ld1h { z10.h }, p1/Z, [x6, x8, LSL #1]\n"
    "ld1h { z6.h }, p3/Z, [x4, #7, MUL VL]\n" // Load from weights and bias
    "fmla z29.h, p3/M, z8.h, z12.h\n"
    "addvl x4, x4, #16\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x12, x8, LSL #1]\n"
    "inch x8\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "ld1h { z7.h }, p3/Z, [x4, #-8, MUL VL]\n" // Load from weights and bias
    "cmp x8, %x[n_channels]\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "ld1h { z8.h }, p3/Z, [x4, #-7, MUL VL]\n" // Load from weights and bias
    "addvl x4, x4, #-6\n"
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "st1h { z31.h }, p0, [x22, x15, LSL #1]\n"
    "mov z31.d, z16.d\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "st1h { z30.h }, p0, [x21, x15, LSL #1]\n"
    "mov z30.d, z16.d\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "st1h { z29.h }, p0, [x20, x15, LSL #1]\n"
    "mov z29.d, z16.d\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "st1h { z28.h }, p0, [x19, x15, LSL #1]\n"
    "mov z28.d, z16.d\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "fmla z31.h, p3/M, z4.h, z9.h\n"
    "inch x15\n"
    "fmla z30.h, p3/M, z3.h, z9.h\n"
    "mov p0.b, p2.b\n"
    "fmla z29.h, p3/M, z1.h, z9.h\n"
    "fmla z28.h, p3/M, z0.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x26, x5, LSL #1]\n"
    "fmla z31.h, p3/M, z0.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x28, x5, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x23, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z2.h, z12.h\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z31.h, p3/M, z5.h, z12.h\n"
    "fmla z30.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x7, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x17, x5, LSL #1]\n"
    "fmla z28.h, p3/M, z3.h, z13.h\n"
    "fmla z31.h, p3/M, z7.h, z13.h\n"
    "fmla z30.h, p3/M, z6.h, z13.h\n"
    "fmla z29.h, p3/M, z4.h, z13.h\n"
    "fmla z28.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x14, x5, LSL #1]\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "fmla z30.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x11, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z4.h, z10.h\n"
    "fmla z31.h, p3/M, z2.h, z9.h\n"
    "fmla z30.h, p3/M, z1.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x10, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z0.h, z11.h\n"
    "fmla z28.h, p3/M, z2.h, z12.h\n"
    "fmla z31.h, p3/M, z8.h, z10.h\n"
    "fmla z30.h, p3/M, z7.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27, x5, LSL #1]\n"
    "fmla z29.h, p3/M, z3.h, z9.h\n"
    "fmla z31.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x25, x5, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x24, x5, LSL #1]\n"
    "fmla z28.h, p3/M, z5.h, z10.h\n"
    "fmla z29.h, p3/M, z7.h, z11.h\n"
    "fmla z31.h, p3/M, z6.h, z9.h\n"
    "fmla z30.h, p3/M, z8.h, z10.h\n"
    "fmla z28.h, p3/M, z6.h, z11.h\n"
    "fmla z29.h, p3/M, z8.h, z12.h\n"
    "fmax z31.h, p3/M, z31.h, z18.h\n"
    "fmax z30.h, p3/M, z30.h, z18.h\n"
    "fmla z28.h, p3/M, z7.h, z12.h\n"
    "fmax z29.h, p3/M, z29.h, z18.h\n"
    "fmin z31.h, p3/M, z31.h, z17.h\n"
    "st1h { z31.h }, p0, [x22, x15, LSL #1]\n"
    "fmin z30.h, p3/M, z30.h, z17.h\n"
    "fmin z29.h, p3/M, z29.h, z17.h\n"
    "st1h { z30.h }, p0, [x21, x15, LSL #1]\n"
    "fmax z28.h, p3/M, z28.h, z18.h\n"
    "st1h { z29.h }, p0, [x20, x15, LSL #1]\n"
    "fmin z28.h, p3/M, z28.h, z17.h\n"
    "st1h { z28.h }, p0, [x19, x15, LSL #1]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS)
