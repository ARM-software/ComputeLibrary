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

#if defined(__ARM_FEATURE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_indirect_impl(
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
    "ldr x2, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ptrue p3.b\n"
    "ldr x3, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x19, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "cntb x4, ALL, MUL #2\n"
    "ldp x5, x6, [x19, #0x0]\n"
    "mov x7, #0x0\n"
    "ldp x8, x17, [x19, #0x10]\n"
    "cntw x16\n"
    "ldp x15, x14, [x19, #0x20]\n"
    "sub x13, XZR, x16\n"
    "ldp x12, x11, [x19, #0x30]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ldp x10, x9, [x19, #0x40]\n"
    "cmp x16, %x[n_channels]\n"
    "ldp x28, x27, [x19, #0x50]\n"
    "ldp x26, x25, [x19, #0x60]\n"
    "ldp x24, x23, [x19, #0x70]\n"
    "ldp x22, x21, [x2, #0x0]\n"
    "ldp x20, x19, [x2, #0x10]\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1w { z16.s }, p3/Z, [x3]\n"
    "mov z31.d, z16.d\n"
    "ld1w { z0.s }, p3/Z, [x3, #1, MUL VL]\n"
    "mov z30.d, z16.d\n"
    "ld1w { z1.s }, p3/Z, [x3, #2, MUL VL]\n"
    "mov z29.d, z16.d\n"
    "ld1w { z2.s }, p3/Z, [x3, #3, MUL VL]\n"
    "mov z28.d, z16.d\n"
    "ld1w { z3.s }, p3/Z, [x3, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x3, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x3, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x3, #7, MUL VL]\n"
    "addvl x3, x3, #16\n"
    "ld1w { z9.s }, p2/Z, [x14, x7, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x3, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x3, #-7, MUL VL]\n"
    "addvl x3, x3, #-6\n"
    "prfm pldl1keep, [x14, x4]\n"
    "ld1w { z10.s }, p2/Z, [x5, x7, LSL #2]\n"
    "prfm pldl1keep, [x5, x4]\n"
    "ld1w { z11.s }, p2/Z, [x17, x7, LSL #2]\n"
    "prfm pldl1keep, [x17, x4]\n"
    "ld1w { z12.s }, p2/Z, [x12, x7, LSL #2]\n"
    "prfm pldl1keep, [x12, x4]\n"
    "ld1w { z13.s }, p2/Z, [x9, x7, LSL #2]\n"
    "prfm pldl1keep, [x9, x4]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "fmla z31.s, p3/M, z4.s, z9.s\n"
    "prfm pldl1keep, [x26, x4]\n"
    "whilelt p1.s, x16, %x[n_channels]\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "prfm pldl1keep, [x23, x4]\n"
    "incw x13\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "prfm pldl1keep, [x6, x4]\n"
    "mov p0.b, p2.b\n"
    "fmla z28.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x26, x7, LSL #2]\n"
    "prfm pldl1keep, [x8, x4]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x7, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x23, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z12.s\n"
    "prfm pldl1keep, [x28, x4]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "prfm pldl1keep, [x15, x4]\n"
    "fmla z31.s, p3/M, z5.s, z12.s\n"
    "prfm pldl1keep, [x11, x4]\n"
    "fmla z30.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x6, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x8, x7, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "prfm pldl1keep, [x10, x4]\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "prfm pldl1keep, [x27, x4]\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "prfm pldl1keep, [x25, x4]\n"
    "fmla z29.s, p3/M, z4.s, z13.s\n"
    "prfm pldl1keep, [x24, x4]\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "addvl x4, x4, #1\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z11.s }, p2/Z, [x15, x7, LSL #2]\n"
    "prfm pldl1keep, [x14, x4]\n"
    "fmla z30.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "prfm pldl1keep, [x5, x4]\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "prfm pldl1keep, [x17, x4]\n"
    "fmla z31.s, p3/M, z2.s, z9.s\n"
    "prfm pldl1keep, [x12, x4]\n"
    "fmla z30.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x10, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "ld1w { z13.s }, p1/Z, [x9, x16, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z12.s\n"
    "prfm pldl1keep, [x9, x4]\n"
    "fmla z31.s, p3/M, z8.s, z10.s\n"
    "ld1w { z16.s }, p3/Z, [x3]\n"
    "fmla z30.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x27, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z3.s, z9.s\n"
    "ld1w { z0.s }, p3/Z, [x3, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x7, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z10.s\n"
    "ld1w { z1.s }, p3/Z, [x3, #2, MUL VL]\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x24, x7, LSL #2]\n"
    "incw x7\n"
    "fmla z29.s, p3/M, z7.s, z11.s\n"
    "ld1w { z2.s }, p3/Z, [x3, #3, MUL VL]\n"
    "whilelt p2.s, x7, %x[n_channels]\n"
    "fmla z31.s, p3/M, z6.s, z9.s\n"
    "ld1w { z9.s }, p1/Z, [x14, x16, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x17, x16, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z10.s\n"
    "ld1w { z10.s }, p1/Z, [x5, x16, LSL #2]\n"
    "ld1w { z3.s }, p3/Z, [x3, #4, MUL VL]\n"
    "fmla z29.s, p3/M, z8.s, z12.s\n"
    "ld1w { z4.s }, p3/Z, [x3, #5, MUL VL]\n"
    "fmla z28.s, p3/M, z7.s, z12.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "ld1w { z12.s }, p1/Z, [x12, x16, LSL #2]\n"
    "incw x16\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "ld1w { z5.s }, p3/Z, [x3, #6, MUL VL]\n"
    "cmp x16, %x[n_channels]\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "ld1w { z6.s }, p3/Z, [x3, #7, MUL VL]\n"
    "addvl x3, x3, #16\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "ld1w { z7.s }, p3/Z, [x3, #-8, MUL VL]\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "ld1w { z8.s }, p3/Z, [x3, #-7, MUL VL]\n"
    "addvl x3, x3, #-6\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "st1w { z31.s }, p0, [x22, x13, LSL #2]\n"
    "mov z31.d, z16.d\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z30.s }, p0, [x21, x13, LSL #2]\n"
    "mov z30.d, z16.d\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z29.s }, p0, [x20, x13, LSL #2]\n"
    "mov z29.d, z16.d\n"
    "st1w { z28.s }, p0, [x19, x13, LSL #2]\n"
    "mov z28.d, z16.d\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "fmla z31.s, p3/M, z4.s, z9.s\n"
    "prfm pldl1keep, [x26, x4]\n"
    "incw x13\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "prfm pldl1keep, [x23, x4]\n"
    "mov p0.b, p2.b\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "prfm pldl1keep, [x6, x4]\n"
    "fmla z28.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x26, x7, LSL #2]\n"
    "prfm pldl1keep, [x8, x4]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x7, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x23, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "prfm pldl1keep, [x28, x4]\n"
    "prfm pldl1keep, [x15, x4]\n"
    "fmla z31.s, p3/M, z5.s, z12.s\n"
    "prfm pldl1keep, [x11, x4]\n"
    "fmla z30.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x6, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z6.s, z9.s\n"
    "fmla z28.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x8, x7, LSL #2]\n"
    "prfm pldl1keep, [x10, x4]\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "prfm pldl1keep, [x27, x4]\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "prfm pldl1keep, [x25, x4]\n"
    "fmla z29.s, p3/M, z4.s, z13.s\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x15, x7, LSL #2]\n"
    "prfm pldl1keep, [x24, x4]\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "fmla z30.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "fmla z31.s, p3/M, z2.s, z9.s\n"
    "fmla z30.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x10, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z11.s\n"
    "fmla z28.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z8.s, z10.s\n"
    "fmla z30.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x27, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x7, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z10.s\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x24, x7, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z9.s\n"
    "fmla z28.s, p3/M, z6.s, z11.s\n"
    "fmla z30.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z8.s, z12.s\n"
    "fmla z28.s, p3/M, z7.s, z12.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z31.s }, p0, [x22, x13, LSL #2]\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "st1w { z30.s }, p0, [x21, x13, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "st1w { z29.s }, p0, [x20, x13, LSL #2]\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "st1w { z28.s }, p0, [x19, x13, LSL #2]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE)
