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

#if __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(
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
    "ldr x19, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ptrue p3.b\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x14, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ld1rw { z19.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x13, #0x0\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "cntw x12\n"
    "ldp x11, x10, [x19, #0x0]\n"
    "sub x9, XZR, x12\n"
    "ldp x28, x27, [x19, #0x10]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1w { z17.s }, p3/Z, [x15]\n"
    "cmp x12, %x[n_channels]\n"
    "ld1w { z0.s }, p3/Z, [x15, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x15, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x15, #3, MUL VL]\n"
    "ld1w { z3.s }, p3/Z, [x15, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x15, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x15, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x15, #7, MUL VL]\n"
    "addvl x15, x15, #16\n"
    "ldp x26, x25, [x14, #0x0]\n"
    "ld1w { z7.s }, p3/Z, [x15, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x15, #-7, MUL VL]\n"
    "addvl x15, x15, #-6\n"
    "ld1w { z9.s }, p2/Z, [x26, x13, LSL #2]\n"
    "ld1w { z10.s }, p2/Z, [x25, x13, LSL #2]\n"
    "ldp x24, x23, [x14, #0x10]\n"
    "ldp x22, x21, [x14, #0x20]\n"
    "ldp x20, x19, [x14, #0x30]\n"
    "ld1w { z11.s }, p2/Z, [x24, x13, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x23, x13, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x22, x13, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x21, x13, LSL #2]\n"
    "ld1w { z15.s }, p2/Z, [x20, x13, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x19, x13, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z31, z17\n fmla z31.s, p3/M, z8.s, z9.s\n"
    "ldr x26, [x14, #0x40]\n"
    "whilelt p1.s, x12, %x[n_channels]\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z6.s, z9.s\n"
    "ldr x25, [x14, #0x48]\n"
    "incw x9\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x24, [x14, #0x50]\n"
    "mov p0.b, p2.b\n"
    "movprfx z28, z17\n fmla z28.s, p3/M, z0.s, z9.s\n"
    "ldr x23, [x14, #0x58]\n"
    "ldr x22, [x14, #0x60]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ldr x21, [x14, #0x68]\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x25, x13, LSL #2]\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x13, LSL #2]\n"
    "ldr x20, [x14, #0x70]\n"
    "fmla z30.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x23, x13, LSL #2]\n"
    "ldr x19, [x14, #0x78]\n"
    "fmla z30.s, p3/M, z0.s, z16.s\n"
    "ldr x26, [x14, #0x80]\n"
    "fmla z29.s, p3/M, z3.s, z14.s\n"
    "ldr x25, [x14, #0x88]\n"
    "ldr x24, [x14, #0x90]\n"
    "fmla z31.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x22, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x21, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z15.s\n"
    "ld1w { z14.s }, p2/Z, [x25, x13, LSL #2]\n"
    "ldr x23, [x14, #0x98]\n"
    "fmla z31.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x23, x13, LSL #2]\n"
    "ldr x22, [x14, #0xa0]\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "ldr x21, [x14, #0xa8]\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x19, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z16.s\n"
    "ldr x20, [x14, #0xb0]\n"
    "ldr x19, [x14, #0xb8]\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "fmla z28.s, p3/M, z4.s, z13.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x22, x13, LSL #2]\n"
    "ldr x26, [x14, #0xc0]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x13, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "ld1w { z17.s }, p3/Z, [x15]\n"
    "fmla z29.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x19, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z11.s\n"
    "ld1w { z0.s }, p3/Z, [x15, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x15, #2, MUL VL]\n"
    "fmla z28.s, p3/M, z5.s, z14.s\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "ld1w { z14.s }, p2/Z, [x20, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "ld1w { z4.s }, p3/Z, [x15, #5, MUL VL]\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x13, LSL #2]\n"
    "incw x13\n"
    "fmla z29.s, p3/M, z5.s, z16.s\n"
    "ldp x26, x25, [x14, #0x0]\n"
    "whilelt p2.s, x13, %x[n_channels]\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "ldp x24, x23, [x14, #0x10]\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "ldp x22, x21, [x14, #0x20]\n"
    "ldp x20, x19, [x14, #0x30]\n"
    "fmla z28.s, p3/M, z3.s, z16.s\n"
    "ld1w { z9.s }, p1/Z, [x26, x12, LSL #2]\n"
    "fmla z29.s, p3/M, z8.s, z15.s\n"
    "ld1w { z10.s }, p1/Z, [x25, x12, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z14.s\n"
    "ld1w { z12.s }, p1/Z, [x23, x12, LSL #2]\n"
    "ld1w { z13.s }, p1/Z, [x22, x12, LSL #2]\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "ld1w { z14.s }, p1/Z, [x21, x12, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p1/Z, [x20, x12, LSL #2]\n"
    "ld1w { z16.s }, p1/Z, [x19, x12, LSL #2]\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z31.s }, p0, [x11, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x24, x12, LSL #2]\n"
    "incw x12\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "st1w { z30.s }, p0, [x10, x9, LSL #2]\n"
    "cmp x12, %x[n_channels]\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "st1w { z29.s }, p0, [x28, x9, LSL #2]\n"
    "ld1w { z2.s }, p3/Z, [x15, #3, MUL VL]\n"
    "ld1w { z3.s }, p3/Z, [x15, #4, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x15, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x15, #7, MUL VL]\n"
    "addvl x15, x15, #16\n"
    "st1w { z28.s }, p0, [x27, x9, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x15, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x15, #-7, MUL VL]\n"
    "addvl x15, x15, #-6\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z31, z17\n fmla z31.s, p3/M, z8.s, z9.s\n"
    "ldr x26, [x14, #0x40]\n"
    "incw x9\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z6.s, z9.s\n"
    "ldr x25, [x14, #0x48]\n"
    "mov p0.b, p2.b\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "ldr x24, [x14, #0x50]\n"
    "movprfx z28, z17\n fmla z28.s, p3/M, z0.s, z9.s\n"
    "ldr x23, [x14, #0x58]\n"
    "ldr x22, [x14, #0x60]\n"
    "fmla z31.s, p3/M, z0.s, z10.s\n"
    "ldr x21, [x14, #0x68]\n"
    "fmla z30.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x25, x13, LSL #2]\n"
    "fmla z31.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x13, LSL #2]\n"
    "ldr x20, [x14, #0x70]\n"
    "fmla z30.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z31.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x23, x13, LSL #2]\n"
    "ldr x19, [x14, #0x78]\n"
    "fmla z30.s, p3/M, z0.s, z16.s\n"
    "ldr x26, [x14, #0x80]\n"
    "fmla z29.s, p3/M, z3.s, z14.s\n"
    "ldr x25, [x14, #0x88]\n"
    "ldr x24, [x14, #0x90]\n"
    "fmla z31.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x22, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x21, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z15.s\n"
    "ld1w { z14.s }, p2/Z, [x25, x13, LSL #2]\n"
    "ldr x23, [x14, #0x98]\n"
    "fmla z31.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x23, x13, LSL #2]\n"
    "ldr x22, [x14, #0xa0]\n"
    "fmla z31.s, p3/M, z5.s, z13.s\n"
    "ldr x21, [x14, #0xa8]\n"
    "fmla z30.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x19, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z1.s, z16.s\n"
    "ldr x20, [x14, #0xb0]\n"
    "ldr x19, [x14, #0xb8]\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "fmla z28.s, p3/M, z4.s, z13.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x22, x13, LSL #2]\n"
    "ldr x26, [x14, #0xc0]\n"
    "fmla z31.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x13, LSL #2]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x19, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z11.s\n"
    "fmla z28.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x20, x13, LSL #2]\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmla z28.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x13, LSL #2]\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x11, x9, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z16.s\n"
    "fmla z28.s, p3/M, z3.s, z16.s\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "st1w { z30.s }, p0, [x10, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z14.s\n"
    "fmla z29.s, p3/M, z8.s, z15.s\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z29.s }, p0, [x28, x9, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "st1w { z28.s }, p0, [x27, x9, LSL #2]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // __aarch64__ && defined(ARM_COMPUTE_ENABLE_SVE)
