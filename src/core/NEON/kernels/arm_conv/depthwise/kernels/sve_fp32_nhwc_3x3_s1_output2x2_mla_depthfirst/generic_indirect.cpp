/*
 * Copyright (c) 2021, 2023 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

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
      inptrs[0] = input_ptrs[5];
      inptrs[1] = input_ptrs[0];
      inptrs[2] = input_ptrs[3];
      inptrs[3] = input_ptrs[6];
      inptrs[4] = input_ptrs[9];
      inptrs[5] = input_ptrs[12];
      inptrs[6] = input_ptrs[15];
      inptrs[7] = input_ptrs[1];
      inptrs[8] = input_ptrs[2];
      inptrs[9] = input_ptrs[10];
      inptrs[10] = input_ptrs[4];
      inptrs[11] = input_ptrs[7];
      inptrs[12] = input_ptrs[8];
      inptrs[13] = input_ptrs[11];
      inptrs[14] = input_ptrs[13];
      inptrs[15] = input_ptrs[14];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ptrue p3.b\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x15, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "cntw x14\n"
    "ldp x13, x12, [x20, #0x0]\n"
    "ldp x11, x10, [x20, #0x10]\n"
    "mov x9, #0x0\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1w { z20.s }, p3/Z, [x16]\n"
    "ld1w { z0.s }, p3/Z, [x16, #1, MUL VL]\n"
    "cmp x14, %x[n_channels]\n"
    "ld1w { z1.s }, p3/Z, [x16, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x16, #3, MUL VL]\n"
    "sub x28, XZR, x14\n"
    "ld1w { z3.s }, p3/Z, [x16, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x16, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x16, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x16, #7, MUL VL]\n"
    "addvl x16, x16, #16\n"
    "ldp x24, x23, [x15, #0x0]\n"
    "ldp x22, x21, [x15, #0x10]\n"
    "ldr x20, [x15, #0x20]\n"
    "ld1rw { z26.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ld1rw { z25.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1w { z7.s }, p3/Z, [x16, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x16, #-7, MUL VL]\n"
    "ld1w { z9.s }, p2/Z, [x24, x9, LSL #2]\n"
    "addvl x16, x16, #-6\n"
    "ld1w { z10.s }, p2/Z, [x23, x9, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x22, x9, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x20, x9, LSL #2]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z24, z20\n fmla z24.s, p3/M, z4.s, z9.s\n"
    "movprfx z23, z20\n fmla z23.s, p3/M, z3.s, z9.s\n"
    "ldr x21, [x15, #0x28]\n"
    "ldr x20, [x15, #0x30]\n"
    "movprfx z22, z20\n fmla z22.s, p3/M, z1.s, z9.s\n"
    "movprfx z21, z20\n fmla z21.s, p3/M, z0.s, z9.s\n"
    "ld1w { z18.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x22, [x15, #0x38]\n"
    "fmla z24.s, p3/M, z0.s, z10.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x15, #0x48]\n"
    "fmla z22.s, p3/M, z2.s, z12.s\n"
    "fmla z21.s, p3/M, z1.s, z12.s\n"
    "ldr x20, [x15, #0x40]\n"
    "ld1w { z20.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z24.s, p3/M, z5.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x22, x9, LSL #2]\n"
    "ldr x22, [x15, #0x50]\n"
    "fmla z22.s, p3/M, z6.s, z18.s\n"
    "fmla z21.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x15, #0x58]\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "ldr x20, [x15, #0x60]\n"
    "ldr x27, [x15, #0x68]\n"
    "fmla z22.s, p3/M, z4.s, z13.s\n"
    "fmla z21.s, p3/M, z8.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x22, x9, LSL #2]\n"
    "ldr x26, [x15, #0x70]\n"
    "fmla z24.s, p3/M, z1.s, z16.s\n"
    "fmla z23.s, p3/M, z0.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x25, [x15, #0x78]\n"
    "fmla z22.s, p3/M, z5.s, z20.s\n"
    "fmla z21.s, p3/M, z4.s, z20.s\n"
    "whilelt p1.s, x14, %x[n_channels]\n"
    "ldp x24, x23, [x15, #0x0]\n"
    "fmla z24.s, p3/M, z2.s, z18.s\n"
    "fmla z23.s, p3/M, z1.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldp x22, x21, [x15, #0x10]\n"
    "fmla z22.s, p3/M, z0.s, z17.s\n"
    "fmla z21.s, p3/M, z2.s, z16.s\n"
    "ldr x20, [x15, #0x20]\n"
    "ld1w { z13.s }, p1/Z, [x20, x14, LSL #2]\n"
    "fmla z24.s, p3/M, z8.s, z20.s\n"
    "fmla z23.s, p3/M, z7.s, z20.s\n"
    "ld1w { z18.s }, p2/Z, [x27, x9, LSL #2]\n"
    "incw x28\n"
    "fmla z22.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z5.s, z18.s\n"
    "mov p0.b, p2.b\n"
    "ld1w { z20.s }, p3/Z, [x16]\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z22.s, p3/M, z7.s, z17.s\n"
    "fmla z21.s, p3/M, z6.s, z17.s\n"
    "incw x9\n"
    "ld1w { z11.s }, p1/Z, [x22, x14, LSL #2]\n"
    "fmla z24.s, p3/M, z6.s, z19.s\n"
    "fmla z23.s, p3/M, z8.s, z18.s\n"
    "ld1w { z9.s }, p1/Z, [x24, x14, LSL #2]\n"
    "ld1w { z10.s }, p1/Z, [x23, x14, LSL #2]\n"
    "fmla z22.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "ld1w { z12.s }, p1/Z, [x21, x14, LSL #2]\n"
    "incw x14\n"
    "fmax z24.s, p3/M, z24.s, z26.s\n"
    "fmax z23.s, p3/M, z23.s, z26.s\n"
    "ld1w { z0.s }, p3/Z, [x16, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x16, #2, MUL VL]\n"
    "fmax z22.s, p3/M, z22.s, z26.s\n"
    "fmax z21.s, p3/M, z21.s, z26.s\n"
    "ld1w { z2.s }, p3/Z, [x16, #3, MUL VL]\n"
    "ld1w { z3.s }, p3/Z, [x16, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x16, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x16, #6, MUL VL]\n"
    "whilelt p2.s, x9, %x[n_channels]\n"
    "cmp x14, %x[n_channels]\n"
    "ld1w { z6.s }, p3/Z, [x16, #7, MUL VL]\n"
    "addvl x16, x16, #16\n"
    "fmin z24.s, p3/M, z24.s, z25.s\n"
    "st1w { z24.s }, p0, [x13, x28, LSL #2]\n"
    "fmin z23.s, p3/M, z23.s, z25.s\n"
    "fmin z22.s, p3/M, z22.s, z25.s\n"
    "st1w { z23.s }, p0, [x12, x28, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x16, #-8, MUL VL]\n"
    "fmin z21.s, p3/M, z21.s, z25.s\n"
    "st1w { z22.s }, p0, [x11, x28, LSL #2]\n"
    "ld1w { z8.s }, p3/Z, [x16, #-7, MUL VL]\n"
    "addvl x16, x16, #-6\n"
    "st1w { z21.s }, p0, [x10, x28, LSL #2]\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z24, z20\n fmla z24.s, p3/M, z4.s, z9.s\n"
    "movprfx z23, z20\n fmla z23.s, p3/M, z3.s, z9.s\n"
    "ldr x21, [x15, #0x28]\n"
    "ldr x20, [x15, #0x30]\n"
    "movprfx z22, z20\n fmla z22.s, p3/M, z1.s, z9.s\n"
    "movprfx z21, z20\n fmla z21.s, p3/M, z0.s, z9.s\n"
    "ld1w { z18.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x22, [x15, #0x38]\n"
    "fmla z24.s, p3/M, z0.s, z10.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "ld1w { z17.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x21, [x15, #0x48]\n"
    "fmla z22.s, p3/M, z2.s, z12.s\n"
    "fmla z21.s, p3/M, z1.s, z12.s\n"
    "ldr x20, [x15, #0x40]\n"
    "ld1w { z20.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z24.s, p3/M, z5.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "ld1w { z16.s }, p2/Z, [x22, x9, LSL #2]\n"
    "ldr x21, [x15, #0x50]\n"
    "fmla z22.s, p3/M, z6.s, z18.s\n"
    "fmla z21.s, p3/M, z3.s, z13.s\n"
    "ld1w { z18.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x15, #0x58]\n"
    "fmla z24.s, p3/M, z7.s, z13.s\n"
    "fmla z23.s, p3/M, z6.s, z13.s\n"
    "ldr x23, [x15, #0x60]\n"
    "ldr x22, [x15, #0x68]\n"
    "fmla z22.s, p3/M, z4.s, z13.s\n"
    "fmla z21.s, p3/M, z8.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x21, x9, LSL #2]\n"
    "ldr x21, [x15, #0x70]\n"
    "fmla z24.s, p3/M, z1.s, z16.s\n"
    "fmla z23.s, p3/M, z0.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "ldr x20, [x15, #0x78]\n"
    "fmla z22.s, p3/M, z5.s, z20.s\n"
    "fmla z21.s, p3/M, z4.s, z20.s\n"
    "incw x28\n"
    "mov p0.b, p2.b\n"
    "fmla z24.s, p3/M, z2.s, z18.s\n"
    "fmla z23.s, p3/M, z1.s, z18.s\n"
    "ld1w { z19.s }, p2/Z, [x23, x9, LSL #2]\n"
    "fmla z22.s, p3/M, z0.s, z17.s\n"
    "fmla z21.s, p3/M, z2.s, z16.s\n"
    "fmla z24.s, p3/M, z8.s, z20.s\n"
    "fmla z23.s, p3/M, z7.s, z20.s\n"
    "ld1w { z18.s }, p2/Z, [x22, x9, LSL #2]\n"
    "fmla z22.s, p3/M, z3.s, z19.s\n"
    "fmla z21.s, p3/M, z5.s, z18.s\n"
    "fmla z24.s, p3/M, z3.s, z17.s\n"
    "ld1w { z17.s }, p2/Z, [x21, x9, LSL #2]\n"
    "fmla z23.s, p3/M, z5.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x20, x9, LSL #2]\n"
    "fmla z22.s, p3/M, z7.s, z17.s\n"
    "fmla z21.s, p3/M, z6.s, z17.s\n"
    "fmla z24.s, p3/M, z6.s, z19.s\n"
    "fmla z23.s, p3/M, z8.s, z18.s\n"
    "fmax z24.s, p3/M, z24.s, z26.s\n"
    "fmax z23.s, p3/M, z23.s, z26.s\n"
    "fmla z22.s, p3/M, z8.s, z16.s\n"
    "fmla z21.s, p3/M, z7.s, z16.s\n"
    "fmax z22.s, p3/M, z22.s, z26.s\n"
    "fmax z21.s, p3/M, z21.s, z26.s\n"
    "fmin z24.s, p3/M, z24.s, z25.s\n"
    "fmin z23.s, p3/M, z23.s, z25.s\n"
    "st1w { z24.s }, p0, [x13, x28, LSL #2]\n"
    "fmin z22.s, p3/M, z22.s, z25.s\n"
    "fmin z21.s, p3/M, z21.s, z25.s\n"
    "st1w { z23.s }, p0, [x12, x28, LSL #2]\n"
    "st1w { z22.s }, p0, [x11, x28, LSL #2]\n"
    "st1w { z21.s }, p0, [x10, x28, LSL #2]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
