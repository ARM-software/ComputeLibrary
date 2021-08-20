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

namespace arm_conv {
namespace depthwise {

void a64_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_indirect_impl(
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
    "ldr x21, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "add x16, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x20, %x[params_struct], %[offsetof_args_min]\n"
    "add x19, %x[params_struct], %[offsetof_args_max]\n"
    "ld1r { v18.4s }, [x20]\n"
    "ld1r { v17.4s }, [x19]\n"
    "mov x14, #0x0\n"
    "ldp x13, x12, [x21, #0x0]\n"
    "mov x11, #0x10\n" // cntb _, ALL, #1
    "ldp x10, x9, [x21, #0x10]\n"
    "sub x28, XZR, x11\n"
    "lsr x27, %x[n_channels], #0x2\n"
    "cbz x27, 3f\n"
    "ldr q16, [x15, #0x0]\n"
    "ldr q0, [x15, #0x10]\n"
    "cmp x11, x27, LSL #4\n"
    "ldr q1, [x15, #0x20]\n"
    "ldr q2, [x15, #0x30]\n"
    "ldr q3, [x15, #0x40]\n"
    "ldr q4, [x15, #0x50]\n"
    "ldr q5, [x15, #0x60]\n"
    "ldr q6, [x15, #0x70]\n"
    "ldr q7, [x15, #0x80]\n"
    "ldr q8, [x15, #0x90]\n"
    "add x15, x15, #0xa0\n"
    "ldp x26, x25, [x16, #0x0]\n"
    "ldp x24, x23, [x16, #0x10]\n"
    "ldr x22, [x16, #0x20]\n"
    "ldr q9, [x26, x14]\n"
    "ldr q10, [x25, x14]\n"
    "ldr q11, [x24, x14]\n"
    "ldr q12, [x23, x14]\n"
    "ldr q13, [x22, x14]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "ldr x21, [x16, #0x28]\n"
    "add x28, x28, #0x10\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "ldr x20, [x16, #0x30]\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "ldr x19, [x16, #0x38]\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "ldr q9, [x21, x14]\n"
    "ldr x26, [x16, #0x40]\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "ldr x25, [x16, #0x48]\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "ldr q11, [x20, x14]\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "ldr q10, [x25, x14]\n"
    "ldr x24, [x16, #0x50]\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "ldr x23, [x16, #0x58]\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "fmla v29.4s, v6.4s, v9.4s\n"
    "ldr q12, [x19, x14]\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "ldr q9, [x26, x14]\n"
    "ldr x22, [x16, #0x60]\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "ldr x21, [x16, #0x68]\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "ldr x20, [x16, #0x70]\n"
    "fmla v28.4s, v8.4s, v11.4s\n"
    "ldr q11, [x24, x14]\n"
    "ldr x19, [x16, #0x78]\n"
    "fmla v31.4s, v1.4s, v12.4s\n"
    "ldp x26, x25, [x16, #0x0]\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "ldr q12, [x23, x14]\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "ldp x24, x23, [x16, #0x10]\n"
    "ldr q16, [x15, #0x0]\n"
    "fmla v31.4s, v2.4s, v9.4s\n"
    "ldr q4, [x15, #0x50]\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "ldr q9, [x22, x14]\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "ldr x22, [x16, #0x20]\n"
    "ldr q0, [x15, #0x10]\n"
    "fmla v31.4s, v8.4s, v10.4s\n"
    "ldr q1, [x15, #0x20]\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "ldr q10, [x21, x14]\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "ldr q13, [x22, x11]\n"
    "fmla v31.4s, v3.4s, v11.4s\n"
    "ldr q11, [x20, x14]\n"
    "ldr q2, [x15, #0x30]\n"
    "fmla v30.4s, v5.4s, v12.4s\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "ldr q12, [x19, x14]\n"
    "add x14, x14, #0x10\n"
    "fmla v31.4s, v6.4s, v9.4s\n"
    "ldr q9, [x26, x11]\n"
    "fmla v29.4s, v7.4s, v11.4s\n"
    "ldr q3, [x15, #0x40]\n"
    "fmla v30.4s, v8.4s, v10.4s\n"
    "ldr q10, [x25, x11]\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "ldr q11, [x24, x11]\n"
    "ldr q5, [x15, #0x60]\n"
    "fmla v29.4s, v8.4s, v12.4s\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "ldr q6, [x15, #0x70]\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "ldr q8, [x15, #0x90]\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "ldr q12, [x23, x11]\n"
    "add x11, x11, #0x10\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "ldr q7, [x15, #0x80]\n"
    "cmp x11, x27, LSL #4\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "str q31, [x13, x28]\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "add x15, x15, #0xa0\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "str q30, [x12, x28]\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "str q29, [x10, x28]\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "str q28, [x9, x28]\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "ldr x21, [x16, #0x28]\n"
    "add x28, x28, #0x10\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "ldr x20, [x16, #0x30]\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "ldr x19, [x16, #0x38]\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "ldr q9, [x21, x14]\n"
    "ldr x26, [x16, #0x40]\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "ldr x25, [x16, #0x48]\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "ldr q11, [x20, x14]\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "ldr q10, [x25, x14]\n"
    "ldr x24, [x16, #0x50]\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "ldr x23, [x16, #0x58]\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "fmla v29.4s, v6.4s, v9.4s\n"
    "ldr q12, [x19, x14]\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "ldr q9, [x26, x14]\n"
    "ldr x22, [x16, #0x60]\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "ldr x21, [x16, #0x68]\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "ldr x20, [x16, #0x70]\n"
    "fmla v28.4s, v8.4s, v11.4s\n"
    "ldr q11, [x24, x14]\n"
    "ldr x19, [x16, #0x78]\n"
    "fmla v31.4s, v1.4s, v12.4s\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "ldr q12, [x23, x14]\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "fmla v31.4s, v2.4s, v9.4s\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "ldr q9, [x22, x14]\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "fmla v31.4s, v8.4s, v10.4s\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "ldr q10, [x21, x14]\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "fmla v31.4s, v3.4s, v11.4s\n"
    "ldr q11, [x20, x14]\n"
    "fmla v30.4s, v5.4s, v12.4s\n"
    "ldr q12, [x19, x14]\n"
    "add x14, x14, #0x10\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "fmla v31.4s, v6.4s, v9.4s\n"
    "fmla v29.4s, v7.4s, v11.4s\n"
    "fmla v30.4s, v8.4s, v10.4s\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "fmla v29.4s, v8.4s, v12.4s\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "str q31, [x13, x28]\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "str q30, [x12, x28]\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "str q29, [x10, x28]\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "str q28, [x9, x28]\n"
    "3:"  // Oddments
    "tst %x[n_channels], #0x3\n"
    "beq 30f\n"
    "ldr q16, [x15, #0x0]\n"
    "ldr q0, [x15, #0x10]\n"
    "mov x28, x14\n"
    "ldr q1, [x15, #0x20]\n"
    "add x13, x13, x28\n"
    "ldr q2, [x15, #0x30]\n"
    "add x12, x12, x28\n"
    "ldr q3, [x15, #0x40]\n"
    "add x10, x10, x28\n"
    "ldr q4, [x15, #0x50]\n"
    "add x9, x9, x28\n"
    "ldr q5, [x15, #0x60]\n"
    "ldr q6, [x15, #0x70]\n"
    "ldr q7, [x15, #0x80]\n"
    "ldr q8, [x15, #0x90]\n"
    "ldr x26, [x16, #0x0]\n"
    "ldr x25, [x16, #0x8]\n"
    "add x26, x26, x14\n"
    "ldr x24, [x16, #0x10]\n"
    "ldr x23, [x16, #0x18]\n"
    "add x25, x25, x14\n"
    "ldr x22, [x16, #0x20]\n"
    "add x24, x24, x14\n"
    "add x23, x23, x14\n"
    "add x22, x22, x14\n"
    "tbz %x[n_channels], #1, 4f\n"
    "ld1 { v9.d }[0], [x26], #0x8\n"
    "ld1 { v10.d }[0], [x25], #0x8\n"
    "ld1 { v11.d }[0], [x24], #0x8\n"
    "ld1 { v12.d }[0], [x23], #0x8\n"
    "ld1 { v13.d }[0], [x22], #0x8\n"
    "tbz %x[n_channels], #0, 5f\n"
    "ld1 { v9.s }[2], [x26], #0x4\n"
    "ld1 { v10.s }[2], [x25], #0x4\n"
    "ld1 { v11.s }[2], [x24], #0x4\n"
    "ld1 { v12.s }[2], [x23], #0x4\n"
    "ld1 { v13.s }[2], [x22], #0x4\n"
    "b 5f\n"
    "4:"  // Oddments: Load inputs (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 1: Unset
    "ld1 { v9.s }[0], [x26], #0x4\n"
    "ld1 { v10.s }[0], [x25], #0x4\n"
    "ld1 { v11.s }[0], [x24], #0x4\n"
    "ld1 { v12.s }[0], [x23], #0x4\n"
    "ld1 { v13.s }[0], [x22], #0x4\n"
    "5:"  // Oddments: Load inputs (1, 1), (0, 0), (0, 3), (1, 2), (2, 1): Bit 1: End
    "mov v31.16b, v16.16b\n fmla v31.4s, v4.4s, v9.4s\n"
    "ldr x21, [x16, #0x28]\n"
    "add x21, x21, x14\n"
    "mov v30.16b, v16.16b\n fmla v30.4s, v3.4s, v9.4s\n"
    "mov v29.16b, v16.16b\n fmla v29.4s, v1.4s, v9.4s\n"
    "mov v28.16b, v16.16b\n fmla v28.4s, v0.4s, v9.4s\n"
    "fmla v31.4s, v0.4s, v10.4s\n"
    "fmla v30.4s, v2.4s, v11.4s\n"
    "fmla v29.4s, v2.4s, v12.4s\n"
    "fmla v28.4s, v1.4s, v12.4s\n"
    "fmla v31.4s, v5.4s, v12.4s\n"
    "fmla v30.4s, v4.4s, v12.4s\n"
    "tbz %x[n_channels], #1, 6f\n"
    "ld1 { v9.d }[0], [x21], #0x8\n"
    "tbz %x[n_channels], #0, 7f\n"
    "ld1 { v9.s }[2], [x21], #0x4\n"
    "b 7f\n"
    "6:"  // Oddments: Load input (3, 0): Bit 1: Unset
    "ld1 { v9.s }[0], [x21], #0x4\n"
    "7:"  // Oddments: Load input (3, 0): Bit 1: End
    "fmla v29.4s, v6.4s, v9.4s\n"
    "ldr x20, [x16, #0x30]\n"
    "fmla v31.4s, v7.4s, v13.4s\n"
    "add x20, x20, x14\n"
    "fmla v30.4s, v6.4s, v13.4s\n"
    "fmla v28.4s, v3.4s, v13.4s\n"
    "fmla v29.4s, v4.4s, v13.4s\n"
    "tbz %x[n_channels], #1, 8f\n"
    "ld1 { v11.d }[0], [x20], #0x8\n"
    "tbz %x[n_channels], #0, 9f\n"
    "ld1 { v11.s }[2], [x20], #0x4\n"
    "b 9f\n"
    "8:"  // Oddments: Load input (3, 3): Bit 1: Unset
    "ld1 { v11.s }[0], [x20], #0x4\n"
    "9:"  // Oddments: Load input (3, 3): Bit 1: End
    "fmla v28.4s, v8.4s, v11.4s\n"
    "ldr x19, [x16, #0x38]\n"
    "add x19, x19, x14\n"
    "tbz %x[n_channels], #1, 10f\n"
    "ld1 { v12.d }[0], [x19], #0x8\n"
    "tbz %x[n_channels], #0, 11f\n"
    "ld1 { v12.s }[2], [x19], #0x4\n"
    "b 11f\n"
    "10:"  // Oddments: Load input (0, 1): Bit 1: Unset
    "ld1 { v12.s }[0], [x19], #0x4\n"
    "11:"  // Oddments: Load input (0, 1): Bit 1: End
    "fmla v31.4s, v1.4s, v12.4s\n"
    "ldr x26, [x16, #0x40]\n"
    "fmla v30.4s, v0.4s, v12.4s\n"
    "add x26, x26, x14\n"
    "tbz %x[n_channels], #1, 12f\n"
    "ld1 { v9.d }[0], [x26], #0x8\n"
    "tbz %x[n_channels], #0, 13f\n"
    "ld1 { v9.s }[2], [x26], #0x4\n"
    "b 13f\n"
    "12:"  // Oddments: Load input (0, 2): Bit 1: Unset
    "ld1 { v9.s }[0], [x26], #0x4\n"
    "13:"  // Oddments: Load input (0, 2): Bit 1: End
    "fmla v31.4s, v2.4s, v9.4s\n"
    "ldr x25, [x16, #0x48]\n"
    "fmla v30.4s, v1.4s, v9.4s\n"
    "add x25, x25, x14\n"
    "tbz %x[n_channels], #1, 14f\n"
    "ld1 { v10.d }[0], [x25], #0x8\n"
    "tbz %x[n_channels], #0, 15f\n"
    "ld1 { v10.s }[2], [x25], #0x4\n"
    "b 15f\n"
    "14:"  // Oddments: Load input (2, 2): Bit 1: Unset
    "ld1 { v10.s }[0], [x25], #0x4\n"
    "15:"  // Oddments: Load input (2, 2): Bit 1: End
    "fmla v31.4s, v8.4s, v10.4s\n"
    "ldr x24, [x16, #0x50]\n"
    "fmla v30.4s, v7.4s, v10.4s\n"
    "add x24, x24, x14\n"
    "fmla v29.4s, v5.4s, v10.4s\n"
    "fmla v28.4s, v4.4s, v10.4s\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ld1 { v11.d }[0], [x24], #0x8\n"
    "tbz %x[n_channels], #0, 17f\n"
    "ld1 { v11.s }[2], [x24], #0x4\n"
    "b 17f\n"
    "16:"  // Oddments: Load input (1, 0): Bit 1: Unset
    "ld1 { v11.s }[0], [x24], #0x4\n"
    "17:"  // Oddments: Load input (1, 0): Bit 1: End
    "fmla v31.4s, v3.4s, v11.4s\n"
    "ldr x23, [x16, #0x58]\n"
    "fmla v29.4s, v0.4s, v11.4s\n"
    "add x23, x23, x14\n"
    "tbz %x[n_channels], #1, 18f\n"
    "ld1 { v12.d }[0], [x23], #0x8\n"
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v12.s }[2], [x23], #0x4\n"
    "b 19f\n"
    "18:"  // Oddments: Load input (1, 3): Bit 1: Unset
    "ld1 { v12.s }[0], [x23], #0x4\n"
    "19:"  // Oddments: Load input (1, 3): Bit 1: End
    "fmla v30.4s, v5.4s, v12.4s\n"
    "ldr x22, [x16, #0x60]\n"
    "fmla v28.4s, v2.4s, v12.4s\n"
    "add x22, x22, x14\n"
    "tbz %x[n_channels], #1, 20f\n"
    "ld1 { v9.d }[0], [x22], #0x8\n"
    "tbz %x[n_channels], #0, 21f\n"
    "ld1 { v9.s }[2], [x22], #0x4\n"
    "b 21f\n"
    "20:"  // Oddments: Load input (2, 0): Bit 1: Unset
    "ld1 { v9.s }[0], [x22], #0x4\n"
    "21:"  // Oddments: Load input (2, 0): Bit 1: End
    "fmla v31.4s, v6.4s, v9.4s\n"
    "ldr x21, [x16, #0x68]\n"
    "fmla v29.4s, v3.4s, v9.4s\n"
    "add x21, x21, x14\n"
    "tbz %x[n_channels], #1, 22f\n"
    "ld1 { v10.d }[0], [x21], #0x8\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v10.s }[2], [x21], #0x4\n"
    "b 23f\n"
    "22:"  // Oddments: Load input (2, 3): Bit 1: Unset
    "ld1 { v10.s }[0], [x21], #0x4\n"
    "23:"  // Oddments: Load input (2, 3): Bit 1: End
    "fmla v30.4s, v8.4s, v10.4s\n"
    "ldr x20, [x16, #0x70]\n"
    "fmla v28.4s, v5.4s, v10.4s\n"
    "add x20, x20, x14\n"
    "tbz %x[n_channels], #1, 24f\n"
    "ld1 { v11.d }[0], [x20], #0x8\n"
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v11.s }[2], [x20], #0x4\n"
    "b 25f\n"
    "24:"  // Oddments: Load input (3, 1): Bit 1: Unset
    "ld1 { v11.s }[0], [x20], #0x4\n"
    "25:"  // Oddments: Load input (3, 1): Bit 1: End
    "fmla v29.4s, v7.4s, v11.4s\n"
    "ldr x19, [x16, #0x78]\n"
    "fmla v28.4s, v6.4s, v11.4s\n"
    "add x19, x19, x14\n"
    "tbz %x[n_channels], #1, 26f\n"
    "ld1 { v12.d }[0], [x19], #0x8\n"
    "tbz %x[n_channels], #0, 27f\n"
    "ld1 { v12.s }[2], [x19], #0x4\n"
    "b 27f\n"
    "26:"  // Oddments: Load input (3, 2): Bit 1: Unset
    "ld1 { v12.s }[0], [x19], #0x4\n"
    "27:"  // Oddments: Load input (3, 2): Bit 1: End
    "fmla v29.4s, v8.4s, v12.4s\n"
    "fmla v28.4s, v7.4s, v12.4s\n"
    "fmax v31.4s, v31.4s, v18.4s\n"
    "fmax v30.4s, v30.4s, v18.4s\n"
    "fmin v31.4s, v31.4s, v17.4s\n"
    "fmax v29.4s, v29.4s, v18.4s\n"
    "fmin v30.4s, v30.4s, v17.4s\n"
    "fmax v28.4s, v28.4s, v18.4s\n"
    "fmin v29.4s, v29.4s, v17.4s\n"
    "fmin v28.4s, v28.4s, v17.4s\n"
    "tbz %x[n_channels], #1, 28f\n"
    "st1 { v31.d }[0], [x13], #0x8\n"
    "st1 { v30.d }[0], [x12], #0x8\n"
    "st1 { v29.d }[0], [x10], #0x8\n"
    "st1 { v28.d }[0], [x9], #0x8\n"
    "tbz %x[n_channels], #0, 29f\n"
    "st1 { v31.s }[2], [x13], #0x4\n"
    "st1 { v30.s }[2], [x12], #0x4\n"
    "st1 { v29.s }[2], [x10], #0x4\n"
    "st1 { v28.s }[2], [x9], #0x4\n"
    "b 29f\n"
    "28:"  // Oddments: Store: Bit 1: Unset
    "st1 { v31.s }[0], [x13], #0x4\n"
    "st1 { v30.s }[0], [x12], #0x4\n"
    "st1 { v29.s }[0], [x10], #0x4\n"
    "st1 { v28.s }[0], [x9], #0x4\n"
    "29:"  // Oddments: Store: Bit 1: End

    "30:"  // End

    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv
