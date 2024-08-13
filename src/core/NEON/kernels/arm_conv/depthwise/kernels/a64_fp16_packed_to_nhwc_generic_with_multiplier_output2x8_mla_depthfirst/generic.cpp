/*
 * Copyright (c) 2021-2024 Arm Limited.
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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace depthwise {

void a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl(
  const __fp16 *const *const inptrs,
  __fp16 *const *const outptrs,
  const __fp16 *weights,
  const __fp16 *bias,
  const unsigned int kernel_points,
  const unsigned int n_output_channels,
  const __fp16 activation_min,
  const __fp16 activation_max
)
{
  const __fp16 minmax_vals[2] = { activation_min, activation_max };

  __asm__ __volatile__(
    "ld1r { v8.8h }, [%x[minmax_vals]]\n"
    "lsr x11, %x[n_output_channels], #0x3\n"
    "add x20, %x[minmax_vals], #0x2\n"
    "ld1r { v7.8h }, [x20]\n"
    "mov x10, #0x0\n"
    "cbz x11, 8f\n"
    "1:"  // Output channel loop
    "movi v31.16b, #0x0\n"
    "cbz %x[bias], 2f\n"
    "lsl x20, x10, #0x1\n"
    "ldr q31, [%x[bias], x20]\n"
    "2:"  // Output channel loop: Load bias: Done
    "ldr q6, [%x[weights], #0x0]\n"
    "mov x22, %x[inptrs]\n"
    "lsr x23, %x[kernel_points], #0x1\n"
    "mov v16.16b, v31.16b\n"
    "mov v17.16b, v31.16b\n"
    "mov v18.16b, v31.16b\n"
    "add %x[weights], %x[weights], #0x10\n"
    "mov v19.16b, v31.16b\n"
    "mov v20.16b, v31.16b\n"
    "ldp x21, x20, [x22], #0x10\n"
    "mov v21.16b, v31.16b\n"
    "mov v22.16b, v31.16b\n"
    "mov v23.16b, v31.16b\n"
    "mov v24.16b, v31.16b\n"
    "ldr q1, [x21, #0x0]\n"
    "ldr q0, [x20, #0x0]\n"
    "mov v25.16b, v31.16b\n"
    "mov v26.16b, v31.16b\n"
    "mov v27.16b, v31.16b\n"
    "mov v28.16b, v31.16b\n"
    "mov v29.16b, v31.16b\n"
    "mov v30.16b, v31.16b\n"
    "mov v31.16b, v31.16b\n"
    "cbz x23, 6f\n"
    "ldr q5, [%x[weights], #0x0]\n"
    "ldp x21, x20, [x22], #0x10\n"
    "subs x23, x23, #0x1\n"
    "add %x[weights], %x[weights], #0x10\n"
    "ldr q4, [x21, #0x0]\n"
    "ldr q3, [x20, #0x0]\n"
    "beq 4f\n"
    "3:"  // Output channel loop: Kernel loop
    "ldp x21, x20, [x22], #0x10\n"
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "subs x23, x23, #0x1\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr q1, [x21, #0x0]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "ldr q0, [x20, #0x0]\n"
    "ldr q6, [%x[weights], #0x0]\n"
    "ldp x21, x20, [x22], #0x10\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "ldr q4, [x21, #0x0]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "ldr q3, [x20, #0x0]\n"
    "ldr q5, [%x[weights], #0x10]\n"
    "add %x[weights], %x[weights], #0x20\n"
    "bgt 3b\n"
    "4:"  // Output channel loop: Kernel loop tail
    "tbnz %x[kernel_points], #0, 5f\n"
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "lsl x28, x10, #0x1\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmin v16.8h, v16.8h, v7.8h\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmin v17.8h, v17.8h, v7.8h\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmin v18.8h, v18.8h, v7.8h\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "fmin v19.8h, v19.8h, v7.8h\n"
    "fmin v20.8h, v20.8h, v7.8h\n"
    "fmin v21.8h, v21.8h, v7.8h\n"
    "fmin v22.8h, v22.8h, v7.8h\n"
    "fmin v23.8h, v23.8h, v7.8h\n"
    "fmax v16.8h, v16.8h, v8.8h\n"
    "fmax v17.8h, v17.8h, v8.8h\n"
    "fmax v18.8h, v18.8h, v8.8h\n"
    "fmax v19.8h, v19.8h, v8.8h\n"
    "fmax v20.8h, v20.8h, v8.8h\n"
    "fmax v21.8h, v21.8h, v8.8h\n"
    "fmax v22.8h, v22.8h, v8.8h\n"
    "fmax v23.8h, v23.8h, v8.8h\n"
    "str q16, [x27, x28]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "fmin v24.8h, v24.8h, v7.8h\n"
    "fmin v25.8h, v25.8h, v7.8h\n"
    "str q17, [x26, x28]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "fmin v26.8h, v26.8h, v7.8h\n"
    "fmin v27.8h, v27.8h, v7.8h\n"
    "str q18, [x25, x28]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "fmin v28.8h, v28.8h, v7.8h\n"
    "fmin v29.8h, v29.8h, v7.8h\n"
    "str q19, [x24, x28]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "fmin v30.8h, v30.8h, v7.8h\n"
    "fmin v31.8h, v31.8h, v7.8h\n"
    "str q20, [x23, x28]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "str q21, [x22, x28]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "fmax v24.8h, v24.8h, v8.8h\n"
    "fmax v25.8h, v25.8h, v8.8h\n"
    "str q22, [x21, x28]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "fmax v26.8h, v26.8h, v8.8h\n"
    "fmax v27.8h, v27.8h, v8.8h\n"
    "str q23, [x20, x28]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "fmax v28.8h, v28.8h, v8.8h\n"
    "fmax v29.8h, v29.8h, v8.8h\n"
    "fmax v30.8h, v30.8h, v8.8h\n"
    "fmax v31.8h, v31.8h, v8.8h\n"
    "str q24, [x27, x28]\n"
    "str q25, [x26, x28]\n"
    "str q26, [x25, x28]\n"
    "str q27, [x24, x28]\n"
    "str q28, [x23, x28]\n"
    "str q29, [x22, x28]\n"
    "str q30, [x21, x28]\n"
    "str q31, [x20, x28]\n"
    "b 7f\n"
    "5:"  // Output channel loop: Odd tail
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "ldp x20, x9, [x22], #0x10\n"
    "lsl x28, x10, #0x1\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr q2, [x20, #0x0]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "ldr q1, [%x[weights], #0x0]\n"
    "ldr q0, [x9, #0x0]\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "add %x[weights], %x[weights], #0x10\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "fmla v16.8h, v1.8h, v2.h[0]\n"
    "fmla v17.8h, v1.8h, v2.h[1]\n"
    "fmla v18.8h, v1.8h, v2.h[2]\n"
    "fmla v19.8h, v1.8h, v2.h[3]\n"
    "fmla v20.8h, v1.8h, v2.h[4]\n"
    "fmla v21.8h, v1.8h, v2.h[5]\n"
    "fmla v22.8h, v1.8h, v2.h[6]\n"
    "fmla v23.8h, v1.8h, v2.h[7]\n"
    "fmla v24.8h, v1.8h, v0.h[0]\n"
    "fmla v25.8h, v1.8h, v0.h[1]\n"
    "fmin v16.8h, v16.8h, v7.8h\n"
    "fmla v26.8h, v1.8h, v0.h[2]\n"
    "fmla v27.8h, v1.8h, v0.h[3]\n"
    "fmin v17.8h, v17.8h, v7.8h\n"
    "fmla v28.8h, v1.8h, v0.h[4]\n"
    "fmla v29.8h, v1.8h, v0.h[5]\n"
    "fmin v18.8h, v18.8h, v7.8h\n"
    "fmla v30.8h, v1.8h, v0.h[6]\n"
    "fmla v31.8h, v1.8h, v0.h[7]\n"
    "fmin v19.8h, v19.8h, v7.8h\n"
    "fmin v20.8h, v20.8h, v7.8h\n"
    "fmin v21.8h, v21.8h, v7.8h\n"
    "fmin v22.8h, v22.8h, v7.8h\n"
    "fmin v23.8h, v23.8h, v7.8h\n"
    "fmax v16.8h, v16.8h, v8.8h\n"
    "fmax v17.8h, v17.8h, v8.8h\n"
    "fmax v18.8h, v18.8h, v8.8h\n"
    "fmax v19.8h, v19.8h, v8.8h\n"
    "fmax v20.8h, v20.8h, v8.8h\n"
    "fmax v21.8h, v21.8h, v8.8h\n"
    "fmax v22.8h, v22.8h, v8.8h\n"
    "fmax v23.8h, v23.8h, v8.8h\n"
    "str q16, [x27, x28]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "fmin v24.8h, v24.8h, v7.8h\n"
    "fmin v25.8h, v25.8h, v7.8h\n"
    "str q17, [x26, x28]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "fmin v26.8h, v26.8h, v7.8h\n"
    "fmin v27.8h, v27.8h, v7.8h\n"
    "str q18, [x25, x28]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "fmin v28.8h, v28.8h, v7.8h\n"
    "fmin v29.8h, v29.8h, v7.8h\n"
    "str q19, [x24, x28]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "fmin v30.8h, v30.8h, v7.8h\n"
    "fmin v31.8h, v31.8h, v7.8h\n"
    "str q20, [x23, x28]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "str q21, [x22, x28]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "fmax v24.8h, v24.8h, v8.8h\n"
    "fmax v25.8h, v25.8h, v8.8h\n"
    "str q22, [x21, x28]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "fmax v26.8h, v26.8h, v8.8h\n"
    "fmax v27.8h, v27.8h, v8.8h\n"
    "str q23, [x20, x28]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "fmax v28.8h, v28.8h, v8.8h\n"
    "fmax v29.8h, v29.8h, v8.8h\n"
    "fmax v30.8h, v30.8h, v8.8h\n"
    "fmax v31.8h, v31.8h, v8.8h\n"
    "str q24, [x27, x28]\n"
    "str q25, [x26, x28]\n"
    "str q26, [x25, x28]\n"
    "str q27, [x24, x28]\n"
    "str q28, [x23, x28]\n"
    "str q29, [x22, x28]\n"
    "str q30, [x21, x28]\n"
    "str q31, [x20, x28]\n"
    "b 7f\n"
    "6:"  // Output channel loop: Single kernel point
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "lsl x28, x10, #0x1\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmin v16.8h, v16.8h, v7.8h\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmin v17.8h, v17.8h, v7.8h\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmin v18.8h, v18.8h, v7.8h\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "fmin v19.8h, v19.8h, v7.8h\n"
    "fmin v20.8h, v20.8h, v7.8h\n"
    "fmin v21.8h, v21.8h, v7.8h\n"
    "fmin v22.8h, v22.8h, v7.8h\n"
    "fmin v23.8h, v23.8h, v7.8h\n"
    "fmax v16.8h, v16.8h, v8.8h\n"
    "fmax v17.8h, v17.8h, v8.8h\n"
    "fmax v18.8h, v18.8h, v8.8h\n"
    "fmax v19.8h, v19.8h, v8.8h\n"
    "fmax v20.8h, v20.8h, v8.8h\n"
    "fmax v21.8h, v21.8h, v8.8h\n"
    "fmax v22.8h, v22.8h, v8.8h\n"
    "fmax v23.8h, v23.8h, v8.8h\n"
    "str q16, [x27, x28]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "fmin v24.8h, v24.8h, v7.8h\n"
    "fmin v25.8h, v25.8h, v7.8h\n"
    "str q17, [x26, x28]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "fmin v26.8h, v26.8h, v7.8h\n"
    "fmin v27.8h, v27.8h, v7.8h\n"
    "str q18, [x25, x28]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "fmin v28.8h, v28.8h, v7.8h\n"
    "fmin v29.8h, v29.8h, v7.8h\n"
    "str q19, [x24, x28]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "fmin v30.8h, v30.8h, v7.8h\n"
    "fmin v31.8h, v31.8h, v7.8h\n"
    "str q20, [x23, x28]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "str q21, [x22, x28]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "fmax v24.8h, v24.8h, v8.8h\n"
    "fmax v25.8h, v25.8h, v8.8h\n"
    "str q22, [x21, x28]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "fmax v26.8h, v26.8h, v8.8h\n"
    "fmax v27.8h, v27.8h, v8.8h\n"
    "str q23, [x20, x28]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "fmax v28.8h, v28.8h, v8.8h\n"
    "fmax v29.8h, v29.8h, v8.8h\n"
    "fmax v30.8h, v30.8h, v8.8h\n"
    "fmax v31.8h, v31.8h, v8.8h\n"
    "str q24, [x27, x28]\n"
    "str q25, [x26, x28]\n"
    "str q26, [x25, x28]\n"
    "str q27, [x24, x28]\n"
    "str q28, [x23, x28]\n"
    "str q29, [x22, x28]\n"
    "str q30, [x21, x28]\n"
    "str q31, [x20, x28]\n"
    "7:"  // Output channel loop: Done
    "add x10, x10, #0x8\n"
    "cmp x10, x11, LSL #3\n"
    "blt 1b\n"
    "tst %x[n_output_channels], #0x7\n"
    "beq 23f\n"
    "8:"  // Output channel oddments
    "movi v31.16b, #0x0\n"
    "cbz %x[bias], 13f\n"
    "add x20, %x[bias], x10, LSL #1\n"
    "tbz %x[n_output_channels], #2, 10f\n"
    "ld1 { v31.d }[0], [x20], #0x8\n"
    "tbz %x[n_output_channels], #1, 9f\n"
    "ld1 { v31.s }[2], [x20], #0x4\n"
    "tbz %x[n_output_channels], #0, 12f\n"
    "ld1 { v31.h }[6], [x20]\n"
    "b 12f\n"
    "9:"  // Output channel oddments: Load bias: Bit 2: Bit 1: Unset
    "tbz %x[n_output_channels], #0, 12f\n"
    "ld1 { v31.h }[4], [x20]\n"
    "b 12f\n"
    "10:"  // Output channel oddments: Load bias: Bit 2: Unset
    "tbz %x[n_output_channels], #1, 11f\n"
    "ld1 { v31.s }[0], [x20], #0x4\n"
    "tbz %x[n_output_channels], #0, 12f\n"
    "ld1 { v31.h }[2], [x20]\n"
    "b 12f\n"
    "11:"  // Output channel oddments: Load bias: Bit 2: Unset: Bit 1: Unset
    "ld1 { v31.h }[0], [x20]\n"
    "12:"  // Output channel oddments: Load bias: Bit 2: End
    "13:"  // Output channel oddments: Load bias: Done
    "ldr q6, [%x[weights], #0x0]\n"
    "mov x22, %x[inptrs]\n"
    "lsr x23, %x[kernel_points], #0x1\n"
    "mov v16.16b, v31.16b\n"
    "mov v17.16b, v31.16b\n"
    "mov v18.16b, v31.16b\n"
    "add %x[weights], %x[weights], #0x10\n"
    "mov v19.16b, v31.16b\n"
    "mov v20.16b, v31.16b\n"
    "ldp x21, x20, [x22], #0x10\n"
    "mov v21.16b, v31.16b\n"
    "mov v22.16b, v31.16b\n"
    "mov v23.16b, v31.16b\n"
    "mov v24.16b, v31.16b\n"
    "ldr q1, [x21, #0x0]\n"
    "ldr q0, [x20, #0x0]\n"
    "mov v25.16b, v31.16b\n"
    "mov v26.16b, v31.16b\n"
    "mov v27.16b, v31.16b\n"
    "mov v28.16b, v31.16b\n"
    "mov v29.16b, v31.16b\n"
    "mov v30.16b, v31.16b\n"
    "mov v31.16b, v31.16b\n"
    "cbz x23, 17f\n"
    "ldr q5, [%x[weights], #0x0]\n"
    "ldp x21, x20, [x22], #0x10\n"
    "subs x23, x23, #0x1\n"
    "add %x[weights], %x[weights], #0x10\n"
    "ldr q4, [x21, #0x0]\n"
    "ldr q3, [x20, #0x0]\n"
    "beq 15f\n"
    "14:"  // Output channel oddments: Kernel loop
    "ldp x21, x20, [x22], #0x10\n"
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "subs x23, x23, #0x1\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr q1, [x21, #0x0]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "ldr q0, [x20, #0x0]\n"
    "ldr q6, [%x[weights], #0x0]\n"
    "ldp x21, x20, [x22], #0x10\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "ldr q4, [x21, #0x0]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "ldr q3, [x20, #0x0]\n"
    "ldr q5, [%x[weights], #0x10]\n"
    "add %x[weights], %x[weights], #0x20\n"
    "bgt 14b\n"
    "15:"  // Output channel oddments: Kernel loop tail
    "tbnz %x[kernel_points], #0, 16f\n"
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "b 18f\n"
    "16:"  // Output channel oddments: Odd tail
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "ldp x21, x20, [x22], #0x10\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "ldr q2, [x21, #0x0]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "ldr q1, [x20, #0x0]\n"
    "ldr q0, [%x[weights], #0x0]\n"
    "fmla v16.8h, v5.8h, v4.h[0]\n"
    "fmla v17.8h, v5.8h, v4.h[1]\n"
    "add %x[weights], %x[weights], #0x10\n"
    "fmla v18.8h, v5.8h, v4.h[2]\n"
    "fmla v19.8h, v5.8h, v4.h[3]\n"
    "fmla v20.8h, v5.8h, v4.h[4]\n"
    "fmla v21.8h, v5.8h, v4.h[5]\n"
    "fmla v22.8h, v5.8h, v4.h[6]\n"
    "fmla v23.8h, v5.8h, v4.h[7]\n"
    "fmla v24.8h, v5.8h, v3.h[0]\n"
    "fmla v25.8h, v5.8h, v3.h[1]\n"
    "fmla v26.8h, v5.8h, v3.h[2]\n"
    "fmla v27.8h, v5.8h, v3.h[3]\n"
    "fmla v28.8h, v5.8h, v3.h[4]\n"
    "fmla v29.8h, v5.8h, v3.h[5]\n"
    "fmla v30.8h, v5.8h, v3.h[6]\n"
    "fmla v31.8h, v5.8h, v3.h[7]\n"
    "fmla v16.8h, v0.8h, v2.h[0]\n"
    "fmla v17.8h, v0.8h, v2.h[1]\n"
    "fmla v18.8h, v0.8h, v2.h[2]\n"
    "fmla v19.8h, v0.8h, v2.h[3]\n"
    "fmla v20.8h, v0.8h, v2.h[4]\n"
    "fmla v21.8h, v0.8h, v2.h[5]\n"
    "fmla v22.8h, v0.8h, v2.h[6]\n"
    "fmla v23.8h, v0.8h, v2.h[7]\n"
    "fmla v24.8h, v0.8h, v1.h[0]\n"
    "fmla v25.8h, v0.8h, v1.h[1]\n"
    "fmla v26.8h, v0.8h, v1.h[2]\n"
    "fmla v27.8h, v0.8h, v1.h[3]\n"
    "fmla v28.8h, v0.8h, v1.h[4]\n"
    "fmla v29.8h, v0.8h, v1.h[5]\n"
    "fmla v30.8h, v0.8h, v1.h[6]\n"
    "fmla v31.8h, v0.8h, v1.h[7]\n"
    "b 18f\n"
    "17:"  // Output channel oddments: Single kernel point
    "fmla v16.8h, v6.8h, v1.h[0]\n"
    "fmla v17.8h, v6.8h, v1.h[1]\n"
    "fmla v18.8h, v6.8h, v1.h[2]\n"
    "fmla v19.8h, v6.8h, v1.h[3]\n"
    "fmla v20.8h, v6.8h, v1.h[4]\n"
    "fmla v21.8h, v6.8h, v1.h[5]\n"
    "fmla v22.8h, v6.8h, v1.h[6]\n"
    "fmla v23.8h, v6.8h, v1.h[7]\n"
    "fmla v24.8h, v6.8h, v0.h[0]\n"
    "fmla v25.8h, v6.8h, v0.h[1]\n"
    "fmla v26.8h, v6.8h, v0.h[2]\n"
    "fmla v27.8h, v6.8h, v0.h[3]\n"
    "fmla v28.8h, v6.8h, v0.h[4]\n"
    "fmla v29.8h, v6.8h, v0.h[5]\n"
    "fmla v30.8h, v6.8h, v0.h[6]\n"
    "fmla v31.8h, v6.8h, v0.h[7]\n"
    "18:"  // Output channel oddments: Done
    "fmin v16.8h, v16.8h, v7.8h\n"
    "fmin v17.8h, v17.8h, v7.8h\n"
    "fmin v18.8h, v18.8h, v7.8h\n"
    "fmin v19.8h, v19.8h, v7.8h\n"
    "fmin v20.8h, v20.8h, v7.8h\n"
    "fmin v21.8h, v21.8h, v7.8h\n"
    "fmin v22.8h, v22.8h, v7.8h\n"
    "fmin v23.8h, v23.8h, v7.8h\n"
    "fmin v24.8h, v24.8h, v7.8h\n"
    "fmin v25.8h, v25.8h, v7.8h\n"
    "fmin v26.8h, v26.8h, v7.8h\n"
    "fmin v27.8h, v27.8h, v7.8h\n"
    "fmin v28.8h, v28.8h, v7.8h\n"
    "fmin v29.8h, v29.8h, v7.8h\n"
    "fmin v30.8h, v30.8h, v7.8h\n"
    "fmin v31.8h, v31.8h, v7.8h\n"
    "fmax v16.8h, v16.8h, v8.8h\n"
    "fmax v17.8h, v17.8h, v8.8h\n"
    "fmax v18.8h, v18.8h, v8.8h\n"
    "fmax v19.8h, v19.8h, v8.8h\n"
    "fmax v20.8h, v20.8h, v8.8h\n"
    "fmax v21.8h, v21.8h, v8.8h\n"
    "fmax v22.8h, v22.8h, v8.8h\n"
    "fmax v23.8h, v23.8h, v8.8h\n"
    "fmax v24.8h, v24.8h, v8.8h\n"
    "fmax v25.8h, v25.8h, v8.8h\n"
    "fmax v26.8h, v26.8h, v8.8h\n"
    "fmax v27.8h, v27.8h, v8.8h\n"
    "fmax v28.8h, v28.8h, v8.8h\n"
    "fmax v29.8h, v29.8h, v8.8h\n"
    "fmax v30.8h, v30.8h, v8.8h\n"
    "fmax v31.8h, v31.8h, v8.8h\n"
    "tbz %x[n_output_channels], #2, 20f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.d }[0], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.d }[0], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.d }[0], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.d }[0], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.d }[0], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.d }[0], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.d }[0], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.d }[0], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.d }[0], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.d }[0], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "add x10, x10, #0x4\n"
    "st1 { v26.d }[0], [x25]\n"
    "st1 { v27.d }[0], [x24]\n"
    "st1 { v28.d }[0], [x23]\n"
    "st1 { v29.d }[0], [x22]\n"
    "st1 { v30.d }[0], [x21]\n"
    "st1 { v31.d }[0], [x20]\n"
    "tbz %x[n_output_channels], #1, 19f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.s }[2], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.s }[2], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.s }[2], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.s }[2], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.s }[2], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.s }[2], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.s }[2], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.s }[2], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.s }[2], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.s }[2], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "add x10, x10, #0x2\n"
    "st1 { v26.s }[2], [x25]\n"
    "st1 { v27.s }[2], [x24]\n"
    "st1 { v28.s }[2], [x23]\n"
    "st1 { v29.s }[2], [x22]\n"
    "st1 { v30.s }[2], [x21]\n"
    "st1 { v31.s }[2], [x20]\n"
    "tbz %x[n_output_channels], #0, 22f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.h }[6], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.h }[6], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.h }[6], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.h }[6], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.h }[6], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.h }[6], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.h }[6], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.h }[6], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.h }[6], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.h }[6], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v26.h }[6], [x25]\n"
    "st1 { v27.h }[6], [x24]\n"
    "st1 { v28.h }[6], [x23]\n"
    "st1 { v29.h }[6], [x22]\n"
    "st1 { v30.h }[6], [x21]\n"
    "st1 { v31.h }[6], [x20]\n"
    "b 22f\n"
    "19:"  // Output channel oddments: Done: Store: Bit 2: Bit 1: Unset
    "tbz %x[n_output_channels], #0, 22f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.h }[4], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.h }[4], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.h }[4], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.h }[4], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.h }[4], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.h }[4], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.h }[4], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.h }[4], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.h }[4], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.h }[4], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v26.h }[4], [x25]\n"
    "st1 { v27.h }[4], [x24]\n"
    "st1 { v28.h }[4], [x23]\n"
    "st1 { v29.h }[4], [x22]\n"
    "st1 { v30.h }[4], [x21]\n"
    "st1 { v31.h }[4], [x20]\n"
    "b 22f\n"
    "20:"  // Output channel oddments: Done: Store: Bit 2: Unset
    "tbz %x[n_output_channels], #1, 21f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.s }[0], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.s }[0], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.s }[0], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.s }[0], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.s }[0], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.s }[0], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.s }[0], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.s }[0], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.s }[0], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.s }[0], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "add x10, x10, #0x2\n"
    "st1 { v26.s }[0], [x25]\n"
    "st1 { v27.s }[0], [x24]\n"
    "st1 { v28.s }[0], [x23]\n"
    "st1 { v29.s }[0], [x22]\n"
    "st1 { v30.s }[0], [x21]\n"
    "st1 { v31.s }[0], [x20]\n"
    "tbz %x[n_output_channels], #0, 22f\n"
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.h }[2], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.h }[2], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.h }[2], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.h }[2], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.h }[2], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.h }[2], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.h }[2], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.h }[2], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.h }[2], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.h }[2], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v26.h }[2], [x25]\n"
    "st1 { v27.h }[2], [x24]\n"
    "st1 { v28.h }[2], [x23]\n"
    "st1 { v29.h }[2], [x22]\n"
    "st1 { v30.h }[2], [x21]\n"
    "st1 { v31.h }[2], [x20]\n"
    "b 22f\n"
    "21:"  // Output channel oddments: Done: Store: Bit 2: Unset: Bit 1: Unset
    "ldr x27, [%x[outptrs], #0x0]\n"
    "ldr x26, [%x[outptrs], #0x8]\n"
    "ldr x25, [%x[outptrs], #0x10]\n"
    "ldr x24, [%x[outptrs], #0x18]\n"
    "ldr x23, [%x[outptrs], #0x20]\n"
    "ldr x22, [%x[outptrs], #0x28]\n"
    "ldr x21, [%x[outptrs], #0x30]\n"
    "ldr x20, [%x[outptrs], #0x38]\n"
    "add x27, x27, x10, LSL #1\n"
    "add x26, x26, x10, LSL #1\n"
    "add x25, x25, x10, LSL #1\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v16.h }[0], [x27]\n"
    "ldr x27, [%x[outptrs], #0x40]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v17.h }[0], [x26]\n"
    "ldr x26, [%x[outptrs], #0x48]\n"
    "add x21, x21, x10, LSL #1\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v18.h }[0], [x25]\n"
    "ldr x25, [%x[outptrs], #0x50]\n"
    "st1 { v19.h }[0], [x24]\n"
    "ldr x24, [%x[outptrs], #0x58]\n"
    "add x27, x27, x10, LSL #1\n"
    "st1 { v20.h }[0], [x23]\n"
    "ldr x23, [%x[outptrs], #0x60]\n"
    "add x26, x26, x10, LSL #1\n"
    "st1 { v21.h }[0], [x22]\n"
    "ldr x22, [%x[outptrs], #0x68]\n"
    "add x25, x25, x10, LSL #1\n"
    "st1 { v22.h }[0], [x21]\n"
    "ldr x21, [%x[outptrs], #0x70]\n"
    "add x24, x24, x10, LSL #1\n"
    "st1 { v23.h }[0], [x20]\n"
    "ldr x20, [%x[outptrs], #0x78]\n"
    "add x23, x23, x10, LSL #1\n"
    "add x22, x22, x10, LSL #1\n"
    "st1 { v24.h }[0], [x27]\n"
    "add x21, x21, x10, LSL #1\n"
    "st1 { v25.h }[0], [x26]\n"
    "add x20, x20, x10, LSL #1\n"
    "st1 { v26.h }[0], [x25]\n"
    "st1 { v27.h }[0], [x24]\n"
    "st1 { v28.h }[0], [x23]\n"
    "st1 { v29.h }[0], [x22]\n"
    "st1 { v30.h }[0], [x21]\n"
    "st1 { v31.h }[0], [x20]\n"
    "22:"  // Output channel oddments: Done: Store: Bit 2: End
    "23:"  // Done
    : [weights] "+&r" (weights)
    : [bias] "r" (bias), [inptrs] "r" (inptrs), [kernel_points] "r" ((uint64_t) kernel_points), [minmax_vals] "r" (minmax_vals), [n_output_channels] "r" ((uint64_t) n_output_channels), [outptrs] "r" (outptrs)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
