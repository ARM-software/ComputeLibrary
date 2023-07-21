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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef __aarch64__
namespace
{
void a64_add_bn_clamp_direct_fp32_2x16(
    float *out, size_t out_stride,
    float *out_direct, size_t out_direct_stride,
    const float *in0, size_t in0_stride,
    const float *in1, size_t in1_stride,
    const float *bn_mul,
    const float *bn_add,
    const float  minval,
    const float  maxval,
    size_t width, size_t height)
{
    struct KernelArgs
    {
        float minval;
        float maxval;
    } ka;
    ka.minval = minval;
    ka.maxval = maxval;

    __asm__ __volatile__(
        "ldr w21, [%x[args_ptr], %[offsetof_minval]]\n"
        "ldr w20, [%x[args_ptr], %[offsetof_maxval]]\n"
        "cmp %x[width], #0x10\n"
        "dup v13.4s, w21\n"
        "dup v12.4s, w20\n"
        "blt 7f\n"
        "1:" // Column loop
        "ldr q24, [%x[bn_mul], #0x0]\n"
        "ldr q25, [%x[bn_mul], #0x10]\n"
        "mov x12, %x[in0]\n"
        "mov x11, %x[in1]\n"
        "ldr q26, [%x[bn_mul], #0x20]\n"
        "ldr q27, [%x[bn_mul], #0x30]\n"
        "mov x10, %x[out]\n"
        "mov x9, %x[out_direct]\n"
        "ldr q28, [%x[bn_add], #0x0]\n"
        "ldr q29, [%x[bn_add], #0x10]\n"
        "mov x20, %x[height]\n"
        "mov x28, x12\n"
        "ldr q30, [%x[bn_add], #0x20]\n"
        "ldr q31, [%x[bn_add], #0x30]\n"
        "mov x27, x11\n"
        "mov x26, x10\n"
        "ldr q11, [x28, #0x0]\n"
        "ldr q10, [x27, #0x0]\n"
        "mov x25, x9\n"
        "add x24, x28, %x[in0_stride]\n"
        "ldr q9, [x28, #0x10]\n"
        "ldr q8, [x27, #0x10]\n"
        "add x23, x27, %x[in1_stride]\n"
        "add x22, x26, %x[out_stride]\n"
        "ldr q7, [x28, #0x20]\n"
        "ldr q6, [x27, #0x20]\n"
        "add x21, x25, %x[out_direct_stride]\n"
        "cmp x20, #0x2\n"
        "ldr q5, [x28, #0x30]\n"
        "ldr q4, [x27, #0x30]\n"
        "add x12, x24, %x[in0_stride]\n"
        "add x11, x23, %x[in1_stride]\n"
        "add x10, x22, %x[out_stride]\n"
        "add x9, x21, %x[out_direct_stride]\n"
        "csel x24, x24, x28, GE\n"
        "csel x23, x23, x27, GE\n"
        "csel x22, x22, x26, GE\n"
        "csel x21, x21, x25, GE\n"
        "subs x20, x20, #0x2\n"
        "add %x[bn_mul], %x[bn_mul], #0x40\n"
        "add %x[bn_add], %x[bn_add], #0x40\n"
        "add x28, x28, #0x40\n"
        "add x27, x27, #0x40\n"
        "ble 4f\n"
        "2:" // Row loop
        "ldr q3, [x24, #0x0]\n"
        "ldr q22, [x23, #0x0]\n"
        "fadd v2.4s, v11.4s, v10.4s\n"
        "fadd v1.4s, v9.4s, v8.4s\n"
        "ldr q21, [x24, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "fadd v0.4s, v7.4s, v6.4s\n"
        "fadd v23.4s, v5.4s, v4.4s\n"
        "ldr q19, [x24, #0x20]\n"
        "ldr q18, [x23, #0x20]\n"
        "fadd v22.4s, v3.4s, v22.4s\n"
        "fadd v21.4s, v21.4s, v20.4s\n"
        "ldr q17, [x24, #0x30]\n"
        "ldr q16, [x23, #0x30]\n"
        "fadd v20.4s, v19.4s, v18.4s\n"
        "fadd v19.4s, v17.4s, v16.4s\n"
        "add x24, x24, #0x40\n"
        "add x23, x23, #0x40\n"
        "cbz %x[out_direct], 3f\n"
        "str q2, [x25, #0x0]\n"
        "str q1, [x25, #0x10]\n"
        "str q0, [x25, #0x20]\n"
        "str q23, [x25, #0x30]\n"
        "add x25, x25, #0x40\n"
        "str q22, [x21, #0x0]\n"
        "str q21, [x21, #0x10]\n"
        "str q20, [x21, #0x20]\n"
        "str q19, [x21, #0x30]\n"
        "add x21, x21, #0x40\n"
        "3:" // Main loop: No direct output
        "mov v16.16b, v2.16b\n"
        "mov v2.16b, v28.16b\n"
        "fmla v2.4s, v16.4s, v24.4s\n"
        "mov x28, x12\n"
        "ldr q11, [x28, #0x0]\n"
        "ldr q9, [x28, #0x10]\n"
        "mov v18.16b, v1.16b\n"
        "mov v1.16b, v29.16b\n"
        "ldr q7, [x28, #0x20]\n"
        "ldr q5, [x28, #0x30]\n"
        "mov v17.16b, v0.16b\n"
        "mov v0.16b, v30.16b\n"
        "mov v16.16b, v23.16b\n"
        "mov v23.16b, v31.16b\n"
        "fmla v1.4s, v18.4s, v25.4s\n"
        "mov x27, x11\n"
        "ldr q10, [x27, #0x0]\n"
        "ldr q8, [x27, #0x10]\n"
        "fmla v0.4s, v17.4s, v26.4s\n"
        "fmla v23.4s, v16.4s, v27.4s\n"
        "ldr q6, [x27, #0x20]\n"
        "ldr q4, [x27, #0x30]\n"
        "mov v17.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v22.4s, v17.4s, v24.4s\n"
        "mov x25, x9\n"
        "mov v17.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v21.4s, v16.4s, v25.4s\n"
        "add x24, x28, %x[in0_stride]\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v20.4s, v17.4s, v26.4s\n"
        "add x23, x27, %x[in1_stride]\n"
        "fmla v19.4s, v16.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v12.4s\n"
        "add x21, x25, %x[out_direct_stride]\n"
        "cmp x20, #0x2\n"
        "fmin v1.4s, v1.4s, v12.4s\n"
        "fmin v0.4s, v0.4s, v12.4s\n"
        "add x12, x24, %x[in0_stride]\n"
        "add x11, x23, %x[in1_stride]\n"
        "fmin v23.4s, v23.4s, v12.4s\n"
        "fmax v2.4s, v2.4s, v13.4s\n"
        "str q2, [x26, #0x0]\n"
        "add x9, x21, %x[out_direct_stride]\n"
        "fmax v1.4s, v1.4s, v13.4s\n"
        "fmax v0.4s, v0.4s, v13.4s\n"
        "str q1, [x26, #0x10]\n"
        "csel x24, x24, x28, GE\n"
        "fmax v23.4s, v23.4s, v13.4s\n"
        "fmin v22.4s, v22.4s, v12.4s\n"
        "str q0, [x26, #0x20]\n"
        "csel x23, x23, x27, GE\n"
        "fmin v21.4s, v21.4s, v12.4s\n"
        "fmin v20.4s, v20.4s, v12.4s\n"
        "str q23, [x26, #0x30]\n"
        "mov x26, x10\n"
        "fmin v19.4s, v19.4s, v12.4s\n"
        "fmax v22.4s, v22.4s, v13.4s\n"
        "str q22, [x22, #0x0]\n"
        "csel x21, x21, x25, GE\n"
        "fmax v21.4s, v21.4s, v13.4s\n"
        "fmax v20.4s, v20.4s, v13.4s\n"
        "str q21, [x22, #0x10]\n"
        "add x28, x28, #0x40\n"
        "fmax v19.4s, v19.4s, v13.4s\n"
        "str q20, [x22, #0x20]\n"
        "add x27, x27, #0x40\n"
        "str q19, [x22, #0x30]\n"
        "add x22, x26, %x[out_stride]\n"
        "add x10, x22, %x[out_stride]\n"
        "csel x22, x22, x26, GE\n"
        "subs x20, x20, #0x2\n"
        "bgt 2b\n"
        "4:" // Row loop skip
        "ldr q3, [x24, #0x0]\n"
        "ldr q22, [x23, #0x0]\n"
        "fadd v2.4s, v11.4s, v10.4s\n"
        "fadd v1.4s, v9.4s, v8.4s\n"
        "ldr q21, [x24, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "fadd v0.4s, v7.4s, v6.4s\n"
        "fadd v23.4s, v5.4s, v4.4s\n"
        "ldr q19, [x24, #0x20]\n"
        "ldr q18, [x23, #0x20]\n"
        "fadd v22.4s, v3.4s, v22.4s\n"
        "fadd v21.4s, v21.4s, v20.4s\n"
        "ldr q17, [x24, #0x30]\n"
        "ldr q16, [x23, #0x30]\n"
        "fadd v20.4s, v19.4s, v18.4s\n"
        "fadd v19.4s, v17.4s, v16.4s\n"
        "add x24, x24, #0x40\n"
        "add x23, x23, #0x40\n"
        "cbz %x[out_direct], 5f\n"
        "str q2, [x25, #0x0]\n"
        "str q1, [x25, #0x10]\n"
        "str q0, [x25, #0x20]\n"
        "str q23, [x25, #0x30]\n"
        "add x25, x25, #0x40\n"
        "str q22, [x21, #0x0]\n"
        "str q21, [x21, #0x10]\n"
        "str q20, [x21, #0x20]\n"
        "str q19, [x21, #0x30]\n"
        "add x21, x21, #0x40\n"
        "5:" // Tail loop: No direct output
        "mov v16.16b, v2.16b\n"
        "mov v2.16b, v28.16b\n"
        "fmla v2.4s, v16.4s, v24.4s\n"
        "add %x[in0], %x[in0], #0x40\n"
        "mov v16.16b, v1.16b\n"
        "mov v1.16b, v29.16b\n"
        "fmla v1.4s, v16.4s, v25.4s\n"
        "add %x[in1], %x[in1], #0x40\n"
        "mov v16.16b, v0.16b\n"
        "mov v0.16b, v30.16b\n"
        "fmla v0.4s, v16.4s, v26.4s\n"
        "add %x[out], %x[out], #0x40\n"
        "mov v16.16b, v23.16b\n"
        "mov v23.16b, v31.16b\n"
        "fmla v23.4s, v16.4s, v27.4s\n"
        "mov v16.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "fmla v22.4s, v16.4s, v24.4s\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v21.4s, v16.4s, v25.4s\n"
        "mov v16.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v20.4s, v16.4s, v26.4s\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v19.4s, v16.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v12.4s\n"
        "fmin v1.4s, v1.4s, v12.4s\n"
        "fmin v0.4s, v0.4s, v12.4s\n"
        "fmin v23.4s, v23.4s, v12.4s\n"
        "fmin v22.4s, v22.4s, v12.4s\n"
        "fmin v21.4s, v21.4s, v12.4s\n"
        "fmin v20.4s, v20.4s, v12.4s\n"
        "fmin v19.4s, v19.4s, v12.4s\n"
        "fmax v2.4s, v2.4s, v13.4s\n"
        "fmax v1.4s, v1.4s, v13.4s\n"
        "str q2, [x26, #0x0]\n"
        "fmax v0.4s, v0.4s, v13.4s\n"
        "fmax v23.4s, v23.4s, v13.4s\n"
        "str q1, [x26, #0x10]\n"
        "fmax v22.4s, v22.4s, v13.4s\n"
        "fmax v21.4s, v21.4s, v13.4s\n"
        "str q0, [x26, #0x20]\n"
        "fmax v20.4s, v20.4s, v13.4s\n"
        "fmax v19.4s, v19.4s, v13.4s\n"
        "str q23, [x26, #0x30]\n"
        "add x26, x26, #0x40\n"
        "str q22, [x22, #0x0]\n"
        "str q21, [x22, #0x10]\n"
        "str q20, [x22, #0x20]\n"
        "str q19, [x22, #0x30]\n"
        "add x22, x22, #0x40\n"
        "cbz %x[out_direct], 6f\n"
        "add %x[out_direct], %x[out_direct], #0x40\n"
        "6:" // No direct pointer update
        "sub %x[width], %x[width], #0x10\n"
        "cmp %x[width], #0x10\n"
        "bge 1b\n"
        "cbz %x[width], 34f\n"
        "7:" // main loop skip
        "ldr q24, [%x[bn_mul], #0x0]\n"
        "ldr q25, [%x[bn_mul], #0x10]\n"
        "mov x20, %x[height]\n"
        "mov x12, %x[in0]\n"
        "ldr q26, [%x[bn_mul], #0x20]\n"
        "ldr q27, [%x[bn_mul], #0x30]\n"
        "mov x11, %x[in1]\n"
        "mov x10, %x[out]\n"
        "ldr q28, [%x[bn_add], #0x0]\n"
        "ldr q29, [%x[bn_add], #0x10]\n"
        "mov x9, %x[out_direct]\n"
        "add %x[bn_mul], %x[bn_mul], #0x40\n"
        "ldr q30, [%x[bn_add], #0x20]\n"
        "ldr q31, [%x[bn_add], #0x30]\n"
        "add %x[bn_add], %x[bn_add], #0x40\n"
        "8:" // tail loop: Row loop
        "mov x28, x12\n"
        "mov x27, x11\n"
        "mov x26, x10\n"
        "mov x25, x9\n"
        "add x24, x28, %x[in0_stride]\n"
        "add x23, x27, %x[in1_stride]\n"
        "add x22, x26, %x[out_stride]\n"
        "add x21, x25, %x[out_direct_stride]\n"
        "cmp x20, #0x2\n"
        "add x12, x24, %x[in0_stride]\n"
        "add x11, x23, %x[in1_stride]\n"
        "add x10, x22, %x[out_stride]\n"
        "add x9, x21, %x[out_direct_stride]\n"
        "csel x24, x24, x28, GE\n"
        "csel x23, x23, x27, GE\n"
        "csel x22, x22, x26, GE\n"
        "csel x21, x21, x25, GE\n"
        "tbz %x[width], #3, 12f\n"
        "ldr q11, [x28, #0x0]\n"
        "ldr q10, [x27, #0x0]\n"
        "ldr q9, [x28, #0x10]\n"
        "ldr q8, [x27, #0x10]\n"
        "add x28, x28, #0x20\n"
        "add x27, x27, #0x20\n"
        "ldr q3, [x24, #0x0]\n"
        "ldr q22, [x23, #0x0]\n"
        "ldr q21, [x24, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "add x24, x24, #0x20\n"
        "add x23, x23, #0x20\n"
        "tbz %x[width], #2, 10f\n"
        "ldr q7, [x28, #0x0]\n"
        "ldr q6, [x27, #0x0]\n"
        "add x28, x28, #0x10\n"
        "add x27, x27, #0x10\n"
        "ldr q19, [x24, #0x0]\n"
        "ldr q18, [x23, #0x0]\n"
        "add x24, x24, #0x10\n"
        "add x23, x23, #0x10\n"
        "tbz %x[width], #1, 9f\n"
        "ldr d5, [x28], #0x8\n"
        "ldr d4, [x27], #0x8\n"
        "ldr d17, [x24], #0x8\n"
        "ldr d16, [x23], #0x8\n"
        "tbz %x[width], #0, 16f\n"
        "ld1 { v5.s }[2], [x28], #0x4\n"
        "ld1 { v4.s }[2], [x27], #0x4\n"
        "ld1 { v17.s }[2], [x24], #0x4\n"
        "ld1 { v16.s }[2], [x23], #0x4\n"
        "b 16f\n"
        "9:" // tail loop: unique 1: partial_0_12
        "tbz %x[width], #0, 16f\n"
        "ldr s5, [x28], #0x4\n"
        "ldr s4, [x27], #0x4\n"
        "ldr s17, [x24], #0x4\n"
        "ldr s16, [x23], #0x4\n"
        "b 16f\n"
        "10:" // tail loop: unique 1: partial_1_8
        "tbz %x[width], #1, 11f\n"
        "ldr d7, [x28], #0x8\n"
        "ldr d6, [x27], #0x8\n"
        "ldr d19, [x24], #0x8\n"
        "ldr d18, [x23], #0x8\n"
        "tbz %x[width], #0, 16f\n"
        "ld1 { v7.s }[2], [x28], #0x4\n"
        "ld1 { v6.s }[2], [x27], #0x4\n"
        "ld1 { v19.s }[2], [x24], #0x4\n"
        "ld1 { v18.s }[2], [x23], #0x4\n"
        "b 16f\n"
        "11:" // tail loop: unique 1: partial_0_8
        "tbz %x[width], #0, 16f\n"
        "ldr s7, [x28], #0x4\n"
        "ldr s6, [x27], #0x4\n"
        "ldr s19, [x24], #0x4\n"
        "ldr s18, [x23], #0x4\n"
        "b 16f\n"
        "12:" // tail loop: unique 1: partial_2_0
        "tbz %x[width], #2, 14f\n"
        "ldr q11, [x28, #0x0]\n"
        "ldr q10, [x27, #0x0]\n"
        "add x28, x28, #0x10\n"
        "add x27, x27, #0x10\n"
        "ldr q3, [x24, #0x0]\n"
        "ldr q22, [x23, #0x0]\n"
        "add x24, x24, #0x10\n"
        "add x23, x23, #0x10\n"
        "tbz %x[width], #1, 13f\n"
        "ldr d9, [x28], #0x8\n"
        "ldr d8, [x27], #0x8\n"
        "ldr d21, [x24], #0x8\n"
        "ldr d20, [x23], #0x8\n"
        "tbz %x[width], #0, 16f\n"
        "ld1 { v9.s }[2], [x28], #0x4\n"
        "ld1 { v8.s }[2], [x27], #0x4\n"
        "ld1 { v21.s }[2], [x24], #0x4\n"
        "ld1 { v20.s }[2], [x23], #0x4\n"
        "b 16f\n"
        "13:" // tail loop: unique 1: partial_0_4
        "tbz %x[width], #0, 16f\n"
        "ldr s9, [x28], #0x4\n"
        "ldr s8, [x27], #0x4\n"
        "ldr s21, [x24], #0x4\n"
        "ldr s20, [x23], #0x4\n"
        "b 16f\n"
        "14:" // tail loop: unique 1: partial_1_0
        "tbz %x[width], #1, 15f\n"
        "ldr d11, [x28], #0x8\n"
        "ldr d10, [x27], #0x8\n"
        "ldr d3, [x24], #0x8\n"
        "ldr d22, [x23], #0x8\n"
        "tbz %x[width], #0, 16f\n"
        "ld1 { v11.s }[2], [x28], #0x4\n"
        "ld1 { v10.s }[2], [x27], #0x4\n"
        "ld1 { v3.s }[2], [x24], #0x4\n"
        "ld1 { v22.s }[2], [x23], #0x4\n"
        "b 16f\n"
        "15:" // tail loop: unique 1: partial_0_0
        "ldr s11, [x28], #0x4\n"
        "ldr s10, [x27], #0x4\n"
        "ldr s3, [x24], #0x4\n"
        "ldr s22, [x23], #0x4\n"
        "16:" // tail loop: unique 1: Done
        "fadd v2.4s, v11.4s, v10.4s\n"
        "fadd v1.4s, v9.4s, v8.4s\n"
        "fadd v0.4s, v7.4s, v6.4s\n"
        "fadd v23.4s, v5.4s, v4.4s\n"
        "fadd v22.4s, v3.4s, v22.4s\n"
        "fadd v21.4s, v21.4s, v20.4s\n"
        "fadd v20.4s, v19.4s, v18.4s\n"
        "fadd v19.4s, v17.4s, v16.4s\n"
        "cbz %x[out_direct], 25f\n"
        "tbz %x[width], #3, 20f\n"
        "str q2, [x25, #0x0]\n"
        "str q1, [x25, #0x10]\n"
        "add x25, x25, #0x20\n"
        "str q22, [x21, #0x0]\n"
        "str q21, [x21, #0x10]\n"
        "add x21, x21, #0x20\n"
        "tbz %x[width], #2, 18f\n"
        "str q0, [x25, #0x0]\n"
        "add x25, x25, #0x10\n"
        "str q20, [x21, #0x0]\n"
        "add x21, x21, #0x10\n"
        "tbz %x[width], #1, 17f\n"
        "str d23, [x25], #0x8\n"
        "str d19, [x21], #0x8\n"
        "tbz %x[width], #0, 24f\n"
        "st1 { v23.s }[2], [x25], #0x4\n"
        "st1 { v19.s }[2], [x21], #0x4\n"
        "b 24f\n"
        "17:" // tail loop: Main loop: unique 2: partial_0_12
        "tbz %x[width], #0, 24f\n"
        "str s23, [x25], #0x4\n"
        "str s19, [x21], #0x4\n"
        "b 24f\n"
        "18:" // tail loop: Main loop: unique 2: partial_1_8
        "tbz %x[width], #1, 19f\n"
        "str d0, [x25], #0x8\n"
        "str d20, [x21], #0x8\n"
        "tbz %x[width], #0, 24f\n"
        "st1 { v0.s }[2], [x25], #0x4\n"
        "st1 { v20.s }[2], [x21], #0x4\n"
        "b 24f\n"
        "19:" // tail loop: Main loop: unique 2: partial_0_8
        "tbz %x[width], #0, 24f\n"
        "str s0, [x25], #0x4\n"
        "str s20, [x21], #0x4\n"
        "b 24f\n"
        "20:" // tail loop: Main loop: unique 2: partial_2_0
        "tbz %x[width], #2, 22f\n"
        "str q2, [x25, #0x0]\n"
        "add x25, x25, #0x10\n"
        "str q22, [x21, #0x0]\n"
        "add x21, x21, #0x10\n"
        "tbz %x[width], #1, 21f\n"
        "str d1, [x25], #0x8\n"
        "str d21, [x21], #0x8\n"
        "tbz %x[width], #0, 24f\n"
        "st1 { v1.s }[2], [x25], #0x4\n"
        "st1 { v21.s }[2], [x21], #0x4\n"
        "b 24f\n"
        "21:" // tail loop: Main loop: unique 2: partial_0_4
        "tbz %x[width], #0, 24f\n"
        "str s1, [x25], #0x4\n"
        "str s21, [x21], #0x4\n"
        "b 24f\n"
        "22:" // tail loop: Main loop: unique 2: partial_1_0
        "tbz %x[width], #1, 23f\n"
        "str d2, [x25], #0x8\n"
        "str d22, [x21], #0x8\n"
        "tbz %x[width], #0, 24f\n"
        "st1 { v2.s }[2], [x25], #0x4\n"
        "st1 { v22.s }[2], [x21], #0x4\n"
        "b 24f\n"
        "23:" // tail loop: Main loop: unique 2: partial_0_0
        "str s2, [x25], #0x4\n"
        "str s22, [x21], #0x4\n"
        "24:" // tail loop: Main loop: unique 2: Done
        "25:" // tail loop: Main loop: No direct output
        "mov v16.16b, v2.16b\n"
        "mov v2.16b, v28.16b\n"
        "fmla v2.4s, v16.4s, v24.4s\n"
        "mov v16.16b, v1.16b\n"
        "mov v1.16b, v29.16b\n"
        "fmla v1.4s, v16.4s, v25.4s\n"
        "mov v16.16b, v0.16b\n"
        "mov v0.16b, v30.16b\n"
        "fmla v0.4s, v16.4s, v26.4s\n"
        "mov v16.16b, v23.16b\n"
        "mov v23.16b, v31.16b\n"
        "fmla v23.4s, v16.4s, v27.4s\n"
        "mov v16.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "fmla v22.4s, v16.4s, v24.4s\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v21.4s, v16.4s, v25.4s\n"
        "mov v16.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v20.4s, v16.4s, v26.4s\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v19.4s, v16.4s, v27.4s\n"
        "fmin v2.4s, v2.4s, v12.4s\n"
        "fmin v1.4s, v1.4s, v12.4s\n"
        "fmin v0.4s, v0.4s, v12.4s\n"
        "fmin v23.4s, v23.4s, v12.4s\n"
        "fmin v22.4s, v22.4s, v12.4s\n"
        "fmin v21.4s, v21.4s, v12.4s\n"
        "fmin v20.4s, v20.4s, v12.4s\n"
        "fmin v19.4s, v19.4s, v12.4s\n"
        "fmax v2.4s, v2.4s, v13.4s\n"
        "fmax v1.4s, v1.4s, v13.4s\n"
        "fmax v0.4s, v0.4s, v13.4s\n"
        "fmax v23.4s, v23.4s, v13.4s\n"
        "fmax v22.4s, v22.4s, v13.4s\n"
        "fmax v21.4s, v21.4s, v13.4s\n"
        "fmax v20.4s, v20.4s, v13.4s\n"
        "fmax v19.4s, v19.4s, v13.4s\n"
        "tbz %x[width], #3, 29f\n"
        "str q2, [x26, #0x0]\n"
        "str q1, [x26, #0x10]\n"
        "add x26, x26, #0x20\n"
        "str q22, [x22, #0x0]\n"
        "str q21, [x22, #0x10]\n"
        "add x22, x22, #0x20\n"
        "tbz %x[width], #2, 27f\n"
        "str q0, [x26, #0x0]\n"
        "add x26, x26, #0x10\n"
        "str q20, [x22, #0x0]\n"
        "add x22, x22, #0x10\n"
        "tbz %x[width], #1, 26f\n"
        "str d23, [x26], #0x8\n"
        "str d19, [x22], #0x8\n"
        "tbz %x[width], #0, 33f\n"
        "st1 { v23.s }[2], [x26], #0x4\n"
        "st1 { v19.s }[2], [x22], #0x4\n"
        "b 33f\n"
        "26:" // tail loop: unique 3: partial_0_12
        "tbz %x[width], #0, 33f\n"
        "str s23, [x26], #0x4\n"
        "str s19, [x22], #0x4\n"
        "b 33f\n"
        "27:" // tail loop: unique 3: partial_1_8
        "tbz %x[width], #1, 28f\n"
        "str d0, [x26], #0x8\n"
        "str d20, [x22], #0x8\n"
        "tbz %x[width], #0, 33f\n"
        "st1 { v0.s }[2], [x26], #0x4\n"
        "st1 { v20.s }[2], [x22], #0x4\n"
        "b 33f\n"
        "28:" // tail loop: unique 3: partial_0_8
        "tbz %x[width], #0, 33f\n"
        "str s0, [x26], #0x4\n"
        "str s20, [x22], #0x4\n"
        "b 33f\n"
        "29:" // tail loop: unique 3: partial_2_0
        "tbz %x[width], #2, 31f\n"
        "str q2, [x26, #0x0]\n"
        "add x26, x26, #0x10\n"
        "str q22, [x22, #0x0]\n"
        "add x22, x22, #0x10\n"
        "tbz %x[width], #1, 30f\n"
        "str d1, [x26], #0x8\n"
        "str d21, [x22], #0x8\n"
        "tbz %x[width], #0, 33f\n"
        "st1 { v1.s }[2], [x26], #0x4\n"
        "st1 { v21.s }[2], [x22], #0x4\n"
        "b 33f\n"
        "30:" // tail loop: unique 3: partial_0_4
        "tbz %x[width], #0, 33f\n"
        "str s1, [x26], #0x4\n"
        "str s21, [x22], #0x4\n"
        "b 33f\n"
        "31:" // tail loop: unique 3: partial_1_0
        "tbz %x[width], #1, 32f\n"
        "str d2, [x26], #0x8\n"
        "str d22, [x22], #0x8\n"
        "tbz %x[width], #0, 33f\n"
        "st1 { v2.s }[2], [x26], #0x4\n"
        "st1 { v22.s }[2], [x22], #0x4\n"
        "b 33f\n"
        "32:" // tail loop: unique 3: partial_0_0
        "str s2, [x26], #0x4\n"
        "str s22, [x22], #0x4\n"
        "33:" // tail loop: unique 3: Done
        "subs x20, x20, #0x2\n"
        "bgt 8b\n"
        "34:" // odd columns skip
        : [bn_add] "+&r"(bn_add), [bn_mul] "+&r"(bn_mul), [in0] "+&r"(in0), [in1] "+&r"(in1), [out] "+&r"(out), [out_direct] "+&r"(out_direct), [width] "+&r"(width)
        : [args_ptr] "r"(&ka), [height] "r"(height), [in0_stride] "r"(in0_stride), [in1_stride] "r"(in1_stride), [offsetof_maxval] "I"(offsetof(KernelArgs, maxval)), [offsetof_minval] "I"(offsetof(KernelArgs, minval)), [out_direct_stride] "r"(out_direct_stride), [out_stride] "r"(out_stride)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}
}

namespace arm_compute
{
namespace cpu
{
void add_mul_add_fp32_neon(const ITensor *input1, const ITensor *input2, const ITensor *bn_mul, const ITensor *bn_add,
                           ITensor *add_output, ITensor *final_output, ConvertPolicy policy, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    const size_t out_stride        = final_output->info()->strides_in_bytes()[1];
    const size_t out_direct_stride = (add_output != nullptr) ? add_output->info()->strides_in_bytes()[1] : 0;
    const size_t in0_stride        = input1->info()->strides_in_bytes()[1];
    const size_t in1_stride        = input2->info()->strides_in_bytes()[1];

    float minval = std::numeric_limits<float>::lowest();
    float maxval = std::numeric_limits<float>::max();

    if(act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU)
    {
        minval = 0.f;
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
    {
        minval = 0.f;
        maxval = act_info.a();
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
    {
        minval = act_info.b();
        maxval = act_info.a();
    }

    // Clear X & Y dimensions on execution window as we handle manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));

    Iterator in1_it(input1, window);
    Iterator in2_it(input2, window);
    Iterator out_it(final_output, window);

    const size_t width  = window.num_iterations(0);
    const size_t height = window.num_iterations(1);

    if(add_output != nullptr)
    {
        Iterator add_out_it(add_output, window);
        execute_window_loop(
            win, [&](const Coordinates &)
        {
            a64_add_bn_clamp_direct_fp32_2x16(
                reinterpret_cast<float *>(out_it.ptr()), out_stride,
                reinterpret_cast<float *>(add_out_it.ptr()), out_direct_stride,
                reinterpret_cast<float *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<float *>(in2_it.ptr()), in1_stride,
                reinterpret_cast<float *>(bn_mul->buffer()),
                reinterpret_cast<float *>(bn_add->buffer()),
                minval,
                maxval,
                width, height);
        },
        in1_it, in2_it, add_out_it, out_it);
    }
    else
    {
        execute_window_loop(
            win, [&](const Coordinates &)
        {
            a64_add_bn_clamp_direct_fp32_2x16(
                reinterpret_cast<float *>(out_it.ptr()), out_stride,
                nullptr, out_direct_stride,
                reinterpret_cast<float *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<float *>(in2_it.ptr()), in1_stride,
                reinterpret_cast<float *>(bn_mul->buffer()),
                reinterpret_cast<float *>(bn_add->buffer()),
                minval,
                maxval,
                width, height);
        },
        in1_it, in2_it, out_it);
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // __aarch64__
