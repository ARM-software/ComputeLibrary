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

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__aarch64__) && defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
namespace
{
void a64_add_bn_clamp_direct_fp16_2x32(
    float16_t *out, size_t out_stride,
    float16_t *out_direct, size_t out_direct_stride,
    const float16_t *in0, size_t in0_stride,
    const float16_t *in1, size_t in1_stride,
    const float16_t *bn_mul,
    const float16_t *bn_add,
    const float16_t  minval,
    const float16_t  maxval,
    size_t width, size_t height)
{
    struct KernelArgs
    {
        float16_t minval;
        float16_t maxval;
    } ka;
    ka.minval = minval;
    ka.maxval = maxval;

    __asm__ __volatile__(
        "ldr w21, [%x[args_ptr], %[offsetof_minval]]\n"
        "ldr w20, [%x[args_ptr], %[offsetof_maxval]]\n"
        "cmp %x[width], #0x20\n"
        "dup v13.8h, w21\n"
        "dup v12.8h, w20\n"
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
        "fadd v2.8h, v11.8h, v10.8h\n"
        "fadd v1.8h, v9.8h, v8.8h\n"
        "ldr q21, [x24, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "fadd v0.8h, v7.8h, v6.8h\n"
        "fadd v23.8h, v5.8h, v4.8h\n"
        "ldr q19, [x24, #0x20]\n"
        "ldr q18, [x23, #0x20]\n"
        "fadd v22.8h, v3.8h, v22.8h\n"
        "fadd v21.8h, v21.8h, v20.8h\n"
        "ldr q17, [x24, #0x30]\n"
        "ldr q16, [x23, #0x30]\n"
        "fadd v20.8h, v19.8h, v18.8h\n"
        "fadd v19.8h, v17.8h, v16.8h\n"
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
        "fmla v2.8h, v16.8h, v24.8h\n"
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
        "fmla v1.8h, v18.8h, v25.8h\n"
        "mov x27, x11\n"
        "ldr q10, [x27, #0x0]\n"
        "ldr q8, [x27, #0x10]\n"
        "fmla v0.8h, v17.8h, v26.8h\n"
        "fmla v23.8h, v16.8h, v27.8h\n"
        "ldr q6, [x27, #0x20]\n"
        "ldr q4, [x27, #0x30]\n"
        "mov v17.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v22.8h, v17.8h, v24.8h\n"
        "mov x25, x9\n"
        "mov v17.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v21.8h, v16.8h, v25.8h\n"
        "add x24, x28, %x[in0_stride]\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v20.8h, v17.8h, v26.8h\n"
        "add x23, x27, %x[in1_stride]\n"
        "fmla v19.8h, v16.8h, v27.8h\n"
        "fmin v2.8h, v2.8h, v12.8h\n"
        "add x21, x25, %x[out_direct_stride]\n"
        "cmp x20, #0x2\n"
        "fmin v1.8h, v1.8h, v12.8h\n"
        "fmin v0.8h, v0.8h, v12.8h\n"
        "add x12, x24, %x[in0_stride]\n"
        "add x11, x23, %x[in1_stride]\n"
        "fmin v23.8h, v23.8h, v12.8h\n"
        "fmax v2.8h, v2.8h, v13.8h\n"
        "str q2, [x26, #0x0]\n"
        "add x9, x21, %x[out_direct_stride]\n"
        "fmax v1.8h, v1.8h, v13.8h\n"
        "fmax v0.8h, v0.8h, v13.8h\n"
        "str q1, [x26, #0x10]\n"
        "csel x24, x24, x28, GE\n"
        "fmax v23.8h, v23.8h, v13.8h\n"
        "fmin v22.8h, v22.8h, v12.8h\n"
        "str q0, [x26, #0x20]\n"
        "csel x23, x23, x27, GE\n"
        "fmin v21.8h, v21.8h, v12.8h\n"
        "fmin v20.8h, v20.8h, v12.8h\n"
        "str q23, [x26, #0x30]\n"
        "mov x26, x10\n"
        "fmin v19.8h, v19.8h, v12.8h\n"
        "fmax v22.8h, v22.8h, v13.8h\n"
        "str q22, [x22, #0x0]\n"
        "csel x21, x21, x25, GE\n"
        "fmax v21.8h, v21.8h, v13.8h\n"
        "fmax v20.8h, v20.8h, v13.8h\n"
        "str q21, [x22, #0x10]\n"
        "add x28, x28, #0x40\n"
        "fmax v19.8h, v19.8h, v13.8h\n"
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
        "fadd v2.8h, v11.8h, v10.8h\n"
        "fadd v1.8h, v9.8h, v8.8h\n"
        "ldr q21, [x24, #0x10]\n"
        "ldr q20, [x23, #0x10]\n"
        "fadd v0.8h, v7.8h, v6.8h\n"
        "fadd v23.8h, v5.8h, v4.8h\n"
        "ldr q19, [x24, #0x20]\n"
        "ldr q18, [x23, #0x20]\n"
        "fadd v22.8h, v3.8h, v22.8h\n"
        "fadd v21.8h, v21.8h, v20.8h\n"
        "ldr q17, [x24, #0x30]\n"
        "ldr q16, [x23, #0x30]\n"
        "fadd v20.8h, v19.8h, v18.8h\n"
        "fadd v19.8h, v17.8h, v16.8h\n"
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
        "fmla v2.8h, v16.8h, v24.8h\n"
        "add %x[in0], %x[in0], #0x40\n"
        "mov v16.16b, v1.16b\n"
        "mov v1.16b, v29.16b\n"
        "fmla v1.8h, v16.8h, v25.8h\n"
        "add %x[in1], %x[in1], #0x40\n"
        "mov v16.16b, v0.16b\n"
        "mov v0.16b, v30.16b\n"
        "fmla v0.8h, v16.8h, v26.8h\n"
        "add %x[out], %x[out], #0x40\n"
        "mov v16.16b, v23.16b\n"
        "mov v23.16b, v31.16b\n"
        "fmla v23.8h, v16.8h, v27.8h\n"
        "mov v16.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "fmla v22.8h, v16.8h, v24.8h\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v21.8h, v16.8h, v25.8h\n"
        "mov v16.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v20.8h, v16.8h, v26.8h\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v19.8h, v16.8h, v27.8h\n"
        "fmin v2.8h, v2.8h, v12.8h\n"
        "fmin v1.8h, v1.8h, v12.8h\n"
        "fmin v0.8h, v0.8h, v12.8h\n"
        "fmin v23.8h, v23.8h, v12.8h\n"
        "fmin v22.8h, v22.8h, v12.8h\n"
        "fmin v21.8h, v21.8h, v12.8h\n"
        "fmin v20.8h, v20.8h, v12.8h\n"
        "fmin v19.8h, v19.8h, v12.8h\n"
        "fmax v2.8h, v2.8h, v13.8h\n"
        "fmax v1.8h, v1.8h, v13.8h\n"
        "str q2, [x26, #0x0]\n"
        "fmax v0.8h, v0.8h, v13.8h\n"
        "fmax v23.8h, v23.8h, v13.8h\n"
        "str q1, [x26, #0x10]\n"
        "fmax v22.8h, v22.8h, v13.8h\n"
        "fmax v21.8h, v21.8h, v13.8h\n"
        "str q0, [x26, #0x20]\n"
        "fmax v20.8h, v20.8h, v13.8h\n"
        "fmax v19.8h, v19.8h, v13.8h\n"
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
        "sub %x[width], %x[width], #0x20\n"
        "cmp %x[width], #0x20\n"
        "bge 1b\n"
        "cbz %x[width], 58f\n"
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
        "tbz %x[width], #4, 16f\n"
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
        "tbz %x[width], #3, 12f\n"
        "ldr q7, [x28, #0x0]\n"
        "ldr q6, [x27, #0x0]\n"
        "add x28, x28, #0x10\n"
        "add x27, x27, #0x10\n"
        "ldr q19, [x24, #0x0]\n"
        "ldr q18, [x23, #0x0]\n"
        "add x24, x24, #0x10\n"
        "add x23, x23, #0x10\n"
        "tbz %x[width], #2, 10f\n"
        "ldr d5, [x28], #0x8\n"
        "ldr d4, [x27], #0x8\n"
        "ldr d17, [x24], #0x8\n"
        "ldr d16, [x23], #0x8\n"
        "tbz %x[width], #1, 9f\n"
        "ld1 { v5.s }[2], [x28], #0x4\n"
        "ld1 { v4.s }[2], [x27], #0x4\n"
        "ld1 { v17.s }[2], [x24], #0x4\n"
        "ld1 { v16.s }[2], [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v5.h }[6], [x28], #0x2\n"
        "ld1 { v4.h }[6], [x27], #0x2\n"
        "ld1 { v17.h }[6], [x24], #0x2\n"
        "ld1 { v16.h }[6], [x23], #0x2\n"
        "b 24f\n"
        "9:" // tail loop: unique 1: partial_0_28
        "tbz %x[width], #0, 24f\n"
        "ld1 { v5.h }[4], [x28], #0x2\n"
        "ld1 { v4.h }[4], [x27], #0x2\n"
        "ld1 { v17.h }[4], [x24], #0x2\n"
        "ld1 { v16.h }[4], [x23], #0x2\n"
        "b 24f\n"
        "10:" // tail loop: unique 1: partial_1_24
        "tbz %x[width], #1, 11f\n"
        "ldr s5, [x28], #0x4\n"
        "ldr s4, [x27], #0x4\n"
        "ldr s17, [x24], #0x4\n"
        "ldr s16, [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v5.h }[2], [x28], #0x2\n"
        "ld1 { v4.h }[2], [x27], #0x2\n"
        "ld1 { v17.h }[2], [x24], #0x2\n"
        "ld1 { v16.h }[2], [x23], #0x2\n"
        "b 24f\n"
        "11:" // tail loop: unique 1: partial_0_24
        "tbz %x[width], #0, 24f\n"
        "ldr h5, [x28], #0x2\n"
        "ldr h4, [x27], #0x2\n"
        "ldr h17, [x24], #0x2\n"
        "ldr h16, [x23], #0x2\n"
        "b 24f\n"
        "12:" // tail loop: unique 1: partial_2_16
        "tbz %x[width], #2, 14f\n"
        "ldr d7, [x28], #0x8\n"
        "ldr d6, [x27], #0x8\n"
        "ldr d19, [x24], #0x8\n"
        "ldr d18, [x23], #0x8\n"
        "tbz %x[width], #1, 13f\n"
        "ld1 { v7.s }[2], [x28], #0x4\n"
        "ld1 { v6.s }[2], [x27], #0x4\n"
        "ld1 { v19.s }[2], [x24], #0x4\n"
        "ld1 { v18.s }[2], [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v7.h }[6], [x28], #0x2\n"
        "ld1 { v6.h }[6], [x27], #0x2\n"
        "ld1 { v19.h }[6], [x24], #0x2\n"
        "ld1 { v18.h }[6], [x23], #0x2\n"
        "b 24f\n"
        "13:" // tail loop: unique 1: partial_0_20
        "tbz %x[width], #0, 24f\n"
        "ld1 { v7.h }[4], [x28], #0x2\n"
        "ld1 { v6.h }[4], [x27], #0x2\n"
        "ld1 { v19.h }[4], [x24], #0x2\n"
        "ld1 { v18.h }[4], [x23], #0x2\n"
        "b 24f\n"
        "14:" // tail loop: unique 1: partial_1_16
        "tbz %x[width], #1, 15f\n"
        "ldr s7, [x28], #0x4\n"
        "ldr s6, [x27], #0x4\n"
        "ldr s19, [x24], #0x4\n"
        "ldr s18, [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v7.h }[2], [x28], #0x2\n"
        "ld1 { v6.h }[2], [x27], #0x2\n"
        "ld1 { v19.h }[2], [x24], #0x2\n"
        "ld1 { v18.h }[2], [x23], #0x2\n"
        "b 24f\n"
        "15:" // tail loop: unique 1: partial_0_16
        "tbz %x[width], #0, 24f\n"
        "ldr h7, [x28], #0x2\n"
        "ldr h6, [x27], #0x2\n"
        "ldr h19, [x24], #0x2\n"
        "ldr h18, [x23], #0x2\n"
        "b 24f\n"
        "16:" // tail loop: unique 1: partial_3_0
        "tbz %x[width], #3, 20f\n"
        "ldr q11, [x28, #0x0]\n"
        "ldr q10, [x27, #0x0]\n"
        "add x28, x28, #0x10\n"
        "add x27, x27, #0x10\n"
        "ldr q3, [x24, #0x0]\n"
        "ldr q22, [x23, #0x0]\n"
        "add x24, x24, #0x10\n"
        "add x23, x23, #0x10\n"
        "tbz %x[width], #2, 18f\n"
        "ldr d9, [x28], #0x8\n"
        "ldr d8, [x27], #0x8\n"
        "ldr d21, [x24], #0x8\n"
        "ldr d20, [x23], #0x8\n"
        "tbz %x[width], #1, 17f\n"
        "ld1 { v9.s }[2], [x28], #0x4\n"
        "ld1 { v8.s }[2], [x27], #0x4\n"
        "ld1 { v21.s }[2], [x24], #0x4\n"
        "ld1 { v20.s }[2], [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v9.h }[6], [x28], #0x2\n"
        "ld1 { v8.h }[6], [x27], #0x2\n"
        "ld1 { v21.h }[6], [x24], #0x2\n"
        "ld1 { v20.h }[6], [x23], #0x2\n"
        "b 24f\n"
        "17:" // tail loop: unique 1: partial_0_12
        "tbz %x[width], #0, 24f\n"
        "ld1 { v9.h }[4], [x28], #0x2\n"
        "ld1 { v8.h }[4], [x27], #0x2\n"
        "ld1 { v21.h }[4], [x24], #0x2\n"
        "ld1 { v20.h }[4], [x23], #0x2\n"
        "b 24f\n"
        "18:" // tail loop: unique 1: partial_1_8
        "tbz %x[width], #1, 19f\n"
        "ldr s9, [x28], #0x4\n"
        "ldr s8, [x27], #0x4\n"
        "ldr s21, [x24], #0x4\n"
        "ldr s20, [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v9.h }[2], [x28], #0x2\n"
        "ld1 { v8.h }[2], [x27], #0x2\n"
        "ld1 { v21.h }[2], [x24], #0x2\n"
        "ld1 { v20.h }[2], [x23], #0x2\n"
        "b 24f\n"
        "19:" // tail loop: unique 1: partial_0_8
        "tbz %x[width], #0, 24f\n"
        "ldr h9, [x28], #0x2\n"
        "ldr h8, [x27], #0x2\n"
        "ldr h21, [x24], #0x2\n"
        "ldr h20, [x23], #0x2\n"
        "b 24f\n"
        "20:" // tail loop: unique 1: partial_2_0
        "tbz %x[width], #2, 22f\n"
        "ldr d11, [x28], #0x8\n"
        "ldr d10, [x27], #0x8\n"
        "ldr d3, [x24], #0x8\n"
        "ldr d22, [x23], #0x8\n"
        "tbz %x[width], #1, 21f\n"
        "ld1 { v11.s }[2], [x28], #0x4\n"
        "ld1 { v10.s }[2], [x27], #0x4\n"
        "ld1 { v3.s }[2], [x24], #0x4\n"
        "ld1 { v22.s }[2], [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v11.h }[6], [x28], #0x2\n"
        "ld1 { v10.h }[6], [x27], #0x2\n"
        "ld1 { v3.h }[6], [x24], #0x2\n"
        "ld1 { v22.h }[6], [x23], #0x2\n"
        "b 24f\n"
        "21:" // tail loop: unique 1: partial_0_4
        "tbz %x[width], #0, 24f\n"
        "ld1 { v11.h }[4], [x28], #0x2\n"
        "ld1 { v10.h }[4], [x27], #0x2\n"
        "ld1 { v3.h }[4], [x24], #0x2\n"
        "ld1 { v22.h }[4], [x23], #0x2\n"
        "b 24f\n"
        "22:" // tail loop: unique 1: partial_1_0
        "tbz %x[width], #1, 23f\n"
        "ldr s11, [x28], #0x4\n"
        "ldr s10, [x27], #0x4\n"
        "ldr s3, [x24], #0x4\n"
        "ldr s22, [x23], #0x4\n"
        "tbz %x[width], #0, 24f\n"
        "ld1 { v11.h }[2], [x28], #0x2\n"
        "ld1 { v10.h }[2], [x27], #0x2\n"
        "ld1 { v3.h }[2], [x24], #0x2\n"
        "ld1 { v22.h }[2], [x23], #0x2\n"
        "b 24f\n"
        "23:" // tail loop: unique 1: partial_0_0
        "ldr h11, [x28], #0x2\n"
        "ldr h10, [x27], #0x2\n"
        "ldr h3, [x24], #0x2\n"
        "ldr h22, [x23], #0x2\n"
        "24:" // tail loop: unique 1: Done
        "fadd v2.8h, v11.8h, v10.8h\n"
        "fadd v1.8h, v9.8h, v8.8h\n"
        "fadd v0.8h, v7.8h, v6.8h\n"
        "fadd v23.8h, v5.8h, v4.8h\n"
        "fadd v22.8h, v3.8h, v22.8h\n"
        "fadd v21.8h, v21.8h, v20.8h\n"
        "fadd v20.8h, v19.8h, v18.8h\n"
        "fadd v19.8h, v17.8h, v16.8h\n"
        "cbz %x[out_direct], 41f\n"
        "tbz %x[width], #4, 32f\n"
        "str q2, [x25, #0x0]\n"
        "str q1, [x25, #0x10]\n"
        "add x25, x25, #0x20\n"
        "str q22, [x21, #0x0]\n"
        "str q21, [x21, #0x10]\n"
        "add x21, x21, #0x20\n"
        "tbz %x[width], #3, 28f\n"
        "str q0, [x25, #0x0]\n"
        "add x25, x25, #0x10\n"
        "str q20, [x21, #0x0]\n"
        "add x21, x21, #0x10\n"
        "tbz %x[width], #2, 26f\n"
        "str d23, [x25], #0x8\n"
        "str d19, [x21], #0x8\n"
        "tbz %x[width], #1, 25f\n"
        "st1 { v23.s }[2], [x25], #0x4\n"
        "st1 { v19.s }[2], [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v23.h }[6], [x25], #0x2\n"
        "st1 { v19.h }[6], [x21], #0x2\n"
        "b 40f\n"
        "25:" // tail loop: Main loop: unique 2: partial_0_28
        "tbz %x[width], #0, 40f\n"
        "st1 { v23.h }[4], [x25], #0x2\n"
        "st1 { v19.h }[4], [x21], #0x2\n"
        "b 40f\n"
        "26:" // tail loop: Main loop: unique 2: partial_1_24
        "tbz %x[width], #1, 27f\n"
        "str s23, [x25], #0x4\n"
        "str s19, [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v23.h }[2], [x25], #0x2\n"
        "st1 { v19.h }[2], [x21], #0x2\n"
        "b 40f\n"
        "27:" // tail loop: Main loop: unique 2: partial_0_24
        "tbz %x[width], #0, 40f\n"
        "str h23, [x25], #0x2\n"
        "str h19, [x21], #0x2\n"
        "b 40f\n"
        "28:" // tail loop: Main loop: unique 2: partial_2_16
        "tbz %x[width], #2, 30f\n"
        "str d0, [x25], #0x8\n"
        "str d20, [x21], #0x8\n"
        "tbz %x[width], #1, 29f\n"
        "st1 { v0.s }[2], [x25], #0x4\n"
        "st1 { v20.s }[2], [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v0.h }[6], [x25], #0x2\n"
        "st1 { v20.h }[6], [x21], #0x2\n"
        "b 40f\n"
        "29:" // tail loop: Main loop: unique 2: partial_0_20
        "tbz %x[width], #0, 40f\n"
        "st1 { v0.h }[4], [x25], #0x2\n"
        "st1 { v20.h }[4], [x21], #0x2\n"
        "b 40f\n"
        "30:" // tail loop: Main loop: unique 2: partial_1_16
        "tbz %x[width], #1, 31f\n"
        "str s0, [x25], #0x4\n"
        "str s20, [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v0.h }[2], [x25], #0x2\n"
        "st1 { v20.h }[2], [x21], #0x2\n"
        "b 40f\n"
        "31:" // tail loop: Main loop: unique 2: partial_0_16
        "tbz %x[width], #0, 40f\n"
        "str h0, [x25], #0x2\n"
        "str h20, [x21], #0x2\n"
        "b 40f\n"
        "32:" // tail loop: Main loop: unique 2: partial_3_0
        "tbz %x[width], #3, 36f\n"
        "str q2, [x25, #0x0]\n"
        "add x25, x25, #0x10\n"
        "str q22, [x21, #0x0]\n"
        "add x21, x21, #0x10\n"
        "tbz %x[width], #2, 34f\n"
        "str d1, [x25], #0x8\n"
        "str d21, [x21], #0x8\n"
        "tbz %x[width], #1, 33f\n"
        "st1 { v1.s }[2], [x25], #0x4\n"
        "st1 { v21.s }[2], [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v1.h }[6], [x25], #0x2\n"
        "st1 { v21.h }[6], [x21], #0x2\n"
        "b 40f\n"
        "33:" // tail loop: Main loop: unique 2: partial_0_12
        "tbz %x[width], #0, 40f\n"
        "st1 { v1.h }[4], [x25], #0x2\n"
        "st1 { v21.h }[4], [x21], #0x2\n"
        "b 40f\n"
        "34:" // tail loop: Main loop: unique 2: partial_1_8
        "tbz %x[width], #1, 35f\n"
        "str s1, [x25], #0x4\n"
        "str s21, [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v1.h }[2], [x25], #0x2\n"
        "st1 { v21.h }[2], [x21], #0x2\n"
        "b 40f\n"
        "35:" // tail loop: Main loop: unique 2: partial_0_8
        "tbz %x[width], #0, 40f\n"
        "str h1, [x25], #0x2\n"
        "str h21, [x21], #0x2\n"
        "b 40f\n"
        "36:" // tail loop: Main loop: unique 2: partial_2_0
        "tbz %x[width], #2, 38f\n"
        "str d2, [x25], #0x8\n"
        "str d22, [x21], #0x8\n"
        "tbz %x[width], #1, 37f\n"
        "st1 { v2.s }[2], [x25], #0x4\n"
        "st1 { v22.s }[2], [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v2.h }[6], [x25], #0x2\n"
        "st1 { v22.h }[6], [x21], #0x2\n"
        "b 40f\n"
        "37:" // tail loop: Main loop: unique 2: partial_0_4
        "tbz %x[width], #0, 40f\n"
        "st1 { v2.h }[4], [x25], #0x2\n"
        "st1 { v22.h }[4], [x21], #0x2\n"
        "b 40f\n"
        "38:" // tail loop: Main loop: unique 2: partial_1_0
        "tbz %x[width], #1, 39f\n"
        "str s2, [x25], #0x4\n"
        "str s22, [x21], #0x4\n"
        "tbz %x[width], #0, 40f\n"
        "st1 { v2.h }[2], [x25], #0x2\n"
        "st1 { v22.h }[2], [x21], #0x2\n"
        "b 40f\n"
        "39:" // tail loop: Main loop: unique 2: partial_0_0
        "str h2, [x25], #0x2\n"
        "str h22, [x21], #0x2\n"
        "40:" // tail loop: Main loop: unique 2: Done
        "41:" // tail loop: Main loop: No direct output
        "mov v16.16b, v2.16b\n"
        "mov v2.16b, v28.16b\n"
        "fmla v2.8h, v16.8h, v24.8h\n"
        "mov v16.16b, v1.16b\n"
        "mov v1.16b, v29.16b\n"
        "fmla v1.8h, v16.8h, v25.8h\n"
        "mov v16.16b, v0.16b\n"
        "mov v0.16b, v30.16b\n"
        "fmla v0.8h, v16.8h, v26.8h\n"
        "mov v16.16b, v23.16b\n"
        "mov v23.16b, v31.16b\n"
        "fmla v23.8h, v16.8h, v27.8h\n"
        "mov v16.16b, v22.16b\n"
        "mov v22.16b, v28.16b\n"
        "fmla v22.8h, v16.8h, v24.8h\n"
        "mov v16.16b, v21.16b\n"
        "mov v21.16b, v29.16b\n"
        "fmla v21.8h, v16.8h, v25.8h\n"
        "mov v16.16b, v20.16b\n"
        "mov v20.16b, v30.16b\n"
        "fmla v20.8h, v16.8h, v26.8h\n"
        "mov v16.16b, v19.16b\n"
        "mov v19.16b, v31.16b\n"
        "fmla v19.8h, v16.8h, v27.8h\n"
        "fmin v2.8h, v2.8h, v12.8h\n"
        "fmin v1.8h, v1.8h, v12.8h\n"
        "fmin v0.8h, v0.8h, v12.8h\n"
        "fmin v23.8h, v23.8h, v12.8h\n"
        "fmin v22.8h, v22.8h, v12.8h\n"
        "fmin v21.8h, v21.8h, v12.8h\n"
        "fmin v20.8h, v20.8h, v12.8h\n"
        "fmin v19.8h, v19.8h, v12.8h\n"
        "fmax v2.8h, v2.8h, v13.8h\n"
        "fmax v1.8h, v1.8h, v13.8h\n"
        "fmax v0.8h, v0.8h, v13.8h\n"
        "fmax v23.8h, v23.8h, v13.8h\n"
        "fmax v22.8h, v22.8h, v13.8h\n"
        "fmax v21.8h, v21.8h, v13.8h\n"
        "fmax v20.8h, v20.8h, v13.8h\n"
        "fmax v19.8h, v19.8h, v13.8h\n"
        "tbz %x[width], #4, 49f\n"
        "str q2, [x26, #0x0]\n"
        "str q1, [x26, #0x10]\n"
        "add x26, x26, #0x20\n"
        "str q22, [x22, #0x0]\n"
        "str q21, [x22, #0x10]\n"
        "add x22, x22, #0x20\n"
        "tbz %x[width], #3, 45f\n"
        "str q0, [x26, #0x0]\n"
        "add x26, x26, #0x10\n"
        "str q20, [x22, #0x0]\n"
        "add x22, x22, #0x10\n"
        "tbz %x[width], #2, 43f\n"
        "str d23, [x26], #0x8\n"
        "str d19, [x22], #0x8\n"
        "tbz %x[width], #1, 42f\n"
        "st1 { v23.s }[2], [x26], #0x4\n"
        "st1 { v19.s }[2], [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v23.h }[6], [x26], #0x2\n"
        "st1 { v19.h }[6], [x22], #0x2\n"
        "b 57f\n"
        "42:" // tail loop: unique 3: partial_0_28
        "tbz %x[width], #0, 57f\n"
        "st1 { v23.h }[4], [x26], #0x2\n"
        "st1 { v19.h }[4], [x22], #0x2\n"
        "b 57f\n"
        "43:" // tail loop: unique 3: partial_1_24
        "tbz %x[width], #1, 44f\n"
        "str s23, [x26], #0x4\n"
        "str s19, [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v23.h }[2], [x26], #0x2\n"
        "st1 { v19.h }[2], [x22], #0x2\n"
        "b 57f\n"
        "44:" // tail loop: unique 3: partial_0_24
        "tbz %x[width], #0, 57f\n"
        "str h23, [x26], #0x2\n"
        "str h19, [x22], #0x2\n"
        "b 57f\n"
        "45:" // tail loop: unique 3: partial_2_16
        "tbz %x[width], #2, 47f\n"
        "str d0, [x26], #0x8\n"
        "str d20, [x22], #0x8\n"
        "tbz %x[width], #1, 46f\n"
        "st1 { v0.s }[2], [x26], #0x4\n"
        "st1 { v20.s }[2], [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v0.h }[6], [x26], #0x2\n"
        "st1 { v20.h }[6], [x22], #0x2\n"
        "b 57f\n"
        "46:" // tail loop: unique 3: partial_0_20
        "tbz %x[width], #0, 57f\n"
        "st1 { v0.h }[4], [x26], #0x2\n"
        "st1 { v20.h }[4], [x22], #0x2\n"
        "b 57f\n"
        "47:" // tail loop: unique 3: partial_1_16
        "tbz %x[width], #1, 48f\n"
        "str s0, [x26], #0x4\n"
        "str s20, [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v0.h }[2], [x26], #0x2\n"
        "st1 { v20.h }[2], [x22], #0x2\n"
        "b 57f\n"
        "48:" // tail loop: unique 3: partial_0_16
        "tbz %x[width], #0, 57f\n"
        "str h0, [x26], #0x2\n"
        "str h20, [x22], #0x2\n"
        "b 57f\n"
        "49:" // tail loop: unique 3: partial_3_0
        "tbz %x[width], #3, 53f\n"
        "str q2, [x26, #0x0]\n"
        "add x26, x26, #0x10\n"
        "str q22, [x22, #0x0]\n"
        "add x22, x22, #0x10\n"
        "tbz %x[width], #2, 51f\n"
        "str d1, [x26], #0x8\n"
        "str d21, [x22], #0x8\n"
        "tbz %x[width], #1, 50f\n"
        "st1 { v1.s }[2], [x26], #0x4\n"
        "st1 { v21.s }[2], [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v1.h }[6], [x26], #0x2\n"
        "st1 { v21.h }[6], [x22], #0x2\n"
        "b 57f\n"
        "50:" // tail loop: unique 3: partial_0_12
        "tbz %x[width], #0, 57f\n"
        "st1 { v1.h }[4], [x26], #0x2\n"
        "st1 { v21.h }[4], [x22], #0x2\n"
        "b 57f\n"
        "51:" // tail loop: unique 3: partial_1_8
        "tbz %x[width], #1, 52f\n"
        "str s1, [x26], #0x4\n"
        "str s21, [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v1.h }[2], [x26], #0x2\n"
        "st1 { v21.h }[2], [x22], #0x2\n"
        "b 57f\n"
        "52:" // tail loop: unique 3: partial_0_8
        "tbz %x[width], #0, 57f\n"
        "str h1, [x26], #0x2\n"
        "str h21, [x22], #0x2\n"
        "b 57f\n"
        "53:" // tail loop: unique 3: partial_2_0
        "tbz %x[width], #2, 55f\n"
        "str d2, [x26], #0x8\n"
        "str d22, [x22], #0x8\n"
        "tbz %x[width], #1, 54f\n"
        "st1 { v2.s }[2], [x26], #0x4\n"
        "st1 { v22.s }[2], [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v2.h }[6], [x26], #0x2\n"
        "st1 { v22.h }[6], [x22], #0x2\n"
        "b 57f\n"
        "54:" // tail loop: unique 3: partial_0_4
        "tbz %x[width], #0, 57f\n"
        "st1 { v2.h }[4], [x26], #0x2\n"
        "st1 { v22.h }[4], [x22], #0x2\n"
        "b 57f\n"
        "55:" // tail loop: unique 3: partial_1_0
        "tbz %x[width], #1, 56f\n"
        "str s2, [x26], #0x4\n"
        "str s22, [x22], #0x4\n"
        "tbz %x[width], #0, 57f\n"
        "st1 { v2.h }[2], [x26], #0x2\n"
        "st1 { v22.h }[2], [x22], #0x2\n"
        "b 57f\n"
        "56:" // tail loop: unique 3: partial_0_0
        "str h2, [x26], #0x2\n"
        "str h22, [x22], #0x2\n"
        "57:" // tail loop: unique 3: Done
        "subs x20, x20, #0x2\n"
        "bgt 8b\n"
        "58:" // odd columns skip
        : [bn_add] "+&r"(bn_add), [bn_mul] "+&r"(bn_mul), [in0] "+&r"(in0), [in1] "+&r"(in1), [out] "+&r"(out), [out_direct] "+&r"(out_direct), [width] "+&r"(width)
        : [args_ptr] "r"(&ka), [height] "r"(height), [in0_stride] "r"(in0_stride), [in1_stride] "r"(in1_stride), [offsetof_maxval] "I"(offsetof(KernelArgs, maxval)), [offsetof_minval] "I"(offsetof(KernelArgs, minval)), [out_direct_stride] "r"(out_direct_stride), [out_stride] "r"(out_stride)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}

} // namespace

namespace arm_compute
{
namespace cpu
{
void add_mul_add_fp16_neon(const ITensor *input1, const ITensor *input2, const ITensor *bn_mul, const ITensor *bn_add,
                           ITensor *add_output, ITensor *final_output, ConvertPolicy policy, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    const size_t out_stride        = final_output->info()->strides_in_bytes()[1];
    const size_t out_direct_stride = (add_output != nullptr) ? add_output->info()->strides_in_bytes()[1] : 0;
    const size_t in0_stride        = input1->info()->strides_in_bytes()[1];
    const size_t in1_stride        = input2->info()->strides_in_bytes()[1];

    float16_t minval = std::numeric_limits<half>::lowest();
    float16_t maxval = std::numeric_limits<half>::max();

    if(act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU)
    {
        minval = static_cast<float16_t>(0.f);
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
    {
        minval = static_cast<float16_t>(0.f);
        maxval = static_cast<float16_t>(act_info.a());
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
    {
        minval = static_cast<float16_t>(act_info.b());
        maxval = static_cast<float16_t>(act_info.a());
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
            a64_add_bn_clamp_direct_fp16_2x32(
                reinterpret_cast<float16_t *>(out_it.ptr()), out_stride,
                reinterpret_cast<float16_t *>(add_out_it.ptr()), out_direct_stride,
                reinterpret_cast<float16_t *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<float16_t *>(in2_it.ptr()), in1_stride,
                reinterpret_cast<float16_t *>(bn_mul->buffer()),
                reinterpret_cast<float16_t *>(bn_add->buffer()),
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
            a64_add_bn_clamp_direct_fp16_2x32(
                reinterpret_cast<float16_t *>(out_it.ptr()), out_stride,
                nullptr, out_direct_stride,
                reinterpret_cast<float16_t *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<float16_t *>(in2_it.ptr()), in1_stride,
                reinterpret_cast<float16_t *>(bn_mul->buffer()),
                reinterpret_cast<float16_t *>(bn_add->buffer()),
                minval,
                maxval,
                width, height);
        },
        in1_it, in2_it, out_it);
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // defined(__aarch64__) && defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
