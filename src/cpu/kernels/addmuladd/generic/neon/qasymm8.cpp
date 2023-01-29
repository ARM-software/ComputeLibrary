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
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef __aarch64__
namespace
{
void a64_add_bn_clamp_direct_u8_fp32_2x16(
    uint8_t *out, size_t out_stride,
    uint8_t *out_direct, size_t out_direct_stride,
    const uint8_t *in0, size_t in0_stride,
    const uint8_t *in1, size_t in1_stride,
    const float *bn_mul,
    const float *bn_add,
    const uint8_t minval,
    const uint8_t maxval,
    int32_t out_zeropt, float out_scale,
    int32_t out_direct_zeropt, float out_direct_scale,
    int32_t in0_zeropt, float in0_scale,
    int32_t in1_zeropt, float in1_scale,
    size_t width, size_t height)
{
    float scales[4] = { in0_scale, in1_scale, 1.0f / out_scale, 1.0f / out_direct_scale };
    struct KernelArgs
    {
        const float *scales;
        int32_t      in0_zeropt;
        int32_t      in1_zeropt;
        int32_t      out_zeropt;
        int32_t      out_direct_zeropt;
        int32_t      minval;
        int32_t      maxval;
    } ka;
    ka.scales            = scales;
    ka.in0_zeropt        = in0_zeropt;
    ka.in1_zeropt        = in1_zeropt;
    ka.out_zeropt        = out_zeropt;
    ka.out_direct_zeropt = out_direct_zeropt;
    ka.minval            = minval;
    ka.maxval            = maxval;

    __asm__ __volatile__(
        "ldr x20, [%x[args_ptr], %[offsetof_scales]]\n"
        "ld1 { v0.4s }, [x20]\n"
        "cmp %x[width], #0x10\n"
        "blt 5f\n"
        "1:" // Column loop
        "ldr q24, [%x[bn_mul], #0x0]\n"
        "ldr q25, [%x[bn_mul], #0x10]\n"
        "mov x23, %x[height]\n"
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
        "2:" // Row loop
        "mov x28, x12\n"
        "ldr d4, [x28, #0x0]\n"
        "ldr d3, [x28, #0x8]\n"
        "add x21, x28, %x[in0_stride]\n"
        "mov x27, x11\n"
        "ldr d13, [x27, #0x0]\n"
        "ldr d12, [x27, #0x8]\n"
        "cmp x23, #0x2\n"
        "add x12, x21, %x[in0_stride]\n"
        "csel x21, x21, x28, GE\n"
        "ldr d2, [x21, #0x0]\n"
        "ldr d11, [x21, #0x8]\n"
        "add x20, x27, %x[in1_stride]\n"
        "add x11, x20, %x[in1_stride]\n"
        "ldr w21, [%x[args_ptr], %[offsetof_in0_zeropt]]\n"
        "ushll v4.8h, v4.8b, #0x0\n"
        "csel x20, x20, x27, GE\n"
        "ldr d10, [x20, #0x0]\n"
        "ldr d9, [x20, #0x8]\n"
        "ushll v3.8h, v3.8b, #0x0\n"
        "ushll v2.8h, v2.8b, #0x0\n"
        "ushll v11.8h, v11.8b, #0x0\n"
        "ldr w20, [%x[args_ptr], %[offsetof_in1_zeropt]]\n"
        "mov x26, x10\n"
        "dup v16.8h, w21\n"
        "ushll v13.8h, v13.8b, #0x0\n"
        "mov x25, x9\n"
        "add x24, x26, %x[out_stride]\n"
        "ushll v12.8h, v12.8b, #0x0\n"
        "ushll v10.8h, v10.8b, #0x0\n"
        "add x22, x25, %x[out_direct_stride]\n"
        "add x10, x24, %x[out_stride]\n"
        "ushll v9.8h, v9.8b, #0x0\n"
        "ssubl v1.4s, v4.4h, v16.4h\n"
        "add x9, x22, %x[out_direct_stride]\n"
        "csel x24, x24, x26, GE\n"
        "ssubl2 v4.4s, v4.8h, v16.8h\n"
        "ssubl v23.4s, v3.4h, v16.4h\n"
        "csel x22, x22, x25, GE\n"
        "ssubl2 v3.4s, v3.8h, v16.8h\n"
        "ssubl v22.4s, v2.4h, v16.4h\n"
        "ssubl2 v2.4s, v2.8h, v16.8h\n"
        "ssubl v21.4s, v11.4h, v16.4h\n"
        "ssubl2 v11.4s, v11.8h, v16.8h\n"
        "dup v20.8h, w20\n"
        "ssubl v19.4s, v13.4h, v20.4h\n"
        "ssubl2 v13.4s, v13.8h, v20.8h\n"
        "ssubl v18.4s, v12.4h, v20.4h\n"
        "ssubl2 v12.4s, v12.8h, v20.8h\n"
        "ssubl v17.4s, v10.4h, v20.4h\n"
        "ssubl2 v10.4s, v10.8h, v20.8h\n"
        "ssubl v16.4s, v9.4h, v20.4h\n"
        "ssubl2 v9.4s, v9.8h, v20.8h\n"
        "scvtf v8.4s, v1.4s\n"
        "scvtf v7.4s, v4.4s\n"
        "scvtf v6.4s, v23.4s\n"
        "scvtf v5.4s, v3.4s\n"
        "scvtf v4.4s, v22.4s\n"
        "scvtf v3.4s, v2.4s\n"
        "scvtf v2.4s, v21.4s\n"
        "scvtf v1.4s, v11.4s\n"
        "scvtf v19.4s, v19.4s\n"
        "fmul v8.4s, v8.4s, v0.s[0]\n"
        "fmla v8.4s, v19.4s, v0.s[1]\n"
        "scvtf v13.4s, v13.4s\n"
        "fmul v7.4s, v7.4s, v0.s[0]\n"
        "fmla v7.4s, v13.4s, v0.s[1]\n"
        "scvtf v18.4s, v18.4s\n"
        "fmul v6.4s, v6.4s, v0.s[0]\n"
        "fmla v6.4s, v18.4s, v0.s[1]\n"
        "scvtf v12.4s, v12.4s\n"
        "fmul v5.4s, v5.4s, v0.s[0]\n"
        "fmla v5.4s, v12.4s, v0.s[1]\n"
        "scvtf v17.4s, v17.4s\n"
        "fmul v4.4s, v4.4s, v0.s[0]\n"
        "fmla v4.4s, v17.4s, v0.s[1]\n"
        "scvtf v10.4s, v10.4s\n"
        "fmul v3.4s, v3.4s, v0.s[0]\n"
        "fmla v3.4s, v10.4s, v0.s[1]\n"
        "scvtf v16.4s, v16.4s\n"
        "fmul v2.4s, v2.4s, v0.s[0]\n"
        "fmla v2.4s, v16.4s, v0.s[1]\n"
        "scvtf v9.4s, v9.4s\n"
        "fmul v1.4s, v1.4s, v0.s[0]\n"
        "fmla v1.4s, v9.4s, v0.s[1]\n"
        "cbz %x[out_direct], 3f\n"
        "fmul v23.4s, v8.4s, v0.s[3]\n"
        "fmul v22.4s, v7.4s, v0.s[3]\n"
        "ldr w20, [%x[args_ptr], %[offsetof_out_direct_zeropt]]\n"
        "fmul v21.4s, v6.4s, v0.s[3]\n"
        "fmul v20.4s, v5.4s, v0.s[3]\n"
        "fmul v19.4s, v4.4s, v0.s[3]\n"
        "fmul v18.4s, v3.4s, v0.s[3]\n"
        "fmul v16.4s, v2.4s, v0.s[3]\n"
        "fmul v17.4s, v1.4s, v0.s[3]\n"
        "fcvtas v23.4s, v23.4s\n"
        "fcvtas v22.4s, v22.4s\n"
        "fcvtas v21.4s, v21.4s\n"
        "fcvtas v20.4s, v20.4s\n"
        "fcvtas v19.4s, v19.4s\n"
        "fcvtas v18.4s, v18.4s\n"
        "fcvtas v16.4s, v16.4s\n"
        "fcvtas v17.4s, v17.4s\n"
        "uzp1 v22.8h, v23.8h, v22.8h\n"
        "uzp1 v20.8h, v21.8h, v20.8h\n"
        "uzp1 v18.8h, v19.8h, v18.8h\n"
        "uzp1 v17.8h, v16.8h, v17.8h\n"
        "dup v16.8h, w20\n"
        "add v22.8h, v22.8h, v16.8h\n"
        "add v20.8h, v20.8h, v16.8h\n"
        "add v18.8h, v18.8h, v16.8h\n"
        "add v17.8h, v17.8h, v16.8h\n"
        "movi v16.8h, #0xff\n"
        "smin v22.8h, v22.8h, v16.8h\n"
        "smin v20.8h, v20.8h, v16.8h\n"
        "smin v18.8h, v18.8h, v16.8h\n"
        "smin v17.8h, v17.8h, v16.8h\n"
        "movi v16.8h, #0x0\n"
        "smax v22.8h, v22.8h, v16.8h\n"
        "smax v20.8h, v20.8h, v16.8h\n"
        "smax v18.8h, v18.8h, v16.8h\n"
        "smax v17.8h, v17.8h, v16.8h\n"
        "xtn v22.8b, v22.8h\n"
        "str d22, [x25, #0x0]\n"
        "xtn v20.8b, v20.8h\n"
        "xtn v18.8b, v18.8h\n"
        "str d20, [x25, #0x8]\n"
        "xtn v17.8b, v17.8h\n"
        "str d18, [x22, #0x0]\n"
        "str d17, [x22, #0x8]\n"
        "3:" // Main loop: No direct output
        "mov v19.16b, v28.16b\n"
        "mov v13.16b, v29.16b\n"
        "fmla v19.4s, v8.4s, v24.4s\n"
        "ldr w22, [%x[args_ptr], %[offsetof_out_zeropt]]\n"
        "mov v18.16b, v30.16b\n"
        "mov v12.16b, v31.16b\n"
        "fmla v13.4s, v7.4s, v25.4s\n"
        "ldr w21, [%x[args_ptr], %[offsetof_maxval]]\n"
        "mov v17.16b, v28.16b\n"
        "mov v10.16b, v29.16b\n"
        "fmla v18.4s, v6.4s, v26.4s\n"
        "ldr w20, [%x[args_ptr], %[offsetof_minval]]\n"
        "mov v16.16b, v30.16b\n"
        "mov v9.16b, v31.16b\n"
        "fmla v12.4s, v5.4s, v27.4s\n"
        "subs x23, x23, #0x2\n"
        "fmla v17.4s, v4.4s, v24.4s\n"
        "fmla v10.4s, v3.4s, v25.4s\n"
        "fmul v8.4s, v19.4s, v0.s[2]\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "fmla v9.4s, v1.4s, v27.4s\n"
        "fmul v7.4s, v13.4s, v0.s[2]\n"
        "fmul v6.4s, v18.4s, v0.s[2]\n"
        "fmul v5.4s, v12.4s, v0.s[2]\n"
        "fmul v4.4s, v17.4s, v0.s[2]\n"
        "fmul v3.4s, v10.4s, v0.s[2]\n"
        "fmul v2.4s, v16.4s, v0.s[2]\n"
        "fmul v1.4s, v9.4s, v0.s[2]\n"
        "fcvtas v8.4s, v8.4s\n"
        "fcvtas v7.4s, v7.4s\n"
        "fcvtas v6.4s, v6.4s\n"
        "fcvtas v5.4s, v5.4s\n"
        "fcvtas v4.4s, v4.4s\n"
        "fcvtas v3.4s, v3.4s\n"
        "fcvtas v2.4s, v2.4s\n"
        "fcvtas v1.4s, v1.4s\n"
        "uzp1 v7.8h, v8.8h, v7.8h\n"
        "uzp1 v5.8h, v6.8h, v5.8h\n"
        "uzp1 v3.8h, v4.8h, v3.8h\n"
        "uzp1 v1.8h, v2.8h, v1.8h\n"
        "dup v16.8h, w22\n"
        "add v7.8h, v7.8h, v16.8h\n"
        "add v5.8h, v5.8h, v16.8h\n"
        "add v3.8h, v3.8h, v16.8h\n"
        "add v1.8h, v1.8h, v16.8h\n"
        "dup v16.8h, w21\n"
        "smin v7.8h, v7.8h, v16.8h\n"
        "smin v5.8h, v5.8h, v16.8h\n"
        "smin v3.8h, v3.8h, v16.8h\n"
        "smin v1.8h, v1.8h, v16.8h\n"
        "dup v16.8h, w20\n"
        "smax v7.8h, v7.8h, v16.8h\n"
        "smax v5.8h, v5.8h, v16.8h\n"
        "smax v3.8h, v3.8h, v16.8h\n"
        "smax v1.8h, v1.8h, v16.8h\n"
        "xtn v7.8b, v7.8h\n"
        "str d7, [x26, #0x0]\n"
        "xtn v5.8b, v5.8h\n"
        "xtn v3.8b, v3.8h\n"
        "str d5, [x26, #0x8]\n"
        "xtn v1.8b, v1.8h\n"
        "str d3, [x24, #0x0]\n"
        "str d1, [x24, #0x8]\n"
        "bgt 2b\n"
        "add %x[in0], %x[in0], #0x10\n"
        "add %x[in1], %x[in1], #0x10\n"
        "add %x[out], %x[out], #0x10\n"
        "cbz %x[out_direct], 4f\n"
        "add %x[out_direct], %x[out_direct], #0x10\n"
        "4:" // No direct pointer update
        "sub %x[width], %x[width], #0x10\n"
        "cmp %x[width], #0x10\n"
        "bge 1b\n"
        "cbz %x[width], 32f\n"
        "5:" // main loop skip
        "ldr q24, [%x[bn_mul], #0x0]\n"
        "ldr q25, [%x[bn_mul], #0x10]\n"
        "mov x23, %x[height]\n"
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
        "6:" // tail loop: Row loop
        "mov x28, x12\n"
        "mov x27, x11\n"
        "mov x26, x10\n"
        "mov x25, x9\n"
        "add x21, x28, %x[in0_stride]\n"
        "add x20, x27, %x[in1_stride]\n"
        "add x24, x26, %x[out_stride]\n"
        "add x22, x25, %x[out_direct_stride]\n"
        "cmp x23, #0x2\n"
        "add x12, x21, %x[in0_stride]\n"
        "add x11, x20, %x[in1_stride]\n"
        "add x10, x24, %x[out_stride]\n"
        "add x9, x22, %x[out_direct_stride]\n"
        "csel x21, x21, x28, GE\n"
        "csel x20, x20, x27, GE\n"
        "csel x24, x24, x26, GE\n"
        "csel x22, x22, x25, GE\n"
        "tbz %x[width], #3, 10f\n"
        "ldr d4, [x28, #0x0]\n"
        "ldr d13, [x27, #0x0]\n"
        "add x28, x28, #0x8\n"
        "add x27, x27, #0x8\n"
        "ldr d2, [x21, #0x0]\n"
        "ldr d10, [x20, #0x0]\n"
        "add x21, x21, #0x8\n"
        "add x20, x20, #0x8\n"
        "tbz %x[width], #2, 8f\n"
        "ldr s3, [x28], #0x4\n"
        "ldr s12, [x27], #0x4\n"
        "ldr s11, [x21], #0x4\n"
        "ldr s9, [x20], #0x4\n"
        "tbz %x[width], #1, 7f\n"
        "ld1 { v3.h }[2], [x28], #0x2\n"
        "ld1 { v12.h }[2], [x27], #0x2\n"
        "ld1 { v11.h }[2], [x21], #0x2\n"
        "ld1 { v9.h }[2], [x20], #0x2\n"
        "tbz %x[width], #0, 14f\n"
        "ld1 { v3.b }[6], [x28], #0x1\n"
        "ld1 { v12.b }[6], [x27], #0x1\n"
        "ld1 { v11.b }[6], [x21], #0x1\n"
        "ld1 { v9.b }[6], [x20], #0x1\n"
        "b 14f\n"
        "7:" // tail loop: unique 1: partial_0_12
        "tbz %x[width], #0, 14f\n"
        "ld1 { v3.b }[4], [x28], #0x1\n"
        "ld1 { v12.b }[4], [x27], #0x1\n"
        "ld1 { v11.b }[4], [x21], #0x1\n"
        "ld1 { v9.b }[4], [x20], #0x1\n"
        "b 14f\n"
        "8:" // tail loop: unique 1: partial_1_8
        "tbz %x[width], #1, 9f\n"
        "ldr h3, [x28], #0x2\n"
        "ldr h12, [x27], #0x2\n"
        "ldr h11, [x21], #0x2\n"
        "ldr h9, [x20], #0x2\n"
        "tbz %x[width], #0, 14f\n"
        "ld1 { v3.b }[2], [x28], #0x1\n"
        "ld1 { v12.b }[2], [x27], #0x1\n"
        "ld1 { v11.b }[2], [x21], #0x1\n"
        "ld1 { v9.b }[2], [x20], #0x1\n"
        "b 14f\n"
        "9:" // tail loop: unique 1: partial_0_8
        "tbz %x[width], #0, 14f\n"
        "ldr b3, [x28], #0x1\n"
        "ldr b12, [x27], #0x1\n"
        "ldr b11, [x21], #0x1\n"
        "ldr b9, [x20], #0x1\n"
        "b 14f\n"
        "10:" // tail loop: unique 1: partial_2_0
        "tbz %x[width], #2, 12f\n"
        "ldr s4, [x28], #0x4\n"
        "ldr s13, [x27], #0x4\n"
        "ldr s2, [x21], #0x4\n"
        "ldr s10, [x20], #0x4\n"
        "tbz %x[width], #1, 11f\n"
        "ld1 { v4.h }[2], [x28], #0x2\n"
        "ld1 { v13.h }[2], [x27], #0x2\n"
        "ld1 { v2.h }[2], [x21], #0x2\n"
        "ld1 { v10.h }[2], [x20], #0x2\n"
        "tbz %x[width], #0, 14f\n"
        "ld1 { v4.b }[6], [x28], #0x1\n"
        "ld1 { v13.b }[6], [x27], #0x1\n"
        "ld1 { v2.b }[6], [x21], #0x1\n"
        "ld1 { v10.b }[6], [x20], #0x1\n"
        "b 14f\n"
        "11:" // tail loop: unique 1: partial_0_4
        "tbz %x[width], #0, 14f\n"
        "ld1 { v4.b }[4], [x28], #0x1\n"
        "ld1 { v13.b }[4], [x27], #0x1\n"
        "ld1 { v2.b }[4], [x21], #0x1\n"
        "ld1 { v10.b }[4], [x20], #0x1\n"
        "b 14f\n"
        "12:" // tail loop: unique 1: partial_1_0
        "tbz %x[width], #1, 13f\n"
        "ldr h4, [x28], #0x2\n"
        "ldr h13, [x27], #0x2\n"
        "ldr h2, [x21], #0x2\n"
        "ldr h10, [x20], #0x2\n"
        "tbz %x[width], #0, 14f\n"
        "ld1 { v4.b }[2], [x28], #0x1\n"
        "ld1 { v13.b }[2], [x27], #0x1\n"
        "ld1 { v2.b }[2], [x21], #0x1\n"
        "ld1 { v10.b }[2], [x20], #0x1\n"
        "b 14f\n"
        "13:" // tail loop: unique 1: partial_0_0
        "ldr b4, [x28], #0x1\n"
        "ldr b13, [x27], #0x1\n"
        "ldr b2, [x21], #0x1\n"
        "ldr b10, [x20], #0x1\n"
        "14:" // tail loop: unique 1: Done
        "ldr w21, [%x[args_ptr], %[offsetof_in0_zeropt]]\n"
        "ushll v4.8h, v4.8b, #0x0\n"
        "ushll v3.8h, v3.8b, #0x0\n"
        "ldr w20, [%x[args_ptr], %[offsetof_in1_zeropt]]\n"
        "ushll v2.8h, v2.8b, #0x0\n"
        "ushll v11.8h, v11.8b, #0x0\n"
        "dup v16.8h, w21\n"
        "ushll v13.8h, v13.8b, #0x0\n"
        "ushll v12.8h, v12.8b, #0x0\n"
        "ushll v10.8h, v10.8b, #0x0\n"
        "ushll v9.8h, v9.8b, #0x0\n"
        "ssubl v1.4s, v4.4h, v16.4h\n"
        "ssubl2 v4.4s, v4.8h, v16.8h\n"
        "ssubl v23.4s, v3.4h, v16.4h\n"
        "ssubl2 v3.4s, v3.8h, v16.8h\n"
        "ssubl v22.4s, v2.4h, v16.4h\n"
        "ssubl2 v2.4s, v2.8h, v16.8h\n"
        "ssubl v21.4s, v11.4h, v16.4h\n"
        "ssubl2 v11.4s, v11.8h, v16.8h\n"
        "dup v20.8h, w20\n"
        "ssubl v19.4s, v13.4h, v20.4h\n"
        "ssubl2 v13.4s, v13.8h, v20.8h\n"
        "ssubl v18.4s, v12.4h, v20.4h\n"
        "ssubl2 v12.4s, v12.8h, v20.8h\n"
        "ssubl v17.4s, v10.4h, v20.4h\n"
        "ssubl2 v10.4s, v10.8h, v20.8h\n"
        "ssubl v16.4s, v9.4h, v20.4h\n"
        "ssubl2 v9.4s, v9.8h, v20.8h\n"
        "scvtf v8.4s, v1.4s\n"
        "scvtf v7.4s, v4.4s\n"
        "scvtf v6.4s, v23.4s\n"
        "scvtf v5.4s, v3.4s\n"
        "scvtf v4.4s, v22.4s\n"
        "scvtf v3.4s, v2.4s\n"
        "scvtf v2.4s, v21.4s\n"
        "scvtf v1.4s, v11.4s\n"
        "scvtf v19.4s, v19.4s\n"
        "fmul v8.4s, v8.4s, v0.s[0]\n"
        "fmla v8.4s, v19.4s, v0.s[1]\n"
        "scvtf v13.4s, v13.4s\n"
        "fmul v7.4s, v7.4s, v0.s[0]\n"
        "fmla v7.4s, v13.4s, v0.s[1]\n"
        "scvtf v18.4s, v18.4s\n"
        "fmul v6.4s, v6.4s, v0.s[0]\n"
        "fmla v6.4s, v18.4s, v0.s[1]\n"
        "scvtf v12.4s, v12.4s\n"
        "fmul v5.4s, v5.4s, v0.s[0]\n"
        "fmla v5.4s, v12.4s, v0.s[1]\n"
        "scvtf v17.4s, v17.4s\n"
        "fmul v4.4s, v4.4s, v0.s[0]\n"
        "fmla v4.4s, v17.4s, v0.s[1]\n"
        "scvtf v10.4s, v10.4s\n"
        "fmul v3.4s, v3.4s, v0.s[0]\n"
        "fmla v3.4s, v10.4s, v0.s[1]\n"
        "scvtf v16.4s, v16.4s\n"
        "fmul v2.4s, v2.4s, v0.s[0]\n"
        "fmla v2.4s, v16.4s, v0.s[1]\n"
        "scvtf v9.4s, v9.4s\n"
        "fmul v1.4s, v1.4s, v0.s[0]\n"
        "fmla v1.4s, v9.4s, v0.s[1]\n"
        "cbz %x[out_direct], 23f\n"
        "fmul v23.4s, v8.4s, v0.s[3]\n"
        "fmul v22.4s, v7.4s, v0.s[3]\n"
        "ldr w20, [%x[args_ptr], %[offsetof_out_direct_zeropt]]\n"
        "fmul v21.4s, v6.4s, v0.s[3]\n"
        "fmul v20.4s, v5.4s, v0.s[3]\n"
        "fmul v19.4s, v4.4s, v0.s[3]\n"
        "fmul v18.4s, v3.4s, v0.s[3]\n"
        "fmul v16.4s, v2.4s, v0.s[3]\n"
        "fmul v17.4s, v1.4s, v0.s[3]\n"
        "fcvtas v23.4s, v23.4s\n"
        "fcvtas v22.4s, v22.4s\n"
        "fcvtas v21.4s, v21.4s\n"
        "fcvtas v20.4s, v20.4s\n"
        "fcvtas v19.4s, v19.4s\n"
        "fcvtas v18.4s, v18.4s\n"
        "fcvtas v16.4s, v16.4s\n"
        "fcvtas v17.4s, v17.4s\n"
        "uzp1 v22.8h, v23.8h, v22.8h\n"
        "uzp1 v20.8h, v21.8h, v20.8h\n"
        "uzp1 v18.8h, v19.8h, v18.8h\n"
        "uzp1 v17.8h, v16.8h, v17.8h\n"
        "dup v16.8h, w20\n"
        "add v22.8h, v22.8h, v16.8h\n"
        "add v20.8h, v20.8h, v16.8h\n"
        "add v18.8h, v18.8h, v16.8h\n"
        "add v17.8h, v17.8h, v16.8h\n"
        "movi v16.8h, #0xff\n"
        "smin v22.8h, v22.8h, v16.8h\n"
        "smin v20.8h, v20.8h, v16.8h\n"
        "smin v18.8h, v18.8h, v16.8h\n"
        "smin v17.8h, v17.8h, v16.8h\n"
        "movi v16.8h, #0x0\n"
        "smax v22.8h, v22.8h, v16.8h\n"
        "smax v20.8h, v20.8h, v16.8h\n"
        "smax v18.8h, v18.8h, v16.8h\n"
        "smax v17.8h, v17.8h, v16.8h\n"
        "xtn v22.8b, v22.8h\n"
        "xtn v20.8b, v20.8h\n"
        "xtn v18.8b, v18.8h\n"
        "xtn v17.8b, v17.8h\n"
        "tbz %x[width], #3, 18f\n"
        "str d22, [x25, #0x0]\n"
        "add x25, x25, #0x8\n"
        "str d18, [x22, #0x0]\n"
        "add x22, x22, #0x8\n"
        "tbz %x[width], #2, 16f\n"
        "str s20, [x25], #0x4\n"
        "str s17, [x22], #0x4\n"
        "tbz %x[width], #1, 15f\n"
        "st1 { v20.h }[2], [x25], #0x2\n"
        "st1 { v17.h }[2], [x22], #0x2\n"
        "tbz %x[width], #0, 22f\n"
        "st1 { v20.b }[6], [x25], #0x1\n"
        "st1 { v17.b }[6], [x22], #0x1\n"
        "b 22f\n"
        "15:" // tail loop: Main loop: unique 2: partial_0_12
        "tbz %x[width], #0, 22f\n"
        "st1 { v20.b }[4], [x25], #0x1\n"
        "st1 { v17.b }[4], [x22], #0x1\n"
        "b 22f\n"
        "16:" // tail loop: Main loop: unique 2: partial_1_8
        "tbz %x[width], #1, 17f\n"
        "str h20, [x25], #0x2\n"
        "str h17, [x22], #0x2\n"
        "tbz %x[width], #0, 22f\n"
        "st1 { v20.b }[2], [x25], #0x1\n"
        "st1 { v17.b }[2], [x22], #0x1\n"
        "b 22f\n"
        "17:" // tail loop: Main loop: unique 2: partial_0_8
        "tbz %x[width], #0, 22f\n"
        "str b20, [x25], #0x1\n"
        "str b17, [x22], #0x1\n"
        "b 22f\n"
        "18:" // tail loop: Main loop: unique 2: partial_2_0
        "tbz %x[width], #2, 20f\n"
        "str s22, [x25], #0x4\n"
        "str s18, [x22], #0x4\n"
        "tbz %x[width], #1, 19f\n"
        "st1 { v22.h }[2], [x25], #0x2\n"
        "st1 { v18.h }[2], [x22], #0x2\n"
        "tbz %x[width], #0, 22f\n"
        "st1 { v22.b }[6], [x25], #0x1\n"
        "st1 { v18.b }[6], [x22], #0x1\n"
        "b 22f\n"
        "19:" // tail loop: Main loop: unique 2: partial_0_4
        "tbz %x[width], #0, 22f\n"
        "st1 { v22.b }[4], [x25], #0x1\n"
        "st1 { v18.b }[4], [x22], #0x1\n"
        "b 22f\n"
        "20:" // tail loop: Main loop: unique 2: partial_1_0
        "tbz %x[width], #1, 21f\n"
        "str h22, [x25], #0x2\n"
        "str h18, [x22], #0x2\n"
        "tbz %x[width], #0, 22f\n"
        "st1 { v22.b }[2], [x25], #0x1\n"
        "st1 { v18.b }[2], [x22], #0x1\n"
        "b 22f\n"
        "21:" // tail loop: Main loop: unique 2: partial_0_0
        "str b22, [x25], #0x1\n"
        "str b18, [x22], #0x1\n"
        "22:" // tail loop: Main loop: unique 2: Done
        "23:" // tail loop: Main loop: No direct output
        "mov v19.16b, v28.16b\n"
        "mov v13.16b, v29.16b\n"
        "fmla v19.4s, v8.4s, v24.4s\n"
        "ldr w22, [%x[args_ptr], %[offsetof_out_zeropt]]\n"
        "mov v18.16b, v30.16b\n"
        "mov v12.16b, v31.16b\n"
        "fmla v13.4s, v7.4s, v25.4s\n"
        "ldr w21, [%x[args_ptr], %[offsetof_maxval]]\n"
        "mov v17.16b, v28.16b\n"
        "mov v10.16b, v29.16b\n"
        "fmla v18.4s, v6.4s, v26.4s\n"
        "ldr w20, [%x[args_ptr], %[offsetof_minval]]\n"
        "mov v16.16b, v30.16b\n"
        "mov v9.16b, v31.16b\n"
        "fmla v12.4s, v5.4s, v27.4s\n"
        "fmla v17.4s, v4.4s, v24.4s\n"
        "fmla v10.4s, v3.4s, v25.4s\n"
        "fmul v8.4s, v19.4s, v0.s[2]\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "fmla v9.4s, v1.4s, v27.4s\n"
        "fmul v7.4s, v13.4s, v0.s[2]\n"
        "fmul v6.4s, v18.4s, v0.s[2]\n"
        "fmul v5.4s, v12.4s, v0.s[2]\n"
        "fmul v4.4s, v17.4s, v0.s[2]\n"
        "fmul v3.4s, v10.4s, v0.s[2]\n"
        "fmul v2.4s, v16.4s, v0.s[2]\n"
        "fmul v1.4s, v9.4s, v0.s[2]\n"
        "fcvtas v8.4s, v8.4s\n"
        "fcvtas v7.4s, v7.4s\n"
        "fcvtas v6.4s, v6.4s\n"
        "fcvtas v5.4s, v5.4s\n"
        "fcvtas v4.4s, v4.4s\n"
        "fcvtas v3.4s, v3.4s\n"
        "fcvtas v2.4s, v2.4s\n"
        "fcvtas v1.4s, v1.4s\n"
        "uzp1 v7.8h, v8.8h, v7.8h\n"
        "uzp1 v5.8h, v6.8h, v5.8h\n"
        "uzp1 v3.8h, v4.8h, v3.8h\n"
        "uzp1 v1.8h, v2.8h, v1.8h\n"
        "dup v16.8h, w22\n"
        "add v7.8h, v7.8h, v16.8h\n"
        "add v5.8h, v5.8h, v16.8h\n"
        "add v3.8h, v3.8h, v16.8h\n"
        "add v1.8h, v1.8h, v16.8h\n"
        "dup v16.8h, w21\n"
        "smin v7.8h, v7.8h, v16.8h\n"
        "smin v5.8h, v5.8h, v16.8h\n"
        "smin v3.8h, v3.8h, v16.8h\n"
        "smin v1.8h, v1.8h, v16.8h\n"
        "dup v16.8h, w20\n"
        "smax v7.8h, v7.8h, v16.8h\n"
        "smax v5.8h, v5.8h, v16.8h\n"
        "smax v3.8h, v3.8h, v16.8h\n"
        "smax v1.8h, v1.8h, v16.8h\n"
        "xtn v7.8b, v7.8h\n"
        "xtn v5.8b, v5.8h\n"
        "xtn v3.8b, v3.8h\n"
        "xtn v1.8b, v1.8h\n"
        "tbz %x[width], #3, 27f\n"
        "str d7, [x26, #0x0]\n"
        "add x26, x26, #0x8\n"
        "str d3, [x24, #0x0]\n"
        "add x24, x24, #0x8\n"
        "tbz %x[width], #2, 25f\n"
        "str s5, [x26], #0x4\n"
        "str s1, [x24], #0x4\n"
        "tbz %x[width], #1, 24f\n"
        "st1 { v5.h }[2], [x26], #0x2\n"
        "st1 { v1.h }[2], [x24], #0x2\n"
        "tbz %x[width], #0, 31f\n"
        "st1 { v5.b }[6], [x26], #0x1\n"
        "st1 { v1.b }[6], [x24], #0x1\n"
        "b 31f\n"
        "24:" // tail loop: unique 3: partial_0_12
        "tbz %x[width], #0, 31f\n"
        "st1 { v5.b }[4], [x26], #0x1\n"
        "st1 { v1.b }[4], [x24], #0x1\n"
        "b 31f\n"
        "25:" // tail loop: unique 3: partial_1_8
        "tbz %x[width], #1, 26f\n"
        "str h5, [x26], #0x2\n"
        "str h1, [x24], #0x2\n"
        "tbz %x[width], #0, 31f\n"
        "st1 { v5.b }[2], [x26], #0x1\n"
        "st1 { v1.b }[2], [x24], #0x1\n"
        "b 31f\n"
        "26:" // tail loop: unique 3: partial_0_8
        "tbz %x[width], #0, 31f\n"
        "str b5, [x26], #0x1\n"
        "str b1, [x24], #0x1\n"
        "b 31f\n"
        "27:" // tail loop: unique 3: partial_2_0
        "tbz %x[width], #2, 29f\n"
        "str s7, [x26], #0x4\n"
        "str s3, [x24], #0x4\n"
        "tbz %x[width], #1, 28f\n"
        "st1 { v7.h }[2], [x26], #0x2\n"
        "st1 { v3.h }[2], [x24], #0x2\n"
        "tbz %x[width], #0, 31f\n"
        "st1 { v7.b }[6], [x26], #0x1\n"
        "st1 { v3.b }[6], [x24], #0x1\n"
        "b 31f\n"
        "28:" // tail loop: unique 3: partial_0_4
        "tbz %x[width], #0, 31f\n"
        "st1 { v7.b }[4], [x26], #0x1\n"
        "st1 { v3.b }[4], [x24], #0x1\n"
        "b 31f\n"
        "29:" // tail loop: unique 3: partial_1_0
        "tbz %x[width], #1, 30f\n"
        "str h7, [x26], #0x2\n"
        "str h3, [x24], #0x2\n"
        "tbz %x[width], #0, 31f\n"
        "st1 { v7.b }[2], [x26], #0x1\n"
        "st1 { v3.b }[2], [x24], #0x1\n"
        "b 31f\n"
        "30:" // tail loop: unique 3: partial_0_0
        "str b7, [x26], #0x1\n"
        "str b3, [x24], #0x1\n"
        "31:" // tail loop: unique 3: Done
        "subs x23, x23, #0x2\n"
        "bgt 6b\n"
        "32:" // odd columns skip
        : [bn_add] "+&r"(bn_add), [bn_mul] "+&r"(bn_mul), [in0] "+&r"(in0), [in1] "+&r"(in1), [out] "+&r"(out), [out_direct] "+&r"(out_direct), [width] "+&r"(width)
        : [args_ptr] "r"(&ka), [height] "r"(height), [in0_stride] "r"(in0_stride), [in1_stride] "r"(in1_stride), [offsetof_in0_zeropt] "I"(offsetof(KernelArgs, in0_zeropt)), [offsetof_in1_zeropt] "I"(offsetof(KernelArgs, in1_zeropt)), [offsetof_maxval] "I"(offsetof(KernelArgs, maxval)), [offsetof_minval] "I"(offsetof(KernelArgs, minval)), [offsetof_out_direct_zeropt] "I"(offsetof(KernelArgs, out_direct_zeropt)), [offsetof_out_zeropt] "I"(offsetof(KernelArgs, out_zeropt)), [offsetof_scales] "I"(offsetof(KernelArgs, scales)), [out_direct_stride] "r"(out_direct_stride), [out_stride] "r"(out_stride)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}

} // namespace

namespace arm_compute
{
namespace cpu
{
void add_mul_add_u8_neon(const ITensor *input1, const ITensor *input2, const ITensor *bn_mul, const ITensor *bn_add,
                         ITensor *add_output, ITensor *final_output, ConvertPolicy policy, const ActivationLayerInfo &act_info, const Window &window)
{
    ARM_COMPUTE_UNUSED(policy);

    const ITensorInfo *final_output_info = final_output->info();
    const ITensorInfo *add_output_info   = (add_output != nullptr) ? add_output->info() : nullptr;
    const ITensorInfo *input1_info       = input1->info();
    const ITensorInfo *input2_info       = input2->info();

    const size_t out_stride        = final_output_info->strides_in_bytes()[1];
    const size_t out_direct_stride = (add_output != nullptr) ? add_output_info->strides_in_bytes()[1] : 0;
    const size_t in0_stride        = input1_info->strides_in_bytes()[1];
    const size_t in1_stride        = input2_info->strides_in_bytes()[1];

    uint8_t minval = std::numeric_limits<uint8_t>::lowest();
    uint8_t maxval = std::numeric_limits<uint8_t>::max();

    const UniformQuantizationInfo final_output_qinfo = final_output_info->quantization_info().uniform();
    if(act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU)
    {
        minval = quantize_qasymm8(0.f, final_output_qinfo);
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
    {
        minval = quantize_qasymm8(0.f, final_output_qinfo);
        maxval = quantize_qasymm8(act_info.a(), final_output_qinfo);
    }
    else if(act_info.activation() == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
    {
        minval = quantize_qasymm8(act_info.b(), final_output_qinfo);
        maxval = quantize_qasymm8(act_info.a(), final_output_qinfo);
    }

    const UniformQuantizationInfo in1_qinfo        = input1_info->quantization_info().uniform();
    const UniformQuantizationInfo in2_qinfo        = input2_info->quantization_info().uniform();
    const UniformQuantizationInfo add_output_qinfo = (add_output != nullptr) ? add_output_info->quantization_info().uniform() : UniformQuantizationInfo();

    const int32_t in1_offset        = in1_qinfo.offset;
    const int32_t in2_offset        = in2_qinfo.offset;
    const int32_t out_offset        = final_output_qinfo.offset;
    const int32_t out_direct_offset = add_output_qinfo.offset;

    const float in1_scale        = in1_qinfo.scale;
    const float in2_scale        = in2_qinfo.scale;
    const float out_scale        = final_output_qinfo.scale;
    const float out_direct_scale = add_output_qinfo.scale;

    const float *bn_mul_buffer = reinterpret_cast<float *>(bn_mul->buffer());
    const float *bn_add_buffer = reinterpret_cast<float *>(bn_add->buffer());

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
            a64_add_bn_clamp_direct_u8_fp32_2x16(
                reinterpret_cast<uint8_t *>(out_it.ptr()), out_stride,
                reinterpret_cast<uint8_t *>(add_out_it.ptr()), out_direct_stride,
                reinterpret_cast<uint8_t *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<uint8_t *>(in2_it.ptr()), in1_stride,
                bn_mul_buffer,
                bn_add_buffer,
                minval,
                maxval,
                out_offset, out_scale,
                out_direct_offset, out_direct_scale,
                in1_offset, in1_scale,
                in2_offset, in2_scale,
                width, height);
        },
        in1_it, in2_it, add_out_it, out_it);
    }
    else
    {
        execute_window_loop(
            win, [&](const Coordinates &)
        {
            a64_add_bn_clamp_direct_u8_fp32_2x16(
                reinterpret_cast<uint8_t *>(out_it.ptr()), out_stride,
                nullptr, out_direct_stride,
                reinterpret_cast<uint8_t *>(in1_it.ptr()), in0_stride,
                reinterpret_cast<uint8_t *>(in2_it.ptr()), in1_stride,
                bn_mul_buffer,
                bn_add_buffer,
                minval,
                maxval,
                out_offset, out_scale,
                out_direct_offset, out_direct_scale,
                in1_offset, in1_scale,
                in2_offset, in2_scale,
                width, height);
        },
        in1_it, in2_it, out_it);
    }
}
} // namespace cpu
} // namespace arm_compute

#endif // __aarch64__
