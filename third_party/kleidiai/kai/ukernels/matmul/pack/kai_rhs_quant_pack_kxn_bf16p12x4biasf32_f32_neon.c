//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else  // Architectural features check.

#define MAX_NR 12

#include "kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 12;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_n_step_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(void) {
    return kai_nr;
}

size_t kai_get_rhs_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_nr == 0);
    return n_idx * sizeof(float);
}

size_t kai_get_bias_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_nr == 0);
    return n_idx * sizeof(uint32_t);
}

size_t kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr) {
    KAI_ASSUME(n_idx % nr == 0);
    KAI_ASSUME(kai_nr == nr);
    KAI_ASSUME(kai_kr == kr);

    return n_idx * (sizeof(uint32_t) + kai_roundup(k, kr) * sizeof(uint16_t));
}

size_t kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(size_t n, size_t k, size_t nr, size_t kr) {
    return kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(kai_roundup(n, nr), k, nr, kr);
}

void kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(kai_nr == nr);
    KAI_ASSUME(kai_kr == kr);
    KAI_ASSUME(kai_sr == sr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);
    KAI_ASSUME(nr <= MAX_NR);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    const float* pad_row = rhs;

    // Fill zeros if bias is nullptr
    size_t bias_step = nr * sizeof(float);
    uint8_t zero_bias[MAX_NR * sizeof(float)];

    if (bias == NULL) {
        memset(zero_bias, 0, MAX_NR * sizeof(float));
        bias_step = 0;
    }

    const void* bias_ptr = bias == NULL ? (const void*)zero_bias : bias;

    const size_t out_stride = nr * kai_roundup(height, kr) * sizeof(uint16_t) + nr * sizeof(uint32_t);

    __asm__ __volatile__(
        "mov x22, %x[width]\n"
        "mov x21, %x[out]\n"
        "cmp x22, #0xc\n"
        "blt 2f\n"
        "1:"  // Bias: Full loop
        "ldr q16, [%x[bias], #0x0]\n"
        "ldr q26, [%x[bias], #0x10]\n"
        "sub x22, x22, #0xc\n"
        "ldr q8, [%x[bias], #0x20]\n"
        "cmp x22, #0xc\n"
        "add %x[bias], %x[bias], %x[bias_step]\n"
        "str q16, [x21, #0x0]\n"
        "str q26, [x21, #0x10]\n"
        "str q8, [x21, #0x20]\n"
        "add x21, x21, %x[out_stride]\n"
        "bge 1b\n"
        "cbz x22, 3f\n"
        "2:"  // Bias: Tail loop
        "ldr w20, [%x[bias], #0x0]\n"
        "sub x22, x22, #0x1\n"
        "add %x[bias], %x[bias], #0x4\n"
        "cmp x22, #0x0\n"
        "str w20, [x21]\n"
        "add x21, x21, #0x4\n"
        "bgt 2b\n"
        "3:"  // Bias: Done
        "cmp %x[height], #0x8\n"
        "add %x[out], %x[out], #0x30\n"
        "blt 12f\n"
        "4:"  // Main row loop: Head
        "mov x9, %x[in]\n"
        "mov x28, %x[width]\n"
        "mov x27, %x[out]\n"
        "sub %x[height], %x[height], #0x8\n"
        "add x26, x9, %x[in_stride]\n"
        "add x25, x26, %x[in_stride]\n"
        "add x24, x25, %x[in_stride]\n"
        "cmp x28, #0xc\n"
        "add x23, x24, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "add %x[in], x20, %x[in_stride]\n"
        "blt 6f\n"
        "5:"  // Main row loop: Column loop
        "ldr q28, [x9], #0x10\n"
        "ldr q27, [x26], #0x10\n"
        "sub x28, x28, #0xc\n"
        "ldr q11, [x25], #0x10\n"
        "ldr q5, [x24], #0x10\n"
        "cmp x28, #0xc\n"
        "ldr q14, [x23], #0x10\n"
        "ldr q6, [x22], #0x10\n"
        "ldr q2, [x21], #0x10\n"
        "ldr q18, [x20], #0x10\n"
        "ldr q1, [x9], #0x10\n"
        "ldr q7, [x26], #0x10\n"
        "zip1 v15.4s, v28.4s, v11.4s\n"
        "zip1 v8.4s, v27.4s, v5.4s\n"
        "ldr q3, [x25], #0x10\n"
        "ldr q23, [x24], #0x10\n"
        "zip2 v17.4s, v28.4s, v11.4s\n"
        "zip2 v27.4s, v27.4s, v5.4s\n"
        "ldr q5, [x23], #0x10\n"
        "ldr q30, [x22], #0x10\n"
        "zip1 v26.4s, v14.4s, v2.4s\n"
        "zip1 v31.4s, v6.4s, v18.4s\n"
        "ldr q20, [x21], #0x10\n"
        "ldr q16, [x20], #0x10\n"
        "zip2 v12.4s, v14.4s, v2.4s\n"
        "zip2 v24.4s, v6.4s, v18.4s\n"
        "ldr q29, [x9], #0x10\n"
        "ldr q6, [x26], #0x10\n"
        "zip1 v18.4s, v1.4s, v3.4s\n"
        "zip1 v4.4s, v7.4s, v23.4s\n"
        "ldr q22, [x25], #0x10\n"
        "ldr q0, [x24], #0x10\n"
        "zip2 v3.4s, v1.4s, v3.4s\n"
        "zip2 v1.4s, v7.4s, v23.4s\n"
        "ldr q2, [x23], #0x10\n"
        "ldr q10, [x22], #0x10\n"
        "zip1 v28.4s, v5.4s, v20.4s\n"
        "zip1 v14.4s, v30.4s, v16.4s\n"
        "ldr q9, [x21], #0x10\n"
        "ldr q23, [x20], #0x10\n"
        "zip2 v13.4s, v5.4s, v20.4s\n"
        "zip2 v30.4s, v30.4s, v16.4s\n"
        "zip1 v16.4s, v29.4s, v22.4s\n"
        "zip1 v5.4s, v6.4s, v0.4s\n"
        "zip2 v22.4s, v29.4s, v22.4s\n"
        "zip2 v0.4s, v6.4s, v0.4s\n"
        "zip1 v7.4s, v2.4s, v9.4s\n"
        "zip1 v19.4s, v10.4s, v23.4s\n"
        "zip2 v21.4s, v2.4s, v9.4s\n"
        "zip2 v25.4s, v10.4s, v23.4s\n"
        "zip1 v11.4s, v15.4s, v8.4s\n"
        "zip1 v9.4s, v17.4s, v27.4s\n"
        "zip1 v6.4s, v18.4s, v4.4s\n"
        "zip1 v2.4s, v3.4s, v1.4s\n"
        "zip1 v29.4s, v16.4s, v5.4s\n"
        "zip1 v20.4s, v22.4s, v0.4s\n"
        "zip1 v10.4s, v26.4s, v31.4s\n"
        "zip1 v23.4s, v12.4s, v24.4s\n"
        ".inst 0x0ea1696b  // bfcvtn v11.4h, v11.4s\n"
        "zip2 v8.4s, v15.4s, v8.4s\n"
        "zip1 v15.4s, v28.4s, v14.4s\n"
        ".inst 0x0ea16929  // bfcvtn v9.4h, v9.4s\n"
        "zip2 v27.4s, v17.4s, v27.4s\n"
        "zip1 v17.4s, v13.4s, v30.4s\n"
        ".inst 0x0ea168c6  // bfcvtn v6.4h, v6.4s\n"
        "zip2 v4.4s, v18.4s, v4.4s\n"
        "zip1 v18.4s, v7.4s, v19.4s\n"
        ".inst 0x0ea16842  // bfcvtn v2.4h, v2.4s\n"
        "zip2 v1.4s, v3.4s, v1.4s\n"
        "zip1 v3.4s, v21.4s, v25.4s\n"
        ".inst 0x0ea16bbd  // bfcvtn v29.4h, v29.4s\n"
        "zip2 v5.4s, v16.4s, v5.4s\n"
        ".inst 0x0ea16a94  // bfcvtn v20.4h, v20.4s\n"
        "zip2 v0.4s, v22.4s, v0.4s\n"
        ".inst 0x0ea16956  // bfcvtn v22.4h, v10.4s\n"
        "zip2 v31.4s, v26.4s, v31.4s\n"
        ".inst 0x0ea16aea  // bfcvtn v10.4h, v23.4s\n"
        "zip2 v26.4s, v12.4s, v24.4s\n"
        ".inst 0x0ea169ef  // bfcvtn v15.4h, v15.4s\n"
        "zip2 v12.4s, v28.4s, v14.4s\n"
        ".inst 0x0ea16a2e  // bfcvtn v14.4h, v17.4s\n"
        "zip2 v24.4s, v13.4s, v30.4s\n"
        ".inst 0x0ea16a57  // bfcvtn v23.4h, v18.4s\n"
        "zip2 v18.4s, v7.4s, v19.4s\n"
        ".inst 0x0ea16871  // bfcvtn v17.4h, v3.4s\n"
        "zip2 v16.4s, v21.4s, v25.4s\n"
        ".inst 0x4ea1690b  // bfcvtn2 v11.8h, v8.4s\n"
        ".inst 0x4ea16b69  // bfcvtn2 v9.8h, v27.4s\n"
        ".inst 0x4ea16886  // bfcvtn2 v6.8h, v4.4s\n"
        ".inst 0x4ea16822  // bfcvtn2 v2.8h, v1.4s\n"
        ".inst 0x4ea168bd  // bfcvtn2 v29.8h, v5.4s\n"
        ".inst 0x4ea16814  // bfcvtn2 v20.8h, v0.4s\n"
        ".inst 0x4ea16bf6  // bfcvtn2 v22.8h, v31.4s\n"
        ".inst 0x4ea16b4a  // bfcvtn2 v10.8h, v26.4s\n"
        "str q11, [x27, #0x0]\n"
        ".inst 0x4ea1698f  // bfcvtn2 v15.8h, v12.4s\n"
        ".inst 0x4ea16b0e  // bfcvtn2 v14.8h, v24.4s\n"
        "str q9, [x27, #0x10]\n"
        ".inst 0x4ea16a57  // bfcvtn2 v23.8h, v18.4s\n"
        ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
        "str q6, [x27, #0x20]\n"
        "str q2, [x27, #0x30]\n"
        "str q29, [x27, #0x40]\n"
        "str q20, [x27, #0x50]\n"
        "str q22, [x27, #0x60]\n"
        "str q10, [x27, #0x70]\n"
        "str q15, [x27, #0x80]\n"
        "str q14, [x27, #0x90]\n"
        "str q23, [x27, #0xa0]\n"
        "str q17, [x27, #0xb0]\n"
        "add x27, x27, %x[out_stride]\n"
        "bge 5b\n"
        "6:"  // Main row loop: Column loop skip
        "cbz x28, 11f\n"
        "cmp x28, #0x4\n"
        "movi v16.16b, #0x0\n"
        "str q16, [x27, #0x0]\n"
        "str q16, [x27, #0x10]\n"
        "str q16, [x27, #0x20]\n"
        "str q16, [x27, #0x30]\n"
        "str q16, [x27, #0x40]\n"
        "str q16, [x27, #0x50]\n"
        "str q16, [x27, #0x60]\n"
        "str q16, [x27, #0x70]\n"
        "str q16, [x27, #0x80]\n"
        "str q16, [x27, #0x90]\n"
        "str q16, [x27, #0xa0]\n"
        "str q16, [x27, #0xb0]\n"
        "blt 8f\n"
        "7:"  // Main row loop: width 4 loop: loop
        "ldr q25, [x9], #0x10\n"
        "ldr q24, [x26], #0x10\n"
        "sub x28, x28, #0x4\n"
        "ldr q21, [x25], #0x10\n"
        "ldr q20, [x24], #0x10\n"
        "cmp x28, #0x4\n"
        "ldr q23, [x23], #0x10\n"
        "ldr q19, [x22], #0x10\n"
        "ldr q18, [x21], #0x10\n"
        "ldr q17, [x20], #0x10\n"
        "zip1 v22.4s, v25.4s, v21.4s\n"
        "zip1 v16.4s, v24.4s, v20.4s\n"
        "zip2 v21.4s, v25.4s, v21.4s\n"
        "zip2 v20.4s, v24.4s, v20.4s\n"
        "zip1 v27.4s, v23.4s, v18.4s\n"
        "zip1 v26.4s, v19.4s, v17.4s\n"
        "zip2 v25.4s, v23.4s, v18.4s\n"
        "zip2 v24.4s, v19.4s, v17.4s\n"
        "zip1 v19.4s, v22.4s, v16.4s\n"
        "zip1 v18.4s, v21.4s, v20.4s\n"
        "zip1 v17.4s, v27.4s, v26.4s\n"
        "zip2 v23.4s, v22.4s, v16.4s\n"
        "zip1 v16.4s, v25.4s, v24.4s\n"
        "zip2 v22.4s, v21.4s, v20.4s\n"
        ".inst 0x0ea16a75  // bfcvtn v21.4h, v19.4s\n"
        ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
        ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
        "zip2 v18.4s, v27.4s, v26.4s\n"
        ".inst 0x0ea16a11  // bfcvtn v17.4h, v16.4s\n"
        "zip2 v16.4s, v25.4s, v24.4s\n"
        ".inst 0x4ea16af5  // bfcvtn2 v21.8h, v23.4s\n"
        ".inst 0x4ea16ad4  // bfcvtn2 v20.8h, v22.4s\n"
        ".inst 0x4ea16a53  // bfcvtn2 v19.8h, v18.4s\n"
        ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
        "str q21, [x27, #0x0]\n"
        "str q20, [x27, #0x10]\n"
        "str q19, [x27, #0x60]\n"
        "str q17, [x27, #0x70]\n"
        "add x27, x27, #0x20\n"
        "bge 7b\n"
        "8:"  // Main row loop: width 4 loop: skip
        "cmp x28, #0x1\n"
        "blt 10f\n"
        "9:"  // Main row loop: width 1 loop: loop
        "ldr s23, [x9], #0x4\n"
        "ldr s22, [x26], #0x4\n"
        "sub x28, x28, #0x1\n"
        "ldr s19, [x25], #0x4\n"
        "ldr s17, [x24], #0x4\n"
        "cmp x28, #0x1\n"
        "ldr s21, [x23], #0x4\n"
        "ldr s20, [x22], #0x4\n"
        "ldr s18, [x21], #0x4\n"
        "ldr s16, [x20], #0x4\n"
        "zip1 v19.4s, v23.4s, v19.4s\n"
        "zip1 v17.4s, v22.4s, v17.4s\n"
        "zip1 v18.4s, v21.4s, v18.4s\n"
        "zip1 v16.4s, v20.4s, v16.4s\n"
        "zip1 v17.4s, v19.4s, v17.4s\n"
        "zip1 v16.4s, v18.4s, v16.4s\n"
        ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        "str d17, [x27, #0x0]\n"
        "str d16, [x27, #0x60]\n"
        "add x27, x27, #0x8\n"
        "bge 9b\n"
        "10:"  // Main row loop: width 1 loop: skip
        "11:"  // Main row loop: odd col skip
        "cmp %x[height], #0x8\n"
        "add %x[out], %x[out], #0xc0\n"
        "bge 4b\n"
        "cbz %x[height], 21f\n"
        "12:"  // Main loop skip
        "13:"  // Tail row loop: Head
        "mov x9, %x[in]\n"
        "mov x20, %x[width]\n"
        "cmp %x[height], #0x3\n"
        "mov x27, %x[out]\n"
        "add x26, x9, %x[in_stride]\n"
        "add x25, x26, %x[in_stride]\n"
        "add x24, x25, %x[in_stride]\n"
        "csel x25, x25, %x[pad_row], GE\n"
        "add %x[in], x24, %x[in_stride]\n"
        "csel x24, x24, %x[pad_row], GT\n"
        "cmp %x[height], #0x1\n"
        "sub %x[height], %x[height], #0x4\n"
        "csel x26, x26, %x[pad_row], GT\n"
        "cmp x20, #0xc\n"
        "blt 15f\n"
        "14:"  // Tail row loop: Column loop
        "ldr q24, [x9], #0x10\n"
        "ldr q23, [x26], #0x10\n"
        "sub x20, x20, #0xc\n"
        "ldr q22, [x25], #0x10\n"
        "ldr q16, [x24], #0x10\n"
        "cmp x20, #0xc\n"
        "ldr q28, [x9], #0x10\n"
        "ldr q27, [x26], #0x10\n"
        "ldr q21, [x25], #0x10\n"
        "ldr q20, [x24], #0x10\n"
        "ldr q19, [x9], #0x10\n"
        "zip1 v26.4s, v24.4s, v22.4s\n"
        "zip1 v25.4s, v23.4s, v16.4s\n"
        "ldr q18, [x26], #0x10\n"
        "ldr q17, [x25], #0x10\n"
        "zip2 v24.4s, v24.4s, v22.4s\n"
        "zip2 v23.4s, v23.4s, v16.4s\n"
        "ldr q16, [x24], #0x10\n"
        "zip1 v2.4s, v28.4s, v21.4s\n"
        "zip1 v22.4s, v27.4s, v20.4s\n"
        "zip2 v1.4s, v28.4s, v21.4s\n"
        "zip2 v0.4s, v27.4s, v20.4s\n"
        "zip1 v31.4s, v19.4s, v17.4s\n"
        "zip1 v30.4s, v18.4s, v16.4s\n"
        "zip2 v29.4s, v19.4s, v17.4s\n"
        "zip2 v28.4s, v18.4s, v16.4s\n"
        "zip1 v21.4s, v26.4s, v25.4s\n"
        "zip1 v20.4s, v24.4s, v23.4s\n"
        "zip1 v19.4s, v2.4s, v22.4s\n"
        "zip1 v18.4s, v1.4s, v0.4s\n"
        "zip1 v17.4s, v31.4s, v30.4s\n"
        "zip1 v16.4s, v29.4s, v28.4s\n"
        ".inst 0x0ea16abb  // bfcvtn v27.4h, v21.4s\n"
        "zip2 v26.4s, v26.4s, v25.4s\n"
        ".inst 0x0ea16a99  // bfcvtn v25.4h, v20.4s\n"
        "zip2 v24.4s, v24.4s, v23.4s\n"
        ".inst 0x0ea16a77  // bfcvtn v23.4h, v19.4s\n"
        "zip2 v22.4s, v2.4s, v22.4s\n"
        ".inst 0x0ea16a55  // bfcvtn v21.4h, v18.4s\n"
        "zip2 v20.4s, v1.4s, v0.4s\n"
        ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
        "zip2 v18.4s, v31.4s, v30.4s\n"
        ".inst 0x0ea16a11  // bfcvtn v17.4h, v16.4s\n"
        "zip2 v16.4s, v29.4s, v28.4s\n"
        ".inst 0x4ea16b5b  // bfcvtn2 v27.8h, v26.4s\n"
        ".inst 0x4ea16b19  // bfcvtn2 v25.8h, v24.4s\n"
        ".inst 0x4ea16ad7  // bfcvtn2 v23.8h, v22.4s\n"
        ".inst 0x4ea16a95  // bfcvtn2 v21.8h, v20.4s\n"
        ".inst 0x4ea16a53  // bfcvtn2 v19.8h, v18.4s\n"
        ".inst 0x4ea16a11  // bfcvtn2 v17.8h, v16.4s\n"
        "str q27, [x27, #0x0]\n"
        "str q25, [x27, #0x10]\n"
        "str q23, [x27, #0x20]\n"
        "str q21, [x27, #0x30]\n"
        "str q19, [x27, #0x40]\n"
        "str q17, [x27, #0x50]\n"
        "add x27, x27, %x[out_stride]\n"
        "bge 14b\n"
        "15:"  // Tail row loop: Column loop skip
        "cbz x20, 20f\n"
        "cmp x20, #0x4\n"
        "movi v16.16b, #0x0\n"
        "str q16, [x27, #0x0]\n"
        "str q16, [x27, #0x10]\n"
        "str q16, [x27, #0x20]\n"
        "str q16, [x27, #0x30]\n"
        "str q16, [x27, #0x40]\n"
        "str q16, [x27, #0x50]\n"
        "blt 17f\n"
        "16:"  // Tail row loop: width 4 loop: loop
        "ldr q21, [x9], #0x10\n"
        "ldr q20, [x26], #0x10\n"
        "sub x20, x20, #0x4\n"
        "ldr q19, [x25], #0x10\n"
        "ldr q17, [x24], #0x10\n"
        "cmp x20, #0x4\n"
        "zip1 v18.4s, v21.4s, v19.4s\n"
        "zip1 v16.4s, v20.4s, v17.4s\n"
        "zip2 v21.4s, v21.4s, v19.4s\n"
        "zip2 v20.4s, v20.4s, v17.4s\n"
        "zip1 v17.4s, v18.4s, v16.4s\n"
        "zip2 v19.4s, v18.4s, v16.4s\n"
        "zip1 v16.4s, v21.4s, v20.4s\n"
        ".inst 0x0ea16a32  // bfcvtn v18.4h, v17.4s\n"
        "zip2 v17.4s, v21.4s, v20.4s\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea16a72  // bfcvtn2 v18.8h, v19.4s\n"
        ".inst 0x4ea16a30  // bfcvtn2 v16.8h, v17.4s\n"
        "str q18, [x27, #0x0]\n"
        "str q16, [x27, #0x10]\n"
        "add x27, x27, #0x20\n"
        "bge 16b\n"
        "17:"  // Tail row loop: width 4 loop: skip
        "cmp x20, #0x1\n"
        "blt 19f\n"
        "18:"  // Tail row loop: width 1 loop: loop
        "ldr s19, [x9], #0x4\n"
        "ldr s18, [x26], #0x4\n"
        "sub x20, x20, #0x1\n"
        "ldr s17, [x25], #0x4\n"
        "ldr s16, [x24], #0x4\n"
        "cmp x20, #0x1\n"
        "zip1 v17.4s, v19.4s, v17.4s\n"
        "zip1 v16.4s, v18.4s, v16.4s\n"
        "zip1 v16.4s, v17.4s, v16.4s\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        "str d16, [x27, #0x0]\n"
        "add x27, x27, #0x8\n"
        "bge 18b\n"
        "19:"  // Tail row loop: width 1 loop: skip
        "20:"  // Tail row loop: odd col skip
        "cmp %x[height], #0x1\n"
        "add %x[out], %x[out], #0x60\n"
        "bge 13b\n"
        "21:"  // Done
        : [bias] "+&r"(bias_ptr), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out)
        : [bias_step] "r"(bias_step), [in_stride] "r"(in_stride), [out_stride] "r"(out_stride), [pad_row] "r"(pad_row),
          [width] "r"(width)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}

#endif  // Architectural features check.
