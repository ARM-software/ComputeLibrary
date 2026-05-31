//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16, FEAT_FP16.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_bf16p12x4biasf16_f16_neon.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 12;
static const size_t kai_kr = 4;
static const size_t kai_num_bytes_input = 2;
static const size_t kai_num_bytes_output = 2;
static const size_t kai_num_bytes_bias = 2;

size_t kai_get_n_step_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(void) {
    return kai_nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(size_t k) {
    return kai_nr * (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx * (kai_num_bytes_bias + kai_roundup(k, kai_kr) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(size_t n, size_t k) {
    return kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(kai_roundup(n, kai_nr), k);
}

void kai_run_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_nr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    const uint16_t* pad_row = rhs;

    // Fill zeros if bias is nullptr
    size_t bias_step = nr * sizeof(uint16_t);
    uint8_t zero_bias[kai_nr * sizeof(uint16_t)];

    if (bias == NULL) {
        memset(zero_bias, 0, kai_nr * sizeof(uint16_t));
        bias_step = 0;
    }

    const void* bias_ptr = bias == NULL ? (const void*)zero_bias : bias;

    size_t out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(height);

    __asm__ __volatile__(
        "mov x22, %x[width]\n"
        "mov x21, %x[out]\n"
        "cmp x22, #0xc\n"
        "blt 2f\n"
        "1:"  // Bias: Full loop
        "ldr q17, [%x[bias], #0x0]\n"
        "ldr d16, [%x[bias], #0x10]\n"
        "sub x22, x22, #0xc\n"
        "add %x[bias], %x[bias], %x[bias_step]\n"
        "cmp x22, #0xc\n"
        "str q17, [x21, #0x0]\n"
        "str d16, [x21, #0x10]\n"
        "add x21, x21, %x[out_stride]\n"
        "bge 1b\n"
        "cbz x22, 3f\n"
        "2:"  // Bias: Tail loop
        "ldr h20, [%x[bias], #0x0]\n"
        "sub x22, x22, #0x1\n"
        "add %x[bias], %x[bias], #0x2\n"
        "cmp x22, #0x0\n"
        "str h20, [x21]\n"
        "add x21, x21, #0x2\n"
        "bgt 2b\n"
        "3:"  // Bias: Done
        "cmp %x[height], #0x8\n"
        "add %x[out], %x[out], #0x18\n"
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
        "ldr q23, [x9], #0x10\n"
        "ldr q22, [x26], #0x10\n"
        "sub x28, x28, #0xc\n"
        "ldr q17, [x25], #0x10\n"
        "ldr q16, [x24], #0x10\n"
        "cmp x28, #0xc\n"
        "ldr q1, [x23], #0x10\n"
        "ldr q0, [x22], #0x10\n"
        "ldr q21, [x21], #0x10\n"
        "ldr q20, [x20], #0x10\n"
        "ldr d31, [x9], #0x8\n"
        "ldr d30, [x26], #0x8\n"
        "zip1 v29.8h, v23.8h, v17.8h\n"
        "zip1 v28.8h, v22.8h, v16.8h\n"
        "ldr d19, [x25], #0x8\n"
        "ldr d18, [x24], #0x8\n"
        "zip2 v27.8h, v23.8h, v17.8h\n"
        "zip2 v26.8h, v22.8h, v16.8h\n"
        "ldr d25, [x23], #0x8\n"
        "ldr d24, [x22], #0x8\n"
        "zip1 v23.8h, v1.8h, v21.8h\n"
        "zip1 v22.8h, v0.8h, v20.8h\n"
        "ldr d17, [x21], #0x8\n"
        "ldr d16, [x20], #0x8\n"
        "zip2 v21.8h, v1.8h, v21.8h\n"
        "zip2 v20.8h, v0.8h, v20.8h\n"
        "zip1 v19.8h, v31.8h, v19.8h\n"
        "zip1 v18.8h, v30.8h, v18.8h\n"
        "zip1 v1.8h, v29.8h, v28.8h\n"
        "zip2 v0.8h, v29.8h, v28.8h\n"
        "zip1 v17.8h, v25.8h, v17.8h\n"
        "zip1 v16.8h, v24.8h, v16.8h\n"
        "zip1 v31.8h, v27.8h, v26.8h\n"
        "zip2 v30.8h, v27.8h, v26.8h\n"
        "zip1 v29.8h, v19.8h, v18.8h\n"
        "zip2 v28.8h, v19.8h, v18.8h\n"
        "zip1 v13.8h, v23.8h, v22.8h\n"
        "zip2 v12.8h, v23.8h, v22.8h\n"
        "zip1 v11.8h, v21.8h, v20.8h\n"
        "zip2 v10.8h, v21.8h, v20.8h\n"
        "zip1 v9.8h, v17.8h, v16.8h\n"
        "zip2 v8.8h, v17.8h, v16.8h\n"
        "fcvtl v27.4s, v1.4h\n"
        "fcvtl v26.4s, v0.4h\n"
        "fcvtl v25.4s, v31.4h\n"
        "fcvtl v24.4s, v30.4h\n"
        "fcvtl v23.4s, v29.4h\n"
        "fcvtl v22.4s, v28.4h\n"
        "fcvtl v21.4s, v13.4h\n"
        "fcvtl v20.4s, v12.4h\n"
        "fcvtl v19.4s, v11.4h\n"
        "fcvtl v18.4s, v10.4h\n"
        "fcvtl v17.4s, v9.4h\n"
        "fcvtl v16.4s, v8.4h\n"
        "fcvtl2 v7.4s, v1.8h\n"
        ".inst 0x0ea16b66  // bfcvtn v6.4h, v27.4s\n"
        "fcvtl2 v5.4s, v0.8h\n"
        ".inst 0x0ea16b44  // bfcvtn v4.4h, v26.4s\n"
        "fcvtl2 v3.4s, v31.8h\n"
        ".inst 0x0ea16b22  // bfcvtn v2.4h, v25.4s\n"
        "fcvtl2 v1.4s, v30.8h\n"
        ".inst 0x0ea16b00  // bfcvtn v0.4h, v24.4s\n"
        "fcvtl2 v31.4s, v29.8h\n"
        ".inst 0x0ea16afe  // bfcvtn v30.4h, v23.4s\n"
        "fcvtl2 v29.4s, v28.8h\n"
        ".inst 0x0ea16adc  // bfcvtn v28.4h, v22.4s\n"
        "fcvtl2 v27.4s, v13.8h\n"
        ".inst 0x0ea16aba  // bfcvtn v26.4h, v21.4s\n"
        "fcvtl2 v25.4s, v12.8h\n"
        ".inst 0x0ea16a98  // bfcvtn v24.4h, v20.4s\n"
        "fcvtl2 v23.4s, v11.8h\n"
        ".inst 0x0ea16a76  // bfcvtn v22.4h, v19.4s\n"
        "fcvtl2 v21.4s, v10.8h\n"
        ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
        "fcvtl2 v19.4s, v9.8h\n"
        ".inst 0x0ea16a32  // bfcvtn v18.4h, v17.4s\n"
        "fcvtl2 v17.4s, v8.8h\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea168e6  // bfcvtn2 v6.8h, v7.4s\n"
        ".inst 0x4ea168a4  // bfcvtn2 v4.8h, v5.4s\n"
        ".inst 0x4ea16862  // bfcvtn2 v2.8h, v3.4s\n"
        ".inst 0x4ea16820  // bfcvtn2 v0.8h, v1.4s\n"
        ".inst 0x4ea16bfe  // bfcvtn2 v30.8h, v31.4s\n"
        ".inst 0x4ea16bbc  // bfcvtn2 v28.8h, v29.4s\n"
        ".inst 0x4ea16b7a  // bfcvtn2 v26.8h, v27.4s\n"
        ".inst 0x4ea16b38  // bfcvtn2 v24.8h, v25.4s\n"
        "str q6, [x27, #0x0]\n"
        ".inst 0x4ea16af6  // bfcvtn2 v22.8h, v23.4s\n"
        ".inst 0x4ea16ab4  // bfcvtn2 v20.8h, v21.4s\n"
        "str q4, [x27, #0x10]\n"
        ".inst 0x4ea16a72  // bfcvtn2 v18.8h, v19.4s\n"
        ".inst 0x4ea16a30  // bfcvtn2 v16.8h, v17.4s\n"
        "str q2, [x27, #0x20]\n"
        "str q0, [x27, #0x30]\n"
        "str q30, [x27, #0x40]\n"
        "str q28, [x27, #0x50]\n"
        "str q26, [x27, #0x60]\n"
        "str q24, [x27, #0x70]\n"
        "str q22, [x27, #0x80]\n"
        "str q20, [x27, #0x90]\n"
        "str q18, [x27, #0xa0]\n"
        "str q16, [x27, #0xb0]\n"
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
        "ldr d23, [x9], #0x8\n"
        "ldr d22, [x26], #0x8\n"
        "sub x28, x28, #0x4\n"
        "ldr d20, [x25], #0x8\n"
        "ldr d16, [x24], #0x8\n"
        "cmp x28, #0x4\n"
        "ldr d19, [x23], #0x8\n"
        "ldr d21, [x22], #0x8\n"
        "ldr d18, [x21], #0x8\n"
        "ldr d17, [x20], #0x8\n"
        "zip1 v20.8h, v23.8h, v20.8h\n"
        "zip1 v16.8h, v22.8h, v16.8h\n"
        "zip1 v19.8h, v19.8h, v18.8h\n"
        "zip1 v18.8h, v21.8h, v17.8h\n"
        "zip1 v17.8h, v20.8h, v16.8h\n"
        "zip2 v16.8h, v20.8h, v16.8h\n"
        "zip1 v25.8h, v19.8h, v18.8h\n"
        "zip2 v24.8h, v19.8h, v18.8h\n"
        "fcvtl v19.4s, v17.4h\n"
        "fcvtl v18.4s, v16.4h\n"
        "fcvtl2 v23.4s, v17.8h\n"
        "fcvtl2 v22.4s, v16.8h\n"
        "fcvtl v17.4s, v25.4h\n"
        "fcvtl v16.4s, v24.4h\n"
        ".inst 0x0ea16a75  // bfcvtn v21.4h, v19.4s\n"
        ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
        "fcvtl2 v19.4s, v25.8h\n"
        "fcvtl2 v18.4s, v24.8h\n"
        ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea16af5  // bfcvtn2 v21.8h, v23.4s\n"
        ".inst 0x4ea16ad4  // bfcvtn2 v20.8h, v22.4s\n"
        ".inst 0x4ea16a71  // bfcvtn2 v17.8h, v19.4s\n"
        ".inst 0x4ea16a50  // bfcvtn2 v16.8h, v18.4s\n"
        "str q21, [x27, #0x0]\n"
        "str q20, [x27, #0x10]\n"
        "str q17, [x27, #0x60]\n"
        "str q16, [x27, #0x70]\n"
        "add x27, x27, #0x20\n"
        "bge 7b\n"
        "8:"  // Main row loop: width 4 loop: skip
        "cmp x28, #0x1\n"
        "blt 10f\n"
        "9:"  // Main row loop: width 1 loop: loop
        "ldr h23, [x9], #0x2\n"
        "ldr h22, [x26], #0x2\n"
        "sub x28, x28, #0x1\n"
        "ldr h19, [x25], #0x2\n"
        "ldr h18, [x24], #0x2\n"
        "cmp x28, #0x1\n"
        "ldr h21, [x23], #0x2\n"
        "ldr h20, [x22], #0x2\n"
        "ldr h17, [x21], #0x2\n"
        "ldr h16, [x20], #0x2\n"
        "zip1 v19.8h, v23.8h, v19.8h\n"
        "zip1 v18.8h, v22.8h, v18.8h\n"
        "zip1 v17.8h, v21.8h, v17.8h\n"
        "zip1 v16.8h, v20.8h, v16.8h\n"
        "zip1 v19.8h, v19.8h, v18.8h\n"
        "zip1 v18.8h, v17.8h, v16.8h\n"
        "fcvtl v17.4s, v19.4h\n"
        "fcvtl v16.4s, v18.4h\n"
        "fcvtl2 v19.4s, v19.8h\n"
        "fcvtl2 v18.4s, v18.8h\n"
        ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea16a71  // bfcvtn2 v17.8h, v19.4s\n"
        ".inst 0x4ea16a50  // bfcvtn2 v16.8h, v18.4s\n"
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
        "ldr q25, [x9], #0x10\n"
        "ldr q24, [x26], #0x10\n"
        "sub x20, x20, #0xc\n"
        "ldr q19, [x25], #0x10\n"
        "ldr q18, [x24], #0x10\n"
        "cmp x20, #0xc\n"
        "ldr d23, [x9], #0x8\n"
        "ldr d22, [x26], #0x8\n"
        "ldr d17, [x25], #0x8\n"
        "ldr d16, [x24], #0x8\n"
        "zip1 v21.8h, v25.8h, v19.8h\n"
        "zip1 v20.8h, v24.8h, v18.8h\n"
        "zip2 v19.8h, v25.8h, v19.8h\n"
        "zip2 v18.8h, v24.8h, v18.8h\n"
        "zip1 v17.8h, v23.8h, v17.8h\n"
        "zip1 v16.8h, v22.8h, v16.8h\n"
        "zip1 v24.8h, v21.8h, v20.8h\n"
        "zip2 v23.8h, v21.8h, v20.8h\n"
        "zip1 v22.8h, v19.8h, v18.8h\n"
        "zip2 v30.8h, v19.8h, v18.8h\n"
        "zip1 v29.8h, v17.8h, v16.8h\n"
        "zip2 v28.8h, v17.8h, v16.8h\n"
        "fcvtl v21.4s, v24.4h\n"
        "fcvtl v20.4s, v23.4h\n"
        "fcvtl v19.4s, v22.4h\n"
        "fcvtl v18.4s, v30.4h\n"
        "fcvtl v17.4s, v29.4h\n"
        "fcvtl v16.4s, v28.4h\n"
        "fcvtl2 v27.4s, v24.8h\n"
        ".inst 0x0ea16aba  // bfcvtn v26.4h, v21.4s\n"
        "fcvtl2 v25.4s, v23.8h\n"
        ".inst 0x0ea16a98  // bfcvtn v24.4h, v20.4s\n"
        "fcvtl2 v23.4s, v22.8h\n"
        ".inst 0x0ea16a76  // bfcvtn v22.4h, v19.4s\n"
        "fcvtl2 v21.4s, v30.8h\n"
        ".inst 0x0ea16a54  // bfcvtn v20.4h, v18.4s\n"
        "fcvtl2 v19.4s, v29.8h\n"
        ".inst 0x0ea16a32  // bfcvtn v18.4h, v17.4s\n"
        "fcvtl2 v17.4s, v28.8h\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea16b7a  // bfcvtn2 v26.8h, v27.4s\n"
        ".inst 0x4ea16b38  // bfcvtn2 v24.8h, v25.4s\n"
        ".inst 0x4ea16af6  // bfcvtn2 v22.8h, v23.4s\n"
        ".inst 0x4ea16ab4  // bfcvtn2 v20.8h, v21.4s\n"
        ".inst 0x4ea16a72  // bfcvtn2 v18.8h, v19.4s\n"
        ".inst 0x4ea16a30  // bfcvtn2 v16.8h, v17.4s\n"
        "str q26, [x27, #0x0]\n"
        "str q24, [x27, #0x10]\n"
        "str q22, [x27, #0x20]\n"
        "str q20, [x27, #0x30]\n"
        "str q18, [x27, #0x40]\n"
        "str q16, [x27, #0x50]\n"
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
        "ldr d18, [x9], #0x8\n"
        "ldr d19, [x26], #0x8\n"
        "sub x20, x20, #0x4\n"
        "ldr d17, [x25], #0x8\n"
        "ldr d16, [x24], #0x8\n"
        "cmp x20, #0x4\n"
        "zip1 v18.8h, v18.8h, v17.8h\n"
        "zip1 v17.8h, v19.8h, v16.8h\n"
        "zip1 v16.8h, v18.8h, v17.8h\n"
        "zip2 v20.8h, v18.8h, v17.8h\n"
        "fcvtl v17.4s, v16.4h\n"
        "fcvtl2 v19.4s, v16.8h\n"
        "fcvtl v16.4s, v20.4h\n"
        ".inst 0x0ea16a32  // bfcvtn v18.4h, v17.4s\n"
        "fcvtl2 v17.4s, v20.8h\n"
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
        "ldr h19, [x9], #0x2\n"
        "ldr h18, [x26], #0x2\n"
        "sub x20, x20, #0x1\n"
        "ldr h17, [x25], #0x2\n"
        "ldr h16, [x24], #0x2\n"
        "cmp x20, #0x1\n"
        "zip1 v17.8h, v19.8h, v17.8h\n"
        "zip1 v16.8h, v18.8h, v16.8h\n"
        "zip1 v17.8h, v17.8h, v16.8h\n"
        "fcvtl v16.4s, v17.4h\n"
        "fcvtl2 v17.4s, v17.8h\n"
        ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
        ".inst 0x4ea16a30  // bfcvtn2 v16.8h, v17.4s\n"
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
        : "cc", "memory", "v0", "v1", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v2", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v3", "v30", "v31", "v4", "v5", "v6", "v7", "v8", "v9",
          "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9");
}

#endif  // Architectural features check.
