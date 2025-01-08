//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_nr = 16;
static const size_t kai_kr = 1;

size_t kai_get_n_step_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(void) {
    return kai_nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx * sizeof(uint16_t);
}

size_t kai_get_bias_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(size_t n_idx) {
    return n_idx * sizeof(uint16_t);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_nr == 0);

    return n_idx * (sizeof(uint16_t) + k * sizeof(uint16_t));
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(size_t n, size_t k) {
    return kai_get_rhs_packed_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(kai_roundup(n, kai_nr), k);
}

void kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_nr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    size_t height = k;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_stride;
    size_t out_stride = kai_nr * height * sizeof(uint16_t) + kai_nr * sizeof(uint16_t);

    __asm__ __volatile__(
        "mov x22, %x[width]\n"
        "mov x21, %x[out]\n"
        "cmp x22, #0x10\n"
        "blt 2f\n"
        "1:"  // Bias: Full loop
        "ldr q17, [%x[bias], #0x0]\n"
        "ldr q16, [%x[bias], #0x10]\n"
        "sub x22, x22, #0x10\n"
        "add %x[bias], %x[bias], #0x20\n"
        "cmp x22, #0x10\n"
        "str q17, [x21, #0x0]\n"
        "str q16, [x21, #0x10]\n"
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
        "cmp %x[height], #0x4\n"
        "add %x[out], %x[out], #0x20\n"
        "blt 12f\n"
        "4:"  // Main row loop: Head
        "mov x25, %x[in]\n"
        "mov x24, %x[width]\n"
        "mov x23, %x[out]\n"
        "sub %x[height], %x[height], #0x4\n"
        "add x22, x25, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "cmp x24, #0x10\n"
        "add %x[in], x20, %x[in_stride]\n"
        "blt 6f\n"
        "5:"  // Main row loop: Column loop
        "ldr q23, [x25], #0x10\n"
        "ldr q22, [x22], #0x10\n"
        "sub x24, x24, #0x10\n"
        "ldr q21, [x21], #0x10\n"
        "ldr q20, [x20], #0x10\n"
        "cmp x24, #0x10\n"
        "ldr q19, [x25], #0x10\n"
        "ldr q18, [x22], #0x10\n"
        "ldr q17, [x21], #0x10\n"
        "ldr q16, [x20], #0x10\n"
        "str q23, [x23, #0x0]\n"
        "str q19, [x23, #0x10]\n"
        "str q22, [x23, #0x20]\n"
        "str q18, [x23, #0x30]\n"
        "str q21, [x23, #0x40]\n"
        "str q17, [x23, #0x50]\n"
        "str q20, [x23, #0x60]\n"
        "str q16, [x23, #0x70]\n"
        "add x23, x23, %x[out_stride]\n"
        "bge 5b\n"
        "6:"  // Main row loop: Column loop skip
        "cbz x24, 11f\n"
        "cmp x24, #0x4\n"
        "movi v16.8h, #0x0\n"
        "str q16, [x23, #0x0]\n"
        "str q16, [x23, #0x10]\n"
        "str q16, [x23, #0x20]\n"
        "str q16, [x23, #0x30]\n"
        "str q16, [x23, #0x40]\n"
        "str q16, [x23, #0x50]\n"
        "str q16, [x23, #0x60]\n"
        "str q16, [x23, #0x70]\n"
        "blt 8f\n"
        "7:"  // Main row loop: width 4 loop: loop
        "ldr d19, [x25], #0x8\n"
        "ldr d18, [x22], #0x8\n"
        "sub x24, x24, #0x4\n"
        "ldr d17, [x21], #0x8\n"
        "ldr d16, [x20], #0x8\n"
        "cmp x24, #0x4\n"
        "str d19, [x23, #0x0]\n"
        "str d18, [x23, #0x20]\n"
        "str d17, [x23, #0x40]\n"
        "str d16, [x23, #0x60]\n"
        "add x23, x23, #0x8\n"
        "bge 7b\n"
        "8:"  // Main row loop: width 4 loop: skip
        "cmp x24, #0x1\n"
        "blt 10f\n"
        "9:"  // Main row loop: width 1 loop: loop
        "ldr h19, [x25], #0x2\n"
        "ldr h18, [x22], #0x2\n"
        "sub x24, x24, #0x1\n"
        "ldr h17, [x21], #0x2\n"
        "ldr h16, [x20], #0x2\n"
        "cmp x24, #0x1\n"
        "str h19, [x23, #0x0]\n"
        "str h18, [x23, #0x20]\n"
        "str h17, [x23, #0x40]\n"
        "str h16, [x23, #0x60]\n"
        "add x23, x23, #0x2\n"
        "bge 9b\n"
        "10:"  // Main row loop: width 1 loop: skip
        "11:"  // Main row loop: odd col skip
        "cmp %x[height], #0x4\n"
        "add %x[out], %x[out], #0x80\n"
        "bge 4b\n"
        "cbz %x[height], 21f\n"
        "12:"  // Main loop skip
        "13:"  // Tail row loop: Head
        "mov x20, %x[width]\n"
        "mov x25, %x[in]\n"
        "mov x23, %x[out]\n"
        "sub %x[height], %x[height], #0x1\n"
        "cmp x20, #0x10\n"
        "add %x[in], x25, %x[in_stride]\n"
        "blt 15f\n"
        "14:"  // Tail row loop: Column loop
        "ldr q17, [x25], #0x10\n"
        "sub x20, x20, #0x10\n"
        "ldr q16, [x25], #0x10\n"
        "cmp x20, #0x10\n"
        "str q17, [x23, #0x0]\n"
        "str q16, [x23, #0x10]\n"
        "add x23, x23, %x[out_stride]\n"
        "bge 14b\n"
        "15:"  // Tail row loop: Column loop skip
        "cbz x20, 20f\n"
        "cmp x20, #0x4\n"
        "movi v16.8h, #0x0\n"
        "str q16, [x23, #0x0]\n"
        "str q16, [x23, #0x10]\n"
        "blt 17f\n"
        "16:"  // Tail row loop: width 4 loop: loop
        "ldr d16, [x25], #0x8\n"
        "sub x20, x20, #0x4\n"
        "cmp x20, #0x4\n"
        "str d16, [x23, #0x0]\n"
        "add x23, x23, #0x8\n"
        "bge 16b\n"
        "17:"  // Tail row loop: width 4 loop: skip
        "cmp x20, #0x1\n"
        "blt 19f\n"
        "18:"  // Tail row loop: width 1 loop: loop
        "ldr h16, [x25], #0x2\n"
        "sub x20, x20, #0x1\n"
        "cmp x20, #0x1\n"
        "str h16, [x23, #0x0]\n"
        "add x23, x23, #0x2\n"
        "bge 18b\n"
        "19:"  // Tail row loop: width 1 loop: skip
        "20:"  // Tail row loop: odd col skip
        "cmp %x[height], #0x1\n"
        "add %x[out], %x[out], #0x20\n"
        "bge 13b\n"
        "21:"  // Done
        : [bias] "+&r"(bias), [height] "+&r"(height), [in] "+&r"(in), [out] "+&r"(out)
        : [in_stride] "r"(in_stride), [out_stride] "r"(out_stride), [width] "r"(width)
        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x20", "x21", "x22", "x23", "x24",
          "x25");
}

#endif  // Architectural features check.
