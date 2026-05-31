//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else  // Architectural features check.

#define MAX_MR 8

#include "kai_lhs_quant_pack_bf16p8x4_f32_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 8;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_lhs_quant_pack_bf16p8x4_f32_neon(size_t mr) {
    KAI_ASSUME(mr == kai_mr);
    return kai_mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_bf16p8x4_f32_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_bf16p8x4_f32_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(sr);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(m_idx % mr == 0);

    return m_idx * kai_roundup(k, kr) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(sr);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    return kai_roundup(m, mr) * kai_roundup(k, kr) * sizeof(uint16_t);
}

void kai_run_lhs_quant_pack_bf16p8x4_f32_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(kr == kai_kr);
    KAI_UNUSED(sr);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    KAI_ASSUME(m_idx_start == 0);
    KAI_ASSUME(mr <= MAX_MR);

    const size_t block_height = mr;
    const size_t row_offset = 0;

    const void* in[MAX_MR];

    for (size_t block_y = 0; block_y < m; block_y += block_height) {
        const size_t height = KAI_MIN(m - block_y, block_height);
        void* out = (char*)lhs_packed + block_y * kai_roundup(k, kr) * sizeof(uint16_t);
        size_t width = k;

        for (size_t y = 0; y < height; y++) {
            in[y] = (const char*)lhs + (block_y + y) * lhs_stride;
        }

        __asm__ __volatile__(
            "ldr x28, [%x[in], #0x0]\n"
            "ldr x27, [%x[in], #0x8]\n"
            "cmp %x[height], #0x8\n"
            "ldr x26, [%x[in], #0x10]\n"
            "ldr x25, [%x[in], #0x18]\n"
            "ldr x24, [%x[in], #0x20]\n"
            "ldr x23, [%x[in], #0x28]\n"
            "ldr x22, [%x[in], #0x30]\n"
            "ldr x21, [%x[in], #0x38]\n"
            "add x28, x28, %x[row_offset], LSL #2\n"
            "add x27, x27, %x[row_offset], LSL #2\n"
            "add x26, x26, %x[row_offset], LSL #2\n"
            "add x25, x25, %x[row_offset], LSL #2\n"
            "add x24, x24, %x[row_offset], LSL #2\n"
            "add x23, x23, %x[row_offset], LSL #2\n"
            "add x22, x22, %x[row_offset], LSL #2\n"
            "add x21, x21, %x[row_offset], LSL #2\n"
            "beq 1f\n"
            "cmp %x[height], #0x2\n"
            "mov x21, x28\n"
            "csel x27, x27, x28, GE\n"
            "csel x26, x26, x28, GT\n"
            "cmp %x[height], #0x4\n"
            "csel x25, x25, x28, GE\n"
            "csel x24, x24, x28, GT\n"
            "cmp %x[height], #0x6\n"
            "csel x23, x23, x28, GE\n"
            "csel x22, x22, x28, GT\n"
            "1:"  // no_pointer_adj
            "cmp %x[width], #0x4\n"
            "prfm pldl1keep, [x28, #0x0]\n"
            "prfm pldl1keep, [x27, #0x0]\n"
            "prfm pldl1keep, [x26, #0x0]\n"
            "prfm pldl1keep, [x25, #0x0]\n"
            "prfm pldl1keep, [x24, #0x0]\n"
            "prfm pldl1keep, [x23, #0x0]\n"
            "prfm pldl1keep, [x22, #0x0]\n"
            "prfm pldl1keep, [x21, #0x0]\n"
            "prfm pldl1keep, [x28, #0x40]\n"
            "prfm pldl1keep, [x27, #0x40]\n"
            "prfm pldl1keep, [x26, #0x40]\n"
            "prfm pldl1keep, [x25, #0x40]\n"
            "prfm pldl1keep, [x24, #0x40]\n"
            "prfm pldl1keep, [x23, #0x40]\n"
            "prfm pldl1keep, [x22, #0x40]\n"
            "prfm pldl1keep, [x21, #0x40]\n"
            "blt 3f\n"
            "2:"  // Main loop head
            "ldr q19, [x28], #0x10\n"
            "ldr q18, [x26], #0x10\n"
            "subs %x[width], %x[width], #0x4\n"
            "ldr q17, [x24], #0x10\n"
            "ldr q16, [x22], #0x10\n"
            "cmp %x[width], #0x4\n"
            "ldr q23, [x27], #0x10\n"
            "ldr q22, [x25], #0x10\n"
            "ldr q21, [x23], #0x10\n"
            "ldr q20, [x21], #0x10\n"
            ".inst 0x0ea16a73  // bfcvtn v19.4h, v19.4s\n"
            ".inst 0x0ea16a52  // bfcvtn v18.4h, v18.4s\n"
            ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
            ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
            "prfm pldl1keep, [x28, #0x70]\n"
            "prfm pldl1keep, [x27, #0x70]\n"
            "prfm pldl1keep, [x26, #0x70]\n"
            "prfm pldl1keep, [x25, #0x70]\n"
            "prfm pldl1keep, [x24, #0x70]\n"
            "prfm pldl1keep, [x23, #0x70]\n"
            ".inst 0x4ea16af3  // bfcvtn2 v19.8h, v23.4s\n"
            ".inst 0x4ea16ad2  // bfcvtn2 v18.8h, v22.4s\n"
            "prfm pldl1keep, [x22, #0x70]\n"
            "prfm pldl1keep, [x21, #0x70]\n"
            ".inst 0x4ea16ab1  // bfcvtn2 v17.8h, v21.4s\n"
            ".inst 0x4ea16a90  // bfcvtn2 v16.8h, v20.4s\n"
            "str q19, [%x[out_ptr], #0x0]\n"
            "str q18, [%x[out_ptr], #0x10]\n"
            "str q17, [%x[out_ptr], #0x20]\n"
            "str q16, [%x[out_ptr], #0x30]\n"
            "add %x[out_ptr], %x[out_ptr], #0x40\n"
            "bge 2b\n"
            "3:"  // Main loop skip
            "cbz %x[width], 6f\n"
            "tbz %x[width], #1, 4f\n"
            "ldr d19, [x28], #0x8\n"
            "ldr d23, [x27], #0x8\n"
            "mov x20, #0x1\n"
            "ldr d18, [x26], #0x8\n"
            "ldr d22, [x25], #0x8\n"
            "ldr d17, [x24], #0x8\n"
            "ldr d21, [x23], #0x8\n"
            "ldr d16, [x22], #0x8\n"
            "ldr d20, [x21], #0x8\n"
            "tbz %x[width], #0, 5f\n"
            "ld1 { v19.s }[2], [x28]\n"
            "ld1 { v23.s }[2], [x27]\n"
            "ld1 { v18.s }[2], [x26]\n"
            "ld1 { v22.s }[2], [x25]\n"
            "ld1 { v17.s }[2], [x24]\n"
            "ld1 { v21.s }[2], [x23]\n"
            "ld1 { v16.s }[2], [x22]\n"
            "ld1 { v20.s }[2], [x21]\n"
            "b 5f\n"
            "4:"  // odd_loads_1_0
            "ldr s19, [x28, #0x0]\n"
            "ldr s23, [x27, #0x0]\n"
            "mov x20, #0x1\n"
            "ldr s18, [x26, #0x0]\n"
            "ldr s22, [x25, #0x0]\n"
            "ldr s17, [x24, #0x0]\n"
            "ldr s21, [x23, #0x0]\n"
            "ldr s16, [x22, #0x0]\n"
            "ldr s20, [x21, #0x0]\n"
            "5:"  // Odd load end
            ".inst 0x0ea16a73  // bfcvtn v19.4h, v19.4s\n"
            ".inst 0x0ea16a52  // bfcvtn v18.4h, v18.4s\n"
            ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
            ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
            ".inst 0x4ea16af3  // bfcvtn2 v19.8h, v23.4s\n"
            ".inst 0x4ea16ad2  // bfcvtn2 v18.8h, v22.4s\n"
            ".inst 0x4ea16ab1  // bfcvtn2 v17.8h, v21.4s\n"
            ".inst 0x4ea16a90  // bfcvtn2 v16.8h, v20.4s\n"
            "str q19, [%x[out_ptr], #0x0]\n"
            "str q18, [%x[out_ptr], #0x10]\n"
            "str q17, [%x[out_ptr], #0x20]\n"
            "str q16, [%x[out_ptr], #0x30]\n"
            "add %x[out_ptr], %x[out_ptr], #0x40\n"
            "6:"  // Odds skip
            : [out_ptr] "+&r"(out), [width] "+&r"(width)
            : [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset)
            : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x20", "x21", "x22", "x23", "x24",
              "x25", "x26", "x27", "x28");
    }
}

#endif  // Architectural features check.
