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

#include "kai_lhs_pack_bf16p8x4_f16_neon.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 8;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_lhs_pack_bf16p8x4_f16_neon(size_t mr) {
    KAI_ASSUME(mr == kai_mr);

    return kai_mr;
}

size_t kai_get_lhs_offset_lhs_pack_bf16p8x4_f16_neon(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % (kai_mr) == 0);

    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    return m_idx * kai_roundup(k, kai_kr) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    return kai_roundup(m, kai_mr) * kai_roundup(k, kai_kr) * sizeof(uint16_t);
}

void kai_run_lhs_pack_bf16p8x4_f16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    KAI_ASSUME(m_idx_start == 0);

    const size_t block_height = kai_mr;
    const size_t row_offset = 0;

    const void* in[kai_mr];

    for (size_t block_y = 0; block_y < m; block_y += block_height) {
        const size_t height = KAI_MIN(m - block_y, block_height);
        void* out = (char*)lhs_packed + block_y * kai_roundup(k, kai_kr) * sizeof(uint16_t);
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
            "add x28, x28, %x[row_offset], LSL #1\n"
            "add x27, x27, %x[row_offset], LSL #1\n"
            "add x26, x26, %x[row_offset], LSL #1\n"
            "add x25, x25, %x[row_offset], LSL #1\n"
            "add x24, x24, %x[row_offset], LSL #1\n"
            "add x23, x23, %x[row_offset], LSL #1\n"
            "add x22, x22, %x[row_offset], LSL #1\n"
            "add x21, x21, %x[row_offset], LSL #1\n"
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
            "cmp %x[width], #0x8\n"
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
            "subs %x[width], %x[width], #0x8\n"
            "ldr q17, [x24], #0x10\n"
            "ldr q16, [x22], #0x10\n"
            "cmp %x[width], #0x8\n"
            "ldr q25, [x27], #0x10\n"
            "ldr q24, [x25], #0x10\n"
            "ldr q1, [x23], #0x10\n"
            "ldr q0, [x21], #0x10\n"
            "fcvtl v23.4s, v19.4h\n"
            "fcvtl2 v22.4s, v19.8h\n"
            "fcvtl v21.4s, v18.4h\n"
            "fcvtl2 v20.4s, v18.8h\n"
            "prfm pldl1keep, [x28, #0x70]\n"
            "fcvtl v19.4s, v17.4h\n"
            "fcvtl2 v18.4s, v17.8h\n"
            "prfm pldl1keep, [x27, #0x70]\n"
            "prfm pldl1keep, [x26, #0x70]\n"
            "fcvtl v17.4s, v16.4h\n"
            "fcvtl2 v16.4s, v16.8h\n"
            "prfm pldl1keep, [x25, #0x70]\n"
            "prfm pldl1keep, [x24, #0x70]\n"
            ".inst 0x0ea16aff  // bfcvtn v31.4h, v23.4s\n"
            ".inst 0x0ea16ade  // bfcvtn v30.4h, v22.4s\n"
            "prfm pldl1keep, [x23, #0x70]\n"
            "prfm pldl1keep, [x22, #0x70]\n"
            "fcvtl v29.4s, v25.4h\n"
            "fcvtl2 v28.4s, v25.8h\n"
            "prfm pldl1keep, [x21, #0x70]\n"
            ".inst 0x0ea16abb  // bfcvtn v27.4h, v21.4s\n"
            ".inst 0x0ea16a9a  // bfcvtn v26.4h, v20.4s\n"
            "fcvtl v25.4s, v24.4h\n"
            "fcvtl2 v24.4s, v24.8h\n"
            ".inst 0x0ea16a77  // bfcvtn v23.4h, v19.4s\n"
            ".inst 0x0ea16a56  // bfcvtn v22.4h, v18.4s\n"
            "fcvtl v21.4s, v1.4h\n"
            "fcvtl2 v20.4s, v1.8h\n"
            ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
            ".inst 0x0ea16a12  // bfcvtn v18.4h, v16.4s\n"
            "fcvtl v17.4s, v0.4h\n"
            "fcvtl2 v16.4s, v0.8h\n"
            ".inst 0x4ea16bbf  // bfcvtn2 v31.8h, v29.4s\n"
            ".inst 0x4ea16b9e  // bfcvtn2 v30.8h, v28.4s\n"
            ".inst 0x4ea16b3b  // bfcvtn2 v27.8h, v25.4s\n"
            ".inst 0x4ea16b1a  // bfcvtn2 v26.8h, v24.4s\n"
            ".inst 0x4ea16ab7  // bfcvtn2 v23.8h, v21.4s\n"
            ".inst 0x4ea16a96  // bfcvtn2 v22.8h, v20.4s\n"
            ".inst 0x4ea16a33  // bfcvtn2 v19.8h, v17.4s\n"
            ".inst 0x4ea16a12  // bfcvtn2 v18.8h, v16.4s\n"
            "str q31, [%x[out_ptr], #0x0]\n"
            "str q27, [%x[out_ptr], #0x10]\n"
            "str q23, [%x[out_ptr], #0x20]\n"
            "str q19, [%x[out_ptr], #0x30]\n"
            "str q30, [%x[out_ptr], #0x40]\n"
            "str q26, [%x[out_ptr], #0x50]\n"
            "str q22, [%x[out_ptr], #0x60]\n"
            "str q18, [%x[out_ptr], #0x70]\n"
            "add %x[out_ptr], %x[out_ptr], #0x80\n"
            "bge 2b\n"
            "3:"  // Main loop skip
            "cbz %x[width], 8f\n"
            "tbz %x[width], #2, 5f\n"
            "ldr d19, [x28], #0x8\n"
            "ldr d25, [x27], #0x8\n"
            "ldr d18, [x26], #0x8\n"
            "ldr d24, [x25], #0x8\n"
            "ldr d17, [x24], #0x8\n"
            "ldr d1, [x23], #0x8\n"
            "ldr d16, [x22], #0x8\n"
            "ldr d0, [x21], #0x8\n"
            "tbz %x[width], #1, 4f\n"
            "ld1 { v19.s }[2], [x28], #0x4\n"
            "ld1 { v25.s }[2], [x27], #0x4\n"
            "mov x20, #0x2\n"
            "ld1 { v18.s }[2], [x26], #0x4\n"
            "ld1 { v24.s }[2], [x25], #0x4\n"
            "ld1 { v17.s }[2], [x24], #0x4\n"
            "ld1 { v1.s }[2], [x23], #0x4\n"
            "ld1 { v16.s }[2], [x22], #0x4\n"
            "ld1 { v0.s }[2], [x21], #0x4\n"
            "tbz %x[width], #0, 7f\n"
            "ld1 { v19.h }[6], [x28]\n"
            "ld1 { v25.h }[6], [x27]\n"
            "ld1 { v18.h }[6], [x26]\n"
            "ld1 { v24.h }[6], [x25]\n"
            "ld1 { v17.h }[6], [x24]\n"
            "ld1 { v1.h }[6], [x23]\n"
            "ld1 { v16.h }[6], [x22]\n"
            "ld1 { v0.h }[6], [x21]\n"
            "b 7f\n"
            "4:"  // odd_loads_1_4
            "mov x20, #0x1\n"
            "tbz %x[width], #0, 7f\n"
            "ld1 { v19.h }[4], [x28]\n"
            "ld1 { v25.h }[4], [x27]\n"
            "mov x20, #0x2\n"
            "ld1 { v18.h }[4], [x26]\n"
            "ld1 { v24.h }[4], [x25]\n"
            "ld1 { v17.h }[4], [x24]\n"
            "ld1 { v1.h }[4], [x23]\n"
            "ld1 { v16.h }[4], [x22]\n"
            "ld1 { v0.h }[4], [x21]\n"
            "b 7f\n"
            "5:"  // odd_loads_2_0
            "tbz %x[width], #1, 6f\n"
            "ldr s19, [x28], #0x4\n"
            "ldr s25, [x27], #0x4\n"
            "mov x20, #0x1\n"
            "ldr s18, [x26], #0x4\n"
            "ldr s24, [x25], #0x4\n"
            "ldr s17, [x24], #0x4\n"
            "ldr s1, [x23], #0x4\n"
            "ldr s16, [x22], #0x4\n"
            "ldr s0, [x21], #0x4\n"
            "tbz %x[width], #0, 7f\n"
            "ld1 { v19.h }[2], [x28]\n"
            "ld1 { v25.h }[2], [x27]\n"
            "ld1 { v18.h }[2], [x26]\n"
            "ld1 { v24.h }[2], [x25]\n"
            "ld1 { v17.h }[2], [x24]\n"
            "ld1 { v1.h }[2], [x23]\n"
            "ld1 { v16.h }[2], [x22]\n"
            "ld1 { v0.h }[2], [x21]\n"
            "b 7f\n"
            "6:"  // odd_loads_1_0
            "ldr h19, [x28, #0x0]\n"
            "ldr h25, [x27, #0x0]\n"
            "mov x20, #0x1\n"
            "ldr h18, [x26, #0x0]\n"
            "ldr h24, [x25, #0x0]\n"
            "ldr h17, [x24, #0x0]\n"
            "ldr h1, [x23, #0x0]\n"
            "ldr h16, [x22, #0x0]\n"
            "ldr h0, [x21, #0x0]\n"
            "7:"  // Odd load end
            "fcvtl v23.4s, v19.4h\n"
            "fcvtl2 v22.4s, v19.8h\n"
            "subs x20, x20, #0x1\n"
            "fcvtl v21.4s, v18.4h\n"
            "fcvtl2 v20.4s, v18.8h\n"
            "fcvtl v19.4s, v17.4h\n"
            "fcvtl2 v18.4s, v17.8h\n"
            "fcvtl v17.4s, v16.4h\n"
            "fcvtl2 v16.4s, v16.8h\n"
            ".inst 0x0ea16aff  // bfcvtn v31.4h, v23.4s\n"
            ".inst 0x0ea16ade  // bfcvtn v30.4h, v22.4s\n"
            "fcvtl v29.4s, v25.4h\n"
            "fcvtl2 v28.4s, v25.8h\n"
            ".inst 0x0ea16abb  // bfcvtn v27.4h, v21.4s\n"
            ".inst 0x0ea16a9a  // bfcvtn v26.4h, v20.4s\n"
            "fcvtl v25.4s, v24.4h\n"
            "fcvtl2 v24.4s, v24.8h\n"
            ".inst 0x0ea16a77  // bfcvtn v23.4h, v19.4s\n"
            ".inst 0x0ea16a56  // bfcvtn v22.4h, v18.4s\n"
            "fcvtl v21.4s, v1.4h\n"
            "fcvtl2 v20.4s, v1.8h\n"
            ".inst 0x0ea16a33  // bfcvtn v19.4h, v17.4s\n"
            ".inst 0x0ea16a12  // bfcvtn v18.4h, v16.4s\n"
            "fcvtl v17.4s, v0.4h\n"
            "fcvtl2 v16.4s, v0.8h\n"
            ".inst 0x4ea16bbf  // bfcvtn2 v31.8h, v29.4s\n"
            ".inst 0x4ea16b9e  // bfcvtn2 v30.8h, v28.4s\n"
            ".inst 0x4ea16b3b  // bfcvtn2 v27.8h, v25.4s\n"
            ".inst 0x4ea16b1a  // bfcvtn2 v26.8h, v24.4s\n"
            ".inst 0x4ea16ab7  // bfcvtn2 v23.8h, v21.4s\n"
            ".inst 0x4ea16a96  // bfcvtn2 v22.8h, v20.4s\n"
            ".inst 0x4ea16a33  // bfcvtn2 v19.8h, v17.4s\n"
            ".inst 0x4ea16a12  // bfcvtn2 v18.8h, v16.4s\n"
            "str q31, [%x[out_ptr], #0x0]\n"
            "str q27, [%x[out_ptr], #0x10]\n"
            "str q23, [%x[out_ptr], #0x20]\n"
            "str q19, [%x[out_ptr], #0x30]\n"
            "add %x[out_ptr], %x[out_ptr], #0x40\n"
            "beq 8f\n"
            "str q30, [%x[out_ptr], #0x0]\n"
            "str q26, [%x[out_ptr], #0x10]\n"
            "str q22, [%x[out_ptr], #0x20]\n"
            "str q18, [%x[out_ptr], #0x30]\n"
            "add %x[out_ptr], %x[out_ptr], #0x40\n"
            "8:"  // Odds skip
            : [out_ptr] "+&r"(out), [width] "+&r"(width)
            : [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset)
            : "cc", "memory", "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
              "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
    }
}

#endif  // Architectural features check.
