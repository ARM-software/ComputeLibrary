//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_bf16p1x4_f32_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_lhs_quant_pack_bf16p1x4_f32_neon(size_t mr) {
    KAI_ASSUME(mr == kai_mr);
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_bf16p1x4_f32_neon(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_bf16p1x4_f32_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(sr);
    KAI_ASSUME(m_idx == 0);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    return m_idx * kai_roundup(k, kr) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(sr);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    return kai_roundup(m, mr) * kai_roundup(k, kr) * sizeof(uint16_t);
}

void kai_run_lhs_quant_pack_bf16p1x4_f32_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_UNUSED(sr);
    KAI_UNUSED(lhs_stride);

    KAI_ASSUME(m == 1);
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    KAI_ASSUME(m_idx_start == 0);

    const float* lhs_ptr = lhs;
    uint16_t* lhs_packed_ptr = lhs_packed;

    // Unroll two 256-bit loops
    size_t i = 0;
    for (; i + 16 <= k; i += 16) {
        const float32x4x4_t val = vld1q_f32_x4(lhs_ptr);
        bfloat16x8x2_t bf_val;

        bf_val.val[0] = vcvtq_low_bf16_f32(val.val[0]);
        bf_val.val[0] = vcvtq_high_bf16_f32(bf_val.val[0], val.val[1]);
        bf_val.val[1] = vcvtq_low_bf16_f32(val.val[2]);
        bf_val.val[1] = vcvtq_high_bf16_f32(bf_val.val[1], val.val[3]);
        vst1q_bf16_x2((bfloat16_t*)(lhs_packed_ptr), bf_val);

        lhs_ptr += 16;
        lhs_packed_ptr += 16;
    }

    // 1 load + 1 convert + 1 store
    for (; i + 8 <= k; i += 8) {
        const float32x4x2_t f32_val = vld1q_f32_x2(lhs_ptr);
        bfloat16x8_t bf_val = vcvtq_low_bf16_f32(f32_val.val[0]);
        bf_val = vcvtq_high_bf16_f32(bf_val, f32_val.val[1]);
        vst1q_bf16((bfloat16_t*)(lhs_packed_ptr), bf_val);

        lhs_ptr += 8;
        lhs_packed_ptr += 8;
    }

    for (; i + 4 <= k; i += 4) {
        const float32x4_t f32_val = vld1q_f32(lhs_ptr);
        bfloat16x4_t bf_val = vcvt_bf16_f32(f32_val);
        vst1_bf16((bfloat16_t*)(lhs_packed_ptr), bf_val);

        lhs_ptr += 4;
        lhs_packed_ptr += 4;
    }

    for (; i < k; ++i) {
        *lhs_packed_ptr = kai_cast_bf16_f32(*lhs_ptr);

        ++lhs_ptr;
        ++lhs_packed_ptr;
    }

    // Zero pad
    const size_t rounded_up_k = kai_roundup(k, kr);
    for (; i < rounded_up_k; ++i) {
        *lhs_packed_ptr = 0;
        ++lhs_packed_ptr;
    }
}

#endif  // Architectural features check.
