//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_lhs_pack_x16p2vlx2_x16_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    MR = 2,
    KR = 2,
    MAX_M_STEP = MR * (KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(uint16_t)) / KR,
    SR = 1,
};

void kai_kernel_lhs_pack_x16p2vlx2_x16_sme(size_t height, size_t width, const void* in, void* out);

static size_t kai_get_mr_lhs_pack_x16p2vlx2_x16_sme(void) {
    return MR * kai_get_sme_vector_length_u16() / KR;
}

size_t kai_get_m_step_lhs_pack_x16p2vlx2_x16_sme(size_t mr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x16p2vlx2_x16_sme());
    KAI_UNUSED(mr);
    return kai_get_mr_lhs_pack_x16p2vlx2_x16_sme();
}

size_t kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme(size_t m_idx, size_t lhs_stride_row) {
    KAI_ASSUME(m_idx % kai_get_mr_lhs_pack_x16p2vlx2_x16_sme() == 0);

    return m_idx * lhs_stride_row;
}

size_t kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_get_m_step_lhs_pack_x16p2vlx2_x16_sme(mr) == 0);
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x16p2vlx2_x16_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return m_idx * kai_roundup(k, KR) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x16p2vlx2_x16_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    return kai_roundup(m, kai_get_mr_lhs_pack_x16p2vlx2_x16_sme()) * kai_roundup(k, KR) * sizeof(uint16_t);
}

void kai_run_lhs_pack_x16p2vlx2_x16_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride_row,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x16p2vlx2_x16_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);
    KAI_ASSUME(m_idx_start == 0);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    const size_t m_step = kai_get_mr_lhs_pack_x16p2vlx2_x16_sme();
    const size_t width = k;

    KAI_ASSERT(m_step <= MAX_M_STEP);
    const uint8_t* in[MAX_M_STEP];

    uint8_t* out_base = lhs_packed;
    const uint8_t* lhs_ptr = lhs;

    kai_commit_za();

    for (size_t i_m = 0; i_m < m; i_m += m_step) {
        const size_t height = KAI_MIN(m - i_m, m_step);
        void* out = out_base;
        out_base += m_step * kai_roundup(k, KR) * sizeof(uint16_t);

        for (size_t y = 0; y < height; y++) {
            in[y] = lhs_ptr + (i_m + y) * lhs_stride_row;
        }

        kai_kernel_lhs_pack_x16p2vlx2_x16_sme(
            height, width, in, out);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }
}

#endif  // Architectural features check.
