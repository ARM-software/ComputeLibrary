//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    MR = 2,
    KR = 4,
    MAX_M_STEP = MR * (KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(int8_t)) / KR,
};

void kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme(size_t height, size_t width, const void* in, void* out);

static size_t kai_get_mr_lhs_imatmul_pack_x8p2vlx4_x8p_sme(void) {
    return MR * kai_get_sme_vector_length_u8() / KR;
}

size_t kai_get_m_step_lhs_imatmul_pack_x8p2vlx4_x8p_sme(void) {
    return kai_get_mr_lhs_imatmul_pack_x8p2vlx4_x8p_sme();
}

size_t kai_get_lhs_packed_offset_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_get_m_step_lhs_imatmul_pack_x8p2vlx4_x8p_sme() == 0);

    return m_idx * k_chunk_count * kai_roundup(k_chunk_length, KR) * sizeof(int8_t);
}

size_t kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
    size_t m, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t m_end = kai_roundup(m, kai_get_mr_lhs_imatmul_pack_x8p2vlx4_x8p_sme());
    return kai_get_lhs_packed_offset_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m_end, k_chunk_count, k_chunk_length);
}

void kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
    size_t m, size_t k_chunk_count, size_t k_chunk_length, const void* const* lhs_ptrs, size_t lhs_ptr_offset,
    const void* pad_ptr, void* lhs_packed) {
    KAI_ASSUME(lhs_ptrs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    const size_t m_step = kai_get_mr_lhs_imatmul_pack_x8p2vlx4_x8p_sme();
    const size_t width = k_chunk_length;

    KAI_ASSERT(m_step <= MAX_M_STEP);
    const uint8_t* in[MAX_M_STEP];

    uint8_t* out_base = lhs_packed;

    kai_commit_za();

    for (size_t i_m = 0; i_m < m; i_m += m_step) {
        for (size_t i_k_chunk = 0; i_k_chunk < k_chunk_count; i_k_chunk += 1) {
            const size_t height = KAI_MIN(m - i_m, m_step);
            void* out = out_base;
            out_base += m_step * kai_roundup(k_chunk_length, KR) * sizeof(int8_t);
            for (size_t y = 0; y < height; y += 1) {
                KAI_ASSERT(i_k_chunk + (i_m + y) * k_chunk_count < m * k_chunk_count);
                in[y] = *(lhs_ptrs + i_m * k_chunk_count + i_k_chunk * m_step + y);
                if (in[y] != pad_ptr) {
                    uintptr_t in_ptr = (uintptr_t)in[y] + lhs_ptr_offset;
                    in[y] = (const uint8_t*)in_ptr;  // NOLINT(performance-no-int-to-ptr)
                }
            }

            kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
                height, width, in, out);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
        }
    }
}

#endif  // Architectural features check.
