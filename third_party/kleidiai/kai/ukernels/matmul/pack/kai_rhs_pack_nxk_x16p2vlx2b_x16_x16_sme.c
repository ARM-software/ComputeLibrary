//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    NR = 2,
    KR = 2,
    SR = 1,
    NUM_BYTES_DATA = 2,
    NUM_BYTES_BIAS = 2,
    MAX_BLOCK_HEIGHT = (NR * (KAI_SME_VEC_LENGTH_MAX_BYTES / NUM_BYTES_DATA) / KR),
};

void kai_kernel_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
    size_t height, size_t width, const void* in, void* out, const void* bias);

static size_t get_block_height(void) {
    const size_t block_height = NR * kai_get_sme_vector_length_u16() / KR;
    return block_height;
}

size_t kai_get_n_step_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(void) {
    return get_block_height();
}

size_t kai_get_rhs_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t rhs_stride) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * rhs_stride;
}

size_t kai_get_bias_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * NUM_BYTES_BIAS;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t k) {
    return NUM_BYTES_BIAS + kai_roundup(k, KR) * NUM_BYTES_DATA;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % get_block_height() == 0);

    return n_idx * kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(size_t n, size_t k) {
    return kai_roundup(n, get_block_height()) * kai_get_rhs_packed_stride_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(k);
}

void kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == get_block_height());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    const size_t block_height = get_block_height();
    const size_t width = k;

    const uint8_t* in[MAX_BLOCK_HEIGHT];
    uint8_t* rhs_packed_ptr = rhs_packed;
    const uint8_t* rhs_ptr = rhs;
    const uint8_t* bias_ptr = bias;

    kai_commit_za();

    for (size_t block_y = 0; block_y < n; block_y += block_height) {
        const size_t height = KAI_MIN(n - block_y, block_height);
        uint8_t* out = rhs_packed_ptr + block_y * (NUM_BYTES_BIAS + kai_roundup(k, KR) * NUM_BYTES_DATA);

        for (size_t y = 0; y < height; y++) {
            in[y] = rhs_ptr + (block_y + y) * rhs_stride;
        }

        kai_kernel_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(
            height, width, in, out, bias);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)

        bias_ptr += height * NUM_BYTES_BIAS;
        bias = bias_ptr;
    }
}

#endif  // Architectural features check.
