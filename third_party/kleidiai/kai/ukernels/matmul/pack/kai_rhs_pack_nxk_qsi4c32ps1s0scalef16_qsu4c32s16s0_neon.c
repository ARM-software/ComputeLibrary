//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);
static const size_t kai_bl = 32;

static inline void convert_s1s0_s16s0(uint8_t* dst_blk, const uint8_t* src_blk) {
    // First half
    for (size_t k = 0; k < kai_bl / 2; k += 2) {
        dst_blk[k / 2] = src_blk[k] & 0xF;
        dst_blk[k / 2] |= src_blk[k + 1] << 4;
    }

    // Second half
    for (size_t k = kai_bl / 2; k < kai_bl; k += 2) {
        dst_blk[k / 2] = src_blk[k - kai_bl / 2] >> 4;
        dst_blk[k / 2] |= src_blk[k - kai_bl / 2 + 1] & 0xF0;
    }
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME(bl == kai_bl);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_num_bytes_per_block(size_t bl) {
    KAI_ASSUME(bl == kai_bl);

    return (bl / 2) + kai_num_bytes_multiplier;
}

inline static size_t kai_rhs_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    return num_bytes_per_block * num_blocks_per_row;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    return nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);

    // The scales are stored after all the nr packed quantized values
    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
}

void kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const float* bias, void* rhs_packed, size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param* params) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME(bias == NULL);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(kr == 4);
    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr >= 1 && kr <= 16);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t num_blocks = k / bl;
    const size_t rhs_stride = kai_rhs_stride(k, bl);
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    uint8_t* rhs_packed_ptr = rhs_packed;

    for (uint64_t n_idx = 0; n_idx < n; n_idx++) {
        uint16_t* rhs_packed_scales =
            (uint16_t*)(rhs_packed_ptr + rhs_packed_stride - (nr * num_blocks * kai_num_bytes_multiplier));

        for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
            uint8_t blk_s1s0[16];

            const uint16_t* blk_scale_ptr =
                (const uint16_t*)(rhs + (block_idx * num_bytes_per_block) + n_idx * rhs_stride);
            const uint8_t* blk_s16s0 = (const uint8_t*)blk_scale_ptr + kai_num_bytes_multiplier;

            convert_s1s0_s16s0(blk_s1s0, blk_s16s0);

            for (size_t bl4_idx = 0; bl4_idx < bl / 4; bl4_idx++) {
                // Uint16 holds 4 int4 values
                ((uint16_t*)rhs_packed_ptr)[(block_idx * bl / 4 + bl4_idx) * nr + (n_idx % nr)] =
                    ((int16_t*)blk_s1s0)[bl4_idx];
            }

            // Num. block (rows) x Nr (cols)
            rhs_packed_scales[(n_idx % nr) + block_idx * nr] = *blk_scale_ptr;
        }

        if (((n_idx + 1) % nr) == 0) {
            rhs_packed_ptr += rhs_packed_stride;
        }
    }
}
#endif  // Architectural features check.
