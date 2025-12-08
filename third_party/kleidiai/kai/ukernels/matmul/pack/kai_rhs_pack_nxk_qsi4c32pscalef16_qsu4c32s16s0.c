//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);
static const size_t kai_bl = 32;

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

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    return nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(k, nr, kr, bl);
}

void kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const float* bias, void* rhs_packed, size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param* params) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME(bias == NULL);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr >= 1 && kr <= 16);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t rhs_stride = kai_rhs_stride(k, bl);
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(k, nr, kr, bl);
    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_segments_per_block = bl / kr;
    const size_t num_bytes_per_segment = kr / 2;

    for (size_t y = 0; y < n; y += nr) {
        const uint8_t* src_row = rhs;
        uint8_t* dst_row = (uint8_t*)rhs_packed + (y / nr) * rhs_packed_stride;

        for (size_t x = 0; x < num_blocks_per_row; ++x) {
            // Store the scales at the end of the block
            uint8_t* scales = (dst_row);

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(y + i, n - 1);
                memcpy(
                    scales + i * kai_num_bytes_multiplier, src_row + src_row_idx * rhs_stride,
                    kai_num_bytes_multiplier);
            }
            src_row += kai_num_bytes_multiplier;

            dst_row += (kai_num_bytes_multiplier * nr);

            // Store the segments
            for (size_t s = 0; s < num_segments_per_block; ++s) {
                for (size_t i = 0; i < nr; ++i) {
                    const size_t src_row_idx = KAI_MIN(y + i, n - 1);

                    if (num_bytes_per_segment == sizeof(uint32_t)) {
                        uint32_t tmp = 0;
                        memcpy(&tmp, src_row + src_row_idx * rhs_stride, num_bytes_per_segment);
                        tmp = tmp ^ 0x88888888;
                        memcpy(dst_row + i * num_bytes_per_segment, &tmp, num_bytes_per_segment);
                    } else if (num_bytes_per_segment == sizeof(uint64_t)) {
                        uint64_t tmp = 0;
                        memcpy(&tmp, src_row + src_row_idx * rhs_stride, num_bytes_per_segment);
                        tmp = tmp ^ 0x8888888888888888ULL;
                        memcpy(dst_row + i * num_bytes_per_segment, &tmp, num_bytes_per_segment);
                    } else {
                        memcpy(
                            dst_row + i * num_bytes_per_segment, src_row + src_row_idx * rhs_stride,
                            num_bytes_per_segment);

                        for (size_t b = 0; b < num_bytes_per_segment; ++b) {
                            uint8_t qs = dst_row[i * num_bytes_per_segment + b];
                            // Add offset (0x88)
                            dst_row[i * num_bytes_per_segment + b] = qs ^ 0x88;
                        }
                    }
                }

                src_row += num_bytes_per_segment;
                dst_row += num_bytes_per_segment * nr;
            }
        }
    }
}
