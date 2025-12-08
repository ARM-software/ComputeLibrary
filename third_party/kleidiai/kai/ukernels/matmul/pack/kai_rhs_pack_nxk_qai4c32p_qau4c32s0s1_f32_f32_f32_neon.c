//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.
#include "kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"

#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_offset_rhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_bl_multiple_of = 32;

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_num_bytes_per_block(size_t bl) {
    return (bl / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_offset_rhs;
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl);
    return nr * (num_bytes_per_block * num_blocks_per_row + kai_num_bytes_bias);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);
    KAI_UNUSED(kr);
    return (n_idx / nr) * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_UNUSED(kr);
    const size_t num_rows = kai_roundup(n, nr) / nr;
    return num_rows * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

void kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const void* zero, const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qai4c32p_params* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr >= 1 && kr <= 16);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(zero != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t num_blocks_per_row = k / bl;
    const size_t rhs_stride = k;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride(k, nr, kr, bl);

    const size_t dst_packed_block_size = kai_get_num_bytes_per_block(bl) * nr;
    const size_t dst_block_data_size = (bl / 2) * nr;
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_bias_offset = num_blocks_per_row * dst_packed_block_size;
    const size_t k_block_length_in_bytes = kr / sr;
    const size_t k_interleaved_v = 16U;

    const size_t rhs_zero_point = params->rhs_zero_point;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;
        float* dst_row_bias = (float*)(dst_row + dst_bias_offset);

        for (size_t block_idx = 0; block_idx < num_blocks_per_row; block_idx++) {
            uint8_t* block_dst_row = dst_row + block_idx * dst_packed_block_size;
            float* block_dst_zp = (float*)(block_dst_row + dst_block_data_size);
            float* block_dst_scale = block_dst_zp + nr;

            for (size_t block_byte_idx = 0; block_byte_idx < dst_block_data_size; ++block_byte_idx) {
                const size_t dst_byte_idx = block_byte_idx;
                const size_t k_block_idx = dst_byte_idx / k_block_length_in_bytes;
                const size_t k_block_byte_idx = dst_byte_idx % k_block_length_in_bytes;
                const size_t super_k_block_idx = k_block_idx / nr;
                const size_t nr_idx = k_block_idx % nr;

                const size_t k_adjustment =
                    ((k_block_byte_idx + super_k_block_idx * k_block_length_in_bytes) / k_interleaved_v) *
                    k_interleaved_v;
                const size_t k0_idx = k_block_byte_idx + super_k_block_idx * k_block_length_in_bytes + k_adjustment;
                const size_t k1_idx = k0_idx + k_interleaved_v;
                const size_t n0_idx = dst_row_idx * nr + nr_idx;

                // Clamp the index to avoid out-of-bound reads
                const size_t n0_valid_idx = KAI_MIN(n0_idx, n - 1);

                const size_t src_addr_byte0 = (k0_idx + n0_valid_idx * rhs_stride + block_idx * bl) / 2;
                const size_t src_addr_byte1 = (k1_idx + n0_valid_idx * rhs_stride + block_idx * bl) / 2;

                uint8_t byte0 = rhs_zero_point | rhs_zero_point << 4;
                uint8_t byte1 = rhs_zero_point | rhs_zero_point << 4;

                if (k0_idx < k) {
                    byte0 = rhs[src_addr_byte0];
                }
                if (k1_idx < k) {
                    byte1 = rhs[src_addr_byte1];
                }

                const size_t shift_right_x0 = (k0_idx % 2 == 0) ? 4 : 0;
                const size_t shift_right_x1 = (k1_idx % 2 == 0) ? 4 : 0;

                const uint8_t src_x0_lo = (byte0 >> shift_right_x0) & 0x0F;
                const uint8_t src_x0_hi = (byte1 >> shift_right_x1) & 0x0F;

                const uint8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);

                *block_dst_row = dst_qs0 ^ 0x88;
                block_dst_row += sizeof(uint8_t);
            }

            // Adjust the zero points and scales
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);

                const float* block_zero = (const float*)zero + num_blocks_per_row * src_row_idx;
                const float* block_scale = (const float*)scale + num_blocks_per_row * src_row_idx;

                *block_dst_zp = block_zero[block_idx];
                *block_dst_scale = block_scale[block_idx] * 0.0625F;

                block_dst_zp++;
                block_dst_scale++;
            }
        }
        // Set the bias
        if (bias == NULL) {
            memset(dst_row_bias, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);

                dst_row_bias[i] = *((const float*)bias + src_row_idx);
            }
        }
    }
}
#endif
