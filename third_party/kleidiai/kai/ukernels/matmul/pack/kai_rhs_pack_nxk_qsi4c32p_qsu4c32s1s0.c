//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_nr_multiple_of = 4;
static const size_t kai_bl_multiple_of = 32;

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_num_bytes_per_block(size_t bl, size_t num_bytes_multiplier_rhs) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return (bl / 2) + num_bytes_multiplier_rhs;
}

inline static size_t kai_get_rhs_packed_offset_end_of_all_blocks(
    size_t k, size_t nr, size_t kr, size_t bl, size_t num_bytes_multiplier_rhs) {
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);

    return (nr * num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
    size_t n_idx,  //
    size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t sr,  //
    size_t bl,  //
    enum kai_datatype scale_dt) {
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);

    return nr * ((num_bytes_per_block * num_blocks_per_row) + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
    size_t n_idx,  //
    size_t k,      //
    size_t nr,     //
    size_t kr,     //
    size_t sr,     //
    size_t bl,     //
    enum kai_datatype scale_dt) {
    KAI_ASSERT((n_idx % nr) == 0);
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
    size_t n,   //
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t sr,  //
    size_t bl,  //
    enum kai_datatype scale_dt) {
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
    size_t num_groups,    //
    size_t n,             //
    size_t k,             //
    size_t nr,            //
    size_t kr,            //
    size_t sr,            //
    size_t bl,            //
    const uint8_t* rhs,   //
    size_t rhs_stride,    //
    const float* bias,    //
    const void* scale,    //
    size_t scale_stride,  //
    void* rhs_packed,     //
    size_t extra_bytes,   //
    const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params* params) {
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(params->scale_dt == kai_dt_bf16);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)
    const enum kai_datatype scale_dt = params->scale_dt;
    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t rhs_packed_offset_end_of_all_blocks =
        kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, kr, bl, num_bytes_multiplier_rhs);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);
    const size_t block_length_in_bytes = kr / sr;

    uint8_t* dst_row = (uint8_t*)rhs_packed;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        // Iterate over the quantized blocks
        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            // Store the scales after packing all K values in the block
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = (const uint8_t*)scale + dst_qblock_idx * num_bytes_multiplier_rhs;

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                const void* src_scales_ptr = scale_ptr + src_row_idx * scale_stride;
                void* dst_scales_ptr = rhs_packed_scale + i * num_bytes_multiplier_rhs;

                memcpy(
                    dst_scales_ptr,             //
                    src_scales_ptr,             //
                    num_bytes_multiplier_rhs);  //
            }

            size_t k0_idx_i = dst_qblock_idx * bl;

            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += 16) {
                for (size_t segment_idx = 0; segment_idx < 16 / block_length_in_bytes; ++segment_idx) {
                    for (size_t nr_idx = 0; nr_idx < nr; ++nr_idx) {
                        const size_t n0_idx = dst_row_idx + nr_idx;

                        // Two int4 values are stored in one byte.
                        // The lower order part of the byte (low) holds the first nibble (K-index + 0).
                        // The higher order of the byte holds the second nibble (K-index + 16).
                        size_t k0_idx = k0_idx_i;
                        size_t k1_idx = k0_idx_i + 16;

                        // Clamp the index to avoid out-of-bound reads
                        const size_t n0_valid_idx = KAI_MIN(n0_idx, n - 1);
                        float d = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx]);

                        int32_t partial_sum = 0;

                        size_t src_addr_byte0 = (k0_idx / 2) + n0_valid_idx * rhs_stride;

                        for (size_t block_byte_idx = 0; block_byte_idx < block_length_in_bytes; block_byte_idx += 2) {
                            // Initialize the byte with the zero-point (8)
                            // e.g. uint8_t byte0 = 8 | 8 << 4
                            uint8_t byte0 = 136;
                            uint8_t byte1 = 136;
                            uint8_t byte2 = 136;
                            uint8_t byte3 = 136;

                            if (k0_idx < k) {
                                byte0 = rhs[src_addr_byte0];
                            }

                            if (k1_idx < k) {
                                byte1 = rhs[src_addr_byte0 + 8];
                            }

                            if (k0_idx + 1 < k) {
                                byte2 = byte0;
                            }

                            if (k1_idx + 1 < k) {
                                byte3 = byte1;
                            }

                            k0_idx += 2;
                            k1_idx += 2;

                            const uint8_t src_x0_lo = byte0 & 0x0F;
                            const uint8_t src_x0_hi = byte1 & 0x0F;
                            const uint8_t src_x1_lo = (byte2 >> 4) & 0x0F;
                            const uint8_t src_x1_hi = (byte3 >> 4) & 0x0F;

                            partial_sum += (int32_t)src_x0_lo;
                            partial_sum += (int32_t)src_x0_hi;
                            partial_sum += (int32_t)src_x1_lo;
                            partial_sum += (int32_t)src_x1_hi;
                            partial_sum -= 32;  // 4 * zero_point (8)

                            const uint16_t dst_q =
                                ((src_x0_lo)) | ((src_x0_hi) << 4) | ((src_x1_lo) << 8) | ((src_x1_hi) << 12);

                            *((uint16_t*)dst_row) = dst_q ^ 0x8888;

                            dst_row += 2;
                            src_addr_byte0 += 1;
                        }
                        // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                        sums[nr_idx] += (float)partial_sum * d;
                        // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    }

                    k0_idx_i += block_length_in_bytes;
                }
                k0_idx_i += 16;
            }
            // Move the pointer after scales
            dst_row += num_bytes_multiplier_rhs * nr;
        }

        // Move the pointer after the row sum
        dst_row += kai_num_bytes_sum_rhs * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
        // Move the pointer after the row sum
        dst_row += kai_num_bytes_bias * nr;
    }
}
