//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"

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

size_t kai_get_n_step_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
    size_t n_idx,  //
    size_t rhs_stride) {
    KAI_UNUSED(rhs_stride);
    KAI_ASSERT((n_idx % 2) == 0);
    return (n_idx / 2) * sizeof(int8_t);
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
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

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
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

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
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

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
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
    const struct kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params* params) {
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
    // "n" columns and "k" rows (kxn)

    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(params->scale_dt);
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(k, nr, kr, sr, bl, params->scale_dt);
    const size_t rhs_packed_offset_end_of_all_blocks =
        kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, kr, bl, num_bytes_multiplier_rhs);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t k_interleaved_v = 16U;
    const size_t block_length_in_bytes = kr / sr;

    const int32_t rhs_zero_point = params->rhs_zero_point;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        // Before packing, it keeps the pointer to the first quantized block
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;

        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        // Iterate over the quantized blocks
        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            // Store the scales after packing all K values
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = scale;

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);

                void* dst_scales_ptr = rhs_packed_scale + i * num_bytes_multiplier_rhs;
                const void* src_scales_ptr = scale_ptr + dst_qblock_idx * num_bytes_multiplier_rhs +  //
                    (src_row_idx * scale_stride);                                                     //

                memcpy(
                    dst_scales_ptr,             //
                    src_scales_ptr,             //
                    num_bytes_multiplier_rhs);  //
            }

            size_t kr_block_idx = 0;
            for (size_t dst_byte_idx = 0; dst_byte_idx < nr * num_bytes_per_block_k;
                 dst_byte_idx += block_length_in_bytes) {
                const size_t super_kr_block_idx = kr_block_idx / nr;
                const size_t nr_idx = kr_block_idx % nr;
                const size_t n0_idx = dst_row_idx * nr + nr_idx;

                // Clamp the index to avoid out-of-bound reads
                const size_t n0_valid_idx = KAI_MIN(n0_idx, n - 1);

                float d = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx]);

                const size_t k_adjustment =
                    ((super_kr_block_idx * block_length_in_bytes) / k_interleaved_v) * k_interleaved_v;
                size_t k0_idx = dst_qblock_idx * bl + super_kr_block_idx * block_length_in_bytes + k_adjustment;
                size_t k1_idx = k0_idx + k_interleaved_v;

                float partial_sum = 0.0F;

                for (size_t block_byte_idx = 0; block_byte_idx < block_length_in_bytes; ++block_byte_idx) {
                    const size_t src_addr_byte0 = (n0_valid_idx / 2) + k0_idx * rhs_stride;
                    const size_t src_addr_byte1 = (n0_valid_idx / 2) + k1_idx * rhs_stride;

                    uint8_t byte0 = rhs_zero_point | rhs_zero_point << 4;
                    uint8_t byte1 = rhs_zero_point | rhs_zero_point << 4;

                    if (k0_idx < k) {
                        byte0 = rhs[src_addr_byte0];
                    }

                    if (k1_idx < k) {
                        byte1 = rhs[src_addr_byte1];
                    }

                    if ((n0_idx % 2) == 0) {
                        const uint8_t src_x0_lo = (byte0 & 0x0F);
                        const uint8_t src_x0_hi = (byte1 & 0x0F);

                        partial_sum += (float)((int32_t)src_x0_lo + (int32_t)src_x0_hi - 2 * rhs_zero_point) * d;

                        const uint8_t dst_qs0 = src_x0_lo | (src_x0_hi << 4);

                        dst_row[dst_byte_idx + block_byte_idx] = dst_qs0 ^ 0x88;
                    } else {
                        const uint8_t src_x1_lo = (byte0 >> 4);
                        const uint8_t src_x1_hi = (byte1 >> 4);

                        partial_sum += (float)((int32_t)src_x1_lo + (int32_t)src_x1_hi - 2 * rhs_zero_point) * d;

                        const uint8_t dst_qs1 = src_x1_lo | (src_x1_hi << 4);

                        dst_row[dst_byte_idx + block_byte_idx] = dst_qs1 ^ 0x88;
                    }
                    k0_idx++;
                    k1_idx++;
                }
                sums[nr_idx] += partial_sum;

                // Increment the Kr block index
                kr_block_idx++;
            }
            // Move the pointer after K values
            dst_row += num_bytes_per_block * nr;
        }

        // Move the pointer after the row sum
        dst_row += kai_num_bytes_sum_rhs * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
    }
}
