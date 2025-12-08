//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// nrx4 => this function can take in generic nr values but the input is expected to have a block depth of 4.
// Block depth is calculated as kr / sr. The values of these parameters are defined in the matmul ukernel.

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon.h"

#include <arm_neon.h>
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

size_t kai_get_n_step_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
    size_t n_idx,  //
    size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
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

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
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

    return (n_idx / nr) *
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
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

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
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
    KAI_ASSERT(block_length_in_bytes == 4);

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
            const uint8x8_t top_mask = vdup_n_u8(0xF0);
            const uint8x8_t bottom_mask = vdup_n_u8(0x0F);
            const uint32x2_t zero_point_conversion_mask = vdup_n_u32(0x88888888);

            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += 16) {
                for (size_t nr_idx = 0; nr_idx < nr; nr_idx += 4) {
                    // Clamp the indices to avoid out-of-bound reads
                    const size_t n0_idx = KAI_MIN(dst_row_idx + nr_idx, n - 1);
                    const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                    const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                    const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);

                    const float d0 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 0]);
                    const float d1 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 1]);
                    const float d2 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 2]);
                    const float d3 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 3]);

                    // Initialize partial sum taking new zero-point (8) into account
                    int32_t partial_sum0 = -(32 * 8);
                    int32_t partial_sum1 = -(32 * 8);
                    int32_t partial_sum2 = -(32 * 8);
                    int32_t partial_sum3 = -(32 * 8);

                    const uint8_t* src_block_base = rhs + ((k0_idx_i / 2) + dst_byte_idx);

                    const uint8x8_t vld0_0 = vld1_u8(src_block_base + n0_idx * rhs_stride);
                    const uint8x8_t vld0_1 = vld1_u8(src_block_base + n0_idx * rhs_stride + 8);
                    const uint8x8_t vld1_0 = vld1_u8(src_block_base + n1_idx * rhs_stride);
                    const uint8x8_t vld1_1 = vld1_u8(src_block_base + n1_idx * rhs_stride + 8);
                    const uint8x8_t vld2_0 = vld1_u8(src_block_base + n2_idx * rhs_stride);
                    const uint8x8_t vld2_1 = vld1_u8(src_block_base + n2_idx * rhs_stride + 8);
                    const uint8x8_t vld3_0 = vld1_u8(src_block_base + n3_idx * rhs_stride);
                    const uint8x8_t vld3_1 = vld1_u8(src_block_base + n3_idx * rhs_stride + 8);

                    // Reorder blocks to give correct packing
                    const uint8x8_t vld0_0_lower = vand_u8(vld0_0, bottom_mask);
                    const uint8x8_t vld0_1_lower = vshl_n_u8(vld0_1, 4);
                    const uint8x8_t vld0_0_upper = vshr_n_u8(vld0_0, 4);
                    const uint8x8_t vld0_1_upper = vand_u8(vld0_1, top_mask);
                    const uint8x8_t vstr0_04 =
                        vorr_u8(vzip1_u8(vld0_0_lower, vld0_0_upper), vzip1_u8(vld0_1_lower, vld0_1_upper));
                    const uint8x8_t vstr0_46 =
                        vorr_u8(vzip2_u8(vld0_0_lower, vld0_0_upper), vzip2_u8(vld0_1_lower, vld0_1_upper));

                    const uint8x8_t vld1_0_lower = vand_u8(vld1_0, bottom_mask);
                    const uint8x8_t vld1_1_lower = vshl_n_u8(vld1_1, 4);
                    const uint8x8_t vld1_0_upper = vshr_n_u8(vld1_0, 4);
                    const uint8x8_t vld1_1_upper = vand_u8(vld1_1, top_mask);
                    const uint8x8_t vstr0_04_1 =
                        vorr_u8(vzip1_u8(vld1_0_lower, vld1_0_upper), vzip1_u8(vld1_1_lower, vld1_1_upper));
                    const uint8x8_t vstr0_46_1 =
                        vorr_u8(vzip2_u8(vld1_0_lower, vld1_0_upper), vzip2_u8(vld1_1_lower, vld1_1_upper));

                    const uint8x8_t vld2_0_lower = vand_u8(vld2_0, bottom_mask);
                    const uint8x8_t vld2_1_lower = vshl_n_u8(vld2_1, 4);
                    const uint8x8_t vld2_0_upper = vshr_n_u8(vld2_0, 4);
                    const uint8x8_t vld2_1_upper = vand_u8(vld2_1, top_mask);
                    const uint8x8_t vstr0_15 =
                        vorr_u8(vzip1_u8(vld2_0_lower, vld2_0_upper), vzip1_u8(vld2_1_lower, vld2_1_upper));
                    const uint8x8_t vstr0_57 =
                        vorr_u8(vzip2_u8(vld2_0_lower, vld2_0_upper), vzip2_u8(vld2_1_lower, vld2_1_upper));

                    const uint8x8_t vld3_0_lower = vand_u8(vld3_0, bottom_mask);
                    const uint8x8_t vld3_1_lower = vshl_n_u8(vld3_1, 4);
                    const uint8x8_t vld3_0_upper = vshr_n_u8(vld3_0, 4);
                    const uint8x8_t vld3_1_upper = vand_u8(vld3_1, top_mask);
                    const uint8x8_t vstr0_15_1 =
                        vorr_u8(vzip1_u8(vld3_0_lower, vld3_0_upper), vzip1_u8(vld3_1_lower, vld3_1_upper));
                    const uint8x8_t vstr0_57_1 =
                        vorr_u8(vzip2_u8(vld3_0_lower, vld3_0_upper), vzip2_u8(vld3_1_lower, vld3_1_upper));

                    const uint32x2_t vstr0_0 =
                        vzip1_u32(vreinterpret_u32_u8(vstr0_04), vreinterpret_u32_u8(vstr0_04_1));
                    const uint32x2_t vstr0_4 =
                        vzip1_u32(vreinterpret_u32_u8(vstr0_46), vreinterpret_u32_u8(vstr0_46_1));
                    const uint32x2_t vstr0_2 =
                        vzip2_u32(vreinterpret_u32_u8(vstr0_04), vreinterpret_u32_u8(vstr0_04_1));
                    const uint32x2_t vstr0_6 =
                        vzip2_u32(vreinterpret_u32_u8(vstr0_46), vreinterpret_u32_u8(vstr0_46_1));
                    const uint32x2_t vstr0_1 =
                        vzip1_u32(vreinterpret_u32_u8(vstr0_15), vreinterpret_u32_u8(vstr0_15_1));
                    const uint32x2_t vstr0_5 =
                        vzip1_u32(vreinterpret_u32_u8(vstr0_57), vreinterpret_u32_u8(vstr0_57_1));
                    const uint32x2_t vstr0_3 =
                        vzip2_u32(vreinterpret_u32_u8(vstr0_15), vreinterpret_u32_u8(vstr0_15_1));
                    const uint32x2_t vstr0_7 =
                        vzip2_u32(vreinterpret_u32_u8(vstr0_57), vreinterpret_u32_u8(vstr0_57_1));

                    // Convert to signed int4 and store repacked values
                    vst1_u32((uint32_t*)dst_row + 0, veor_u32(vstr0_0, zero_point_conversion_mask));
                    vst1_u32((uint32_t*)dst_row + 2, veor_u32(vstr0_1, zero_point_conversion_mask));

                    vst1_u32(
                        (uint32_t*)(dst_row + nr * block_length_in_bytes) + 0,
                        veor_u32(vstr0_2, zero_point_conversion_mask));
                    vst1_u32(
                        (uint32_t*)(dst_row + nr * block_length_in_bytes) + 2,
                        veor_u32(vstr0_3, zero_point_conversion_mask));

                    vst1_u32(
                        (uint32_t*)(dst_row + (2 * nr * block_length_in_bytes)) + 0,
                        veor_u32(vstr0_4, zero_point_conversion_mask));
                    vst1_u32(
                        (uint32_t*)(dst_row + (2 * nr * block_length_in_bytes)) + 2,
                        veor_u32(vstr0_5, zero_point_conversion_mask));

                    vst1_u32(
                        (uint32_t*)(dst_row + (3 * nr * block_length_in_bytes)) + 0,
                        veor_u32(vstr0_6, zero_point_conversion_mask));
                    vst1_u32(
                        (uint32_t*)(dst_row + (3 * nr * block_length_in_bytes)) + 2,
                        veor_u32(vstr0_7, zero_point_conversion_mask));

                    // Calculate and store row sums
                    partial_sum0 += (int32_t)vaddlvq_u16(vaddl_u8(
                        vadd_u8(vld0_0_lower, vand_u8(vld0_1, bottom_mask)),
                        vadd_u8(vld0_0_upper, vshr_n_u8(vld0_1, 4))));
                    partial_sum1 += (int32_t)vaddlvq_u16(vaddl_u8(
                        vadd_u8(vld1_0_lower, vand_u8(vld1_1, bottom_mask)),
                        vadd_u8(vld1_0_upper, vshr_n_u8(vld1_1, 4))));
                    partial_sum2 += (int32_t)vaddlvq_u16(vaddl_u8(
                        vadd_u8(vld2_0_lower, vand_u8(vld2_1, bottom_mask)),
                        vadd_u8(vld2_0_upper, vshr_n_u8(vld2_1, 4))));
                    partial_sum3 += (int32_t)vaddlvq_u16(vaddl_u8(
                        vadd_u8(vld3_0_lower, vand_u8(vld3_1, bottom_mask)),
                        vadd_u8(vld3_0_upper, vshr_n_u8(vld3_1, 4))));

                    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    sums[nr_idx + 0] += (float)partial_sum0 * d0;
                    sums[nr_idx + 1] += (float)partial_sum1 * d1;
                    sums[nr_idx + 2] += (float)partial_sum2 * d2;
                    sums[nr_idx + 3] += (float)partial_sum3 * d3;
                    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

                    dst_row += (4 * block_length_in_bytes);
                }
                // Skip to end of qblock
                dst_row += 3 * nr * block_length_in_bytes;
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
#endif  // Architectural features check.
