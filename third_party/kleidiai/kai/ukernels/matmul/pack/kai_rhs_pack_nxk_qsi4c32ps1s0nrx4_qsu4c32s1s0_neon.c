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

#include "kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_nr_multiple_of = 4;
static const size_t kai_bl_multiple_of = 32;

static size_t kai_get_num_blocks_per_row(const size_t k, const size_t bl) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

static size_t kai_get_num_bytes_per_block(const size_t bl, const size_t num_bytes_multiplier_rhs) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return (bl / 2) + num_bytes_multiplier_rhs;
}

static size_t kai_get_rhs_packed_offset_end_of_all_blocks(
    // clang-format off
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t bl,
    const size_t num_bytes_multiplier_rhs) {
    // clang-format on
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);

    return (nr * num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(const size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(const size_t n_idx, const size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
    // clang-format off
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const enum kai_datatype scale_dt) {
    // clang-format on
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

// clang-format off
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
    const size_t n_idx,
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const enum kai_datatype scale_dt) {
    // clang-format on
    KAI_ASSERT((n_idx % nr) == 0);
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    return (n_idx / nr) *
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

// clang-format off
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
    const size_t n,   //
    const size_t k,   //
    const size_t nr,  //
    const size_t kr,  //
    const size_t sr,  //
    const size_t bl,  //
    const enum kai_datatype scale_dt) {
    // clang-format on
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows *
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
    // clang-format off
    const size_t num_groups,
    const size_t n,
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const uint8_t* rhs,
    const size_t rhs_stride,
    const float* bias,
    const void* scale,
    const size_t scale_stride,
    void* rhs_packed,
    const size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params* params) {
    // clang-format on
    KAI_UNUSED(num_groups);
    KAI_UNUSED(extra_bytes);
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
    const size_t block_length = kr / sr;
    KAI_ASSERT(block_length == 4);
    const enum kai_datatype scale_dt = params->scale_dt;
    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t rhs_packed_offset_end_of_all_blocks =
        kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, kr, bl, num_bytes_multiplier_rhs);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);
    const size_t block_length_in_bytes = block_length / 2;

    const int8x16_t rhs_zero_point = vdupq_n_s8(8);
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const size_t num_bytes_processed = 16;

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

            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += num_bytes_processed) {
                for (size_t nr_idx = 0; nr_idx < nr; nr_idx += 4) {
                    // Clamp the indices to avoid out-of-bound reads
                    const size_t n0_idx = KAI_MIN(dst_row_idx + nr_idx, n - 1);
                    const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                    const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                    const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);

                    // Load scales
                    const float d0 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 0]);
                    const float d1 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 1]);
                    const float d2 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 2]);
                    const float d3 = kai_cast_f32_bf16(((uint16_t*)rhs_packed_scale)[nr_idx + 3]);

                    // Initialize partial sum
                    int32_t partial_sum0 = 0;
                    int32_t partial_sum1 = 0;
                    int32_t partial_sum2 = 0;
                    int32_t partial_sum3 = 0;

                    const uint8_t* src_block_base = rhs + ((k0_idx_i / 2) + dst_byte_idx);
                    const uint8x16_t vsrc0_0 = vld1q_u8(src_block_base + n0_idx * rhs_stride);
                    const uint8x16_t vsrc1_0 = vld1q_u8(src_block_base + n1_idx * rhs_stride);
                    const uint8x16_t vsrc2_0 = vld1q_u8(src_block_base + n2_idx * rhs_stride);
                    const uint8x16_t vsrc3_0 = vld1q_u8(src_block_base + n3_idx * rhs_stride);

                    // Get the lower and higher nibble and apply zero-points
                    const int8x16_t vsrc0_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc0_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc0_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc0_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc1_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc1_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc1_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc1_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc2_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc2_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc2_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc2_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc3_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc3_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc3_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc3_0, 4)), rhs_zero_point);

                    // Calculate and store row sums
                    partial_sum0 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc0_0_lo), vget_high_s8(vsrc0_0_lo)),
                        vadd_s8(vget_low_s8(vsrc0_0_hi), vget_high_s8(vsrc0_0_hi))));
                    partial_sum1 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc1_0_lo), vget_high_s8(vsrc1_0_lo)),
                        vadd_s8(vget_low_s8(vsrc1_0_hi), vget_high_s8(vsrc1_0_hi))));
                    partial_sum2 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc2_0_lo), vget_high_s8(vsrc2_0_lo)),
                        vadd_s8(vget_low_s8(vsrc2_0_hi), vget_high_s8(vsrc2_0_hi))));
                    partial_sum3 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc3_0_lo), vget_high_s8(vsrc3_0_lo)),
                        vadd_s8(vget_low_s8(vsrc3_0_hi), vget_high_s8(vsrc3_0_hi))));

                    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    sums[nr_idx + 0] += (float)partial_sum0 * d0;
                    sums[nr_idx + 1] += (float)partial_sum1 * d1;
                    sums[nr_idx + 2] += (float)partial_sum2 * d2;
                    sums[nr_idx + 3] += (float)partial_sum3 * d3;
                    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

                    const uint8x16_t vdst_u8_0 = vorrq_u8(
                        vandq_u8(vreinterpretq_u8_s8(vsrc0_0_lo), low_mask),
                        vshlq_n_u8(vreinterpretq_u8_s8(vsrc0_0_hi), 4));
                    const uint8x16_t vdst_u8_1 = vorrq_u8(
                        vandq_u8(vreinterpretq_u8_s8(vsrc1_0_lo), low_mask),
                        vshlq_n_u8(vreinterpretq_u8_s8(vsrc1_0_hi), 4));
                    const uint8x16_t vdst_u8_2 = vorrq_u8(
                        vandq_u8(vreinterpretq_u8_s8(vsrc2_0_lo), low_mask),
                        vshlq_n_u8(vreinterpretq_u8_s8(vsrc2_0_hi), 4));
                    const uint8x16_t vdst_u8_3 = vorrq_u8(
                        vandq_u8(vreinterpretq_u8_s8(vsrc3_0_lo), low_mask),
                        vshlq_n_u8(vreinterpretq_u8_s8(vsrc3_0_hi), 4));

                    // Reorder to interleave nr rows
                    const uint16x8_t vdst_u16_0 = vreinterpretq_u16_u8(vdst_u8_0);
                    const uint16x8_t vdst_u16_1 = vreinterpretq_u16_u8(vdst_u8_1);
                    const uint16x8_t vdst_u16_2 = vreinterpretq_u16_u8(vdst_u8_2);
                    const uint16x8_t vdst_u16_3 = vreinterpretq_u16_u8(vdst_u8_3);

                    const uint32x4_t vdst_u32_0 = vreinterpretq_u32_u16(vzip1q_u16(vdst_u16_0, vdst_u16_1));
                    const uint32x4_t vdst_u32_1 = vreinterpretq_u32_u16(vzip1q_u16(vdst_u16_2, vdst_u16_3));
                    const uint32x4_t vdst_u32_2 = vreinterpretq_u32_u16(vzip2q_u16(vdst_u16_0, vdst_u16_1));
                    const uint32x4_t vdst_u32_3 = vreinterpretq_u32_u16(vzip2q_u16(vdst_u16_2, vdst_u16_3));

                    const uint32x4_t vdst0_0 = vzip1q_u32(vdst_u32_0, vdst_u32_1);
                    const uint32x4_t vdst1_0 = vzip2q_u32(vdst_u32_0, vdst_u32_1);
                    const uint32x4_t vdst2_0 = vzip1q_u32(vdst_u32_2, vdst_u32_3);
                    const uint32x4_t vdst3_0 = vzip2q_u32(vdst_u32_2, vdst_u32_3);

                    // Store packed values
                    vst1_u32((uint32_t*)dst_row, vget_low_u32(vdst0_0));
                    vst1_u32((uint32_t*)(dst_row + nr * block_length_in_bytes), vget_high_u32(vdst0_0));
                    vst1_u32((uint32_t*)(dst_row + (2 * nr * block_length_in_bytes)), vget_low_u32(vdst1_0));

                    vst1_u32((uint32_t*)(dst_row + (3 * nr * block_length_in_bytes)), vget_high_u32(vdst1_0));
                    vst1_u32((uint32_t*)(dst_row + (4 * nr * block_length_in_bytes)), vget_low_u32(vdst2_0));
                    vst1_u32((uint32_t*)(dst_row + (5 * nr * block_length_in_bytes)), vget_high_u32(vdst2_0));
                    vst1_u32((uint32_t*)(dst_row + (6 * nr * block_length_in_bytes)), vget_low_u32(vdst3_0));
                    vst1_u32((uint32_t*)(dst_row + (7 * nr * block_length_in_bytes)), vget_high_u32(vdst3_0));

                    dst_row += (4 * block_length_in_bytes);
                }
                // Skip to end of qblock
                dst_row += 7 * nr * block_length_in_bytes;
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
