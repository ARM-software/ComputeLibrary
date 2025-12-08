//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

// nrx4 => this function can take in generic nr values but the input is expected to have a block depth of 4.
// Block depth is calculated as kr / sr. The values of these parameters are defined in the matmul ukernel.

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

size_t kai_get_n_step_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(const size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
    const size_t n_idx,  //
    const size_t rhs_stride) {
    KAI_UNUSED(rhs_stride);
    KAI_ASSERT((n_idx % 2) == 0);

    return (n_idx / 2) * sizeof(int8_t);
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
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
size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
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
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

// clang-format off
size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
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
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
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
    const struct kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params* params) {
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
    // "k" rows and "n" columns (kxn)
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

    const int8x8_t rhs_zero_point = vdup_n_s8(8);
    const uint8x8_t low_mask = vdup_n_u8(0x0F);
    const size_t num_bytes_processed = 2;

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
                for (size_t nr_idx = 0; nr_idx < nr; nr_idx += 16) {
                    // Clamp the indices to avoid out-of-bound reads
                    const size_t n0_idx = KAI_MIN(dst_row_idx + nr_idx, n - 1);

                    // Load scales and convert to float
#if defined(__ARM_FEATURE_BF16)

                    const bfloat16_t* rhs_bf16_scale = (const bfloat16_t*)rhs_packed_scale + nr_idx + 0;
                    const bfloat16x4x4_t vd_bf16 = vld1_bf16_x4(rhs_bf16_scale);
                    const float32x4_t vd_0 = vcvt_f32_bf16(vd_bf16.val[0]);
                    const float32x4_t vd_1 = vcvt_f32_bf16(vd_bf16.val[1]);
                    const float32x4_t vd_2 = vcvt_f32_bf16(vd_bf16.val[2]);
                    const float32x4_t vd_3 = vcvt_f32_bf16(vd_bf16.val[3]);
#else
                    // Portable BF16 -> F32 conversion using integer NEON: (u16 << 16) reinterpret as f32
                    const uint16_t* bf16_ptr = ((const uint16_t*)rhs_packed_scale) + nr_idx;
                    const uint16x4_t vbf0 = vld1_u16(bf16_ptr + 0);
                    const uint16x4_t vbf1 = vld1_u16(bf16_ptr + 4);
                    const uint16x4_t vbf2 = vld1_u16(bf16_ptr + 8);
                    const uint16x4_t vbf3 = vld1_u16(bf16_ptr + 12);
                    const uint32x4_t vbf0_u32 = vshlq_n_u32(vmovl_u16(vbf0), 16);
                    const uint32x4_t vbf1_u32 = vshlq_n_u32(vmovl_u16(vbf1), 16);
                    const uint32x4_t vbf2_u32 = vshlq_n_u32(vmovl_u16(vbf2), 16);
                    const uint32x4_t vbf3_u32 = vshlq_n_u32(vmovl_u16(vbf3), 16);
                    const float32x4_t vd_0 = vreinterpretq_f32_u32(vbf0_u32);
                    const float32x4_t vd_1 = vreinterpretq_f32_u32(vbf1_u32);
                    const float32x4_t vd_2 = vreinterpretq_f32_u32(vbf2_u32);
                    const float32x4_t vd_3 = vreinterpretq_f32_u32(vbf3_u32);
#endif

                    const uint8_t* src_block_base = rhs + n0_idx / 2;
                    const size_t k_idx = k0_idx_i + dst_byte_idx * 2;
                    const uint8x8_t vsrc0_0 = vld1_u8(src_block_base + ((k_idx)*rhs_stride));
                    const uint8x8_t vsrc1_0 = vld1_u8(src_block_base + ((k_idx + 1) * rhs_stride));
                    const uint8x8_t vsrc2_0 = vld1_u8(src_block_base + ((k_idx + 2) * rhs_stride));
                    const uint8x8_t vsrc3_0 = vld1_u8(src_block_base + ((k_idx + 3) * rhs_stride));

                    // Get the lower and higher nibble and apply zero-points
                    const int8x8_t vsrc0_lo = vsub_s8(vreinterpret_s8_u8(vand_u8(vsrc0_0, low_mask)), rhs_zero_point);
                    const int8x8_t vsrc0_hi = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(vsrc0_0, 4)), rhs_zero_point);
                    const int8x8_t vsrc1_lo = vsub_s8(vreinterpret_s8_u8(vand_u8(vsrc1_0, low_mask)), rhs_zero_point);
                    const int8x8_t vsrc1_hi = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(vsrc1_0, 4)), rhs_zero_point);
                    const int8x8_t vsrc2_lo = vsub_s8(vreinterpret_s8_u8(vand_u8(vsrc2_0, low_mask)), rhs_zero_point);
                    const int8x8_t vsrc2_hi = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(vsrc2_0, 4)), rhs_zero_point);
                    const int8x8_t vsrc3_lo = vsub_s8(vreinterpret_s8_u8(vand_u8(vsrc3_0, low_mask)), rhs_zero_point);
                    const int8x8_t vsrc3_hi = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(vsrc3_0, 4)), rhs_zero_point);

                    // Calculate and store row sums
                    const int16x8_t vsum_lo = vaddl_s8(vadd_s8(vsrc0_lo, vsrc1_lo), vadd_s8(vsrc2_lo, vsrc3_lo));
                    const int16x8_t vsum_hi = vaddl_s8(vadd_s8(vsrc0_hi, vsrc1_hi), vadd_s8(vsrc2_hi, vsrc3_hi));

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                    const float16x8_t vsum_0 = vcvtq_f16_s16(vzip1q_s16(vsum_lo, vsum_hi));
                    const float16x8_t vsum_1 = vcvtq_f16_s16(vzip2q_s16(vsum_lo, vsum_hi));

                    const float32x4_t vpartialsum_0 = vcvt_f32_f16(vget_low_f16(vsum_0));
                    const float32x4_t vpartialsum_1 = vcvt_high_f32_f16(vsum_0);
                    const float32x4_t vpartialsum_2 = vcvt_f32_f16(vget_low_f16(vsum_1));
                    const float32x4_t vpartialsum_3 = vcvt_high_f32_f16(vsum_1);
#else
                    // Portable int16 -> f32 path without FP16 vector arithmetic
                    const int16x8_t _zip0 = vzip1q_s16(vsum_lo, vsum_hi);
                    const int16x8_t _zip1 = vzip2q_s16(vsum_lo, vsum_hi);
                    const int32x4_t i0_lo = vmovl_s16(vget_low_s16(_zip0));
                    const int32x4_t i0_hi = vmovl_s16(vget_high_s16(_zip0));
                    const int32x4_t i1_lo = vmovl_s16(vget_low_s16(_zip1));
                    const int32x4_t i1_hi = vmovl_s16(vget_high_s16(_zip1));
                    const float32x4_t vpartialsum_0 = vcvtq_f32_s32(i0_lo);
                    const float32x4_t vpartialsum_1 = vcvtq_f32_s32(i0_hi);
                    const float32x4_t vpartialsum_2 = vcvtq_f32_s32(i1_lo);
                    const float32x4_t vpartialsum_3 = vcvtq_f32_s32(i1_hi);
#endif

                    float32x4_t vsum_f32_0 = vld1q_f32(sums + nr_idx);
                    float32x4_t vsum_f32_1 = vld1q_f32(sums + nr_idx + 4);
                    float32x4_t vsum_f32_2 = vld1q_f32(sums + nr_idx + 8);
                    float32x4_t vsum_f32_3 = vld1q_f32(sums + nr_idx + 12);

                    vsum_f32_0 = vfmaq_f32(vsum_f32_0, vpartialsum_0, vd_0);
                    vsum_f32_1 = vfmaq_f32(vsum_f32_1, vpartialsum_1, vd_1);
                    vsum_f32_2 = vfmaq_f32(vsum_f32_2, vpartialsum_2, vd_2);
                    vsum_f32_3 = vfmaq_f32(vsum_f32_3, vpartialsum_3, vd_3);

                    vst1q_f32(sums + nr_idx, vsum_f32_0);
                    vst1q_f32(sums + nr_idx + 4, vsum_f32_1);
                    vst1q_f32(sums + nr_idx + 8, vsum_f32_2);
                    vst1q_f32(sums + nr_idx + 12, vsum_f32_3);

                    const uint8x8_t vdst_u8_0 = vorr_u8(
                        vand_u8(vreinterpret_u8_s8(vsrc0_lo), low_mask), vshl_n_u8(vreinterpret_u8_s8(vsrc1_lo), 4));
                    const uint8x8_t vdst_u8_1 = vorr_u8(
                        vand_u8(vreinterpret_u8_s8(vsrc2_lo), low_mask), vshl_n_u8(vreinterpret_u8_s8(vsrc3_lo), 4));
                    const uint8x8_t vdst_u8_2 = vorr_u8(
                        vand_u8(vreinterpret_u8_s8(vsrc0_hi), low_mask), vshl_n_u8(vreinterpret_u8_s8(vsrc1_hi), 4));
                    const uint8x8_t vdst_u8_3 = vorr_u8(
                        vand_u8(vreinterpret_u8_s8(vsrc2_hi), low_mask), vshl_n_u8(vreinterpret_u8_s8(vsrc3_hi), 4));

                    const uint16x8_t vdst_u16_even = vreinterpretq_u16_u8(
                        vcombine_u8(vzip1_u8(vdst_u8_0, vdst_u8_1), vzip2_u8(vdst_u8_0, vdst_u8_1)));
                    const uint16x8_t vdst_u16_odd = vreinterpretq_u16_u8(
                        vcombine_u8(vzip1_u8(vdst_u8_2, vdst_u8_3), vzip2_u8(vdst_u8_2, vdst_u8_3)));

                    const uint16x8_t vdst_0 = vzip1q_u16(vdst_u16_even, vdst_u16_odd);
                    const uint16x8_t vdst_1 = vzip2q_u16(vdst_u16_even, vdst_u16_odd);

                    vst1q_u16((uint16_t*)dst_row, vdst_0);
                    vst1q_u16((uint16_t*)(dst_row + 8 * block_length_in_bytes), vdst_1);

                    dst_row += (16 * block_length_in_bytes);
                }
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
