//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);

inline static size_t kai_num_bytes_per_block(size_t bl) {
    return bl * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSERT((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t bl) {
    KAI_UNUSED(kr);
    return mr * kai_num_blocks_per_row(k, bl) * kai_num_bytes_per_block(bl);
}

size_t kai_get_m_step_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(size_t mr) {
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, bl);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, bl);
}

void kai_run_lhs_quant_pack_qsi8d32p4x8sb_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed) {
    if (m == 0) {
        return;
    }

    KAI_ASSUME(bl == 32);
    KAI_ASSUME(mr == 4);
    KAI_ASSUME(kr == 16);
    KAI_ASSUME(sr == 2);

    const size_t local_bl = 32;
    const size_t local_mr = 4;
    const size_t local_kr = 16;
    const size_t local_sr = 2;
    const size_t num_rows = m;
    const size_t k_block_len = local_kr / local_sr;
    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, local_mr, local_kr, local_bl);
    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, local_bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(local_bl);

    size_t row_idx = 0;

    const size_t write_mem_increment = 2 * k_block_len * sizeof(int8_t);
    const size_t read_mem_increment = num_blocks_per_row * local_bl * sizeof(int8_t);

    if (num_rows >= 4) {
        for (; row_idx + 4 <= num_rows; row_idx += 4) {
            const float* src_ptr = (const float*)((const uint8_t*)lhs + (row_idx + m_idx_start) * lhs_stride);

            for (size_t b = 0; b < num_blocks_per_row; ++b) {
                const size_t dst_x = ((row_idx + m_idx_start) % local_mr);
                int8_t* dst_ptr = (int8_t*)lhs_packed + (b * local_mr) * num_bytes_per_block;

                float abs_max_0 = 0.0F;
                float abs_max_1 = 0.0F;
                float abs_max_2 = 0.0F;
                float abs_max_3 = 0.0F;

                float32x4_t v_currentmax_0 = vdupq_n_f32(0);
                float32x4_t v_currentmax_1 = vdupq_n_f32(0);
                float32x4_t v_currentmax_2 = vdupq_n_f32(0);
                float32x4_t v_currentmax_3 = vdupq_n_f32(0);

                for (size_t idx_v = 0; idx_v < local_bl; idx_v += 4) {
                    const float32x4_t v_f32_maxvals_0 = vld1q_f32(src_ptr + idx_v);
                    const float32x4_t v_f32_abs_values_0 = vabsq_f32(v_f32_maxvals_0);
                    v_currentmax_0 = vmaxq_f32(v_f32_abs_values_0, v_currentmax_0);
                    const float32x4_t v_f32_maxvals_1 = vld1q_f32(src_ptr + idx_v + read_mem_increment);
                    const float32x4_t v_f32_abs_values_1 = vabsq_f32(v_f32_maxvals_1);
                    v_currentmax_1 = vmaxq_f32(v_f32_abs_values_1, v_currentmax_1);
                    const float32x4_t v_f32_maxvals_2 = vld1q_f32(src_ptr + idx_v + 2 * read_mem_increment);
                    const float32x4_t v_f32_abs_values_2 = vabsq_f32(v_f32_maxvals_2);
                    v_currentmax_2 = vmaxq_f32(v_f32_abs_values_2, v_currentmax_2);
                    const float32x4_t v_f32_maxvals_3 = vld1q_f32(src_ptr + idx_v + 3 * read_mem_increment);
                    const float32x4_t v_f32_abs_values_3 = vabsq_f32(v_f32_maxvals_3);
                    v_currentmax_3 = vmaxq_f32(v_f32_abs_values_3, v_currentmax_3);
                }

                abs_max_0 = vmaxvq_f32(v_currentmax_0);
                abs_max_1 = vmaxvq_f32(v_currentmax_1);
                abs_max_2 = vmaxvq_f32(v_currentmax_2);
                abs_max_3 = vmaxvq_f32(v_currentmax_3);

                float32x4_t abs_max_vec = vdupq_n_f32(abs_max_0);
                abs_max_vec = vsetq_lane_f32(abs_max_1, abs_max_vec, 1);
                abs_max_vec = vsetq_lane_f32(abs_max_2, abs_max_vec, 2);
                abs_max_vec = vsetq_lane_f32(abs_max_3, abs_max_vec, 3);

                // Calculate scale and reciprocals
                const float32x4_t scales = vdivq_f32(abs_max_vec, vdupq_n_f32((1 << 7) - 1));
                const uint32x4_t valid_scales = vmvnq_u32(vceqq_f32(scales, vdupq_n_f32(0.0F)));
                const float32x4_t reciprocals = vdivq_f32(vdupq_n_f32(1.0F), scales);
                const float32x4_t rep_scales = vbslq_f32(valid_scales, reciprocals, vdupq_n_f32(0.0F));
                const float16x4_t f16_scales = vcvt_f16_f32(scales);

                vst1_u16((uint16_t*)(dst_ptr + dst_x * kai_num_bytes_multiplier), vreinterpret_u16_f16(f16_scales));

                dst_ptr += local_mr * kai_num_bytes_multiplier;

                dst_ptr += dst_x * k_block_len * sizeof(int8_t);

                // Quantize and pack the blocks
                for (size_t k_idx = 0; k_idx < local_bl; k_idx += k_block_len * 2) {
                    // Row 1 blocks
                    const float32x4_t v_f32_block1 = vld1q_f32(src_ptr + k_idx);
                    const float32x4_t v_f32_sblock1 = vmulq_n_f32(v_f32_block1, vgetq_lane_f32(rep_scales, 0));
                    const int32x4_t v_i32_block1 = vcvtnq_s32_f32(v_f32_sblock1);

                    const float32x4_t v_f32_block2 = vld1q_f32(src_ptr + k_idx + 4);
                    const float32x4_t v_f32_sblock2 = vmulq_n_f32(v_f32_block2, vgetq_lane_f32(rep_scales, 0));
                    const int32x4_t v_i32_block2 = vcvtnq_s32_f32(v_f32_sblock2);

                    const int16x8_t v_full_i16_block1 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block1), vreinterpretq_s16_s32(v_i32_block2));

                    const float32x4_t v_f32_block3 = vld1q_f32(src_ptr + k_idx + 8);
                    const float32x4_t v_f32_sblock3 = vmulq_n_f32(v_f32_block3, vgetq_lane_f32(rep_scales, 0));
                    const int32x4_t v_i32_block3 = vcvtnq_s32_f32(v_f32_sblock3);

                    const float32x4_t v_f32_block4 = vld1q_f32(src_ptr + k_idx + 12);
                    const float32x4_t v_f32_sblock4 = vmulq_n_f32(v_f32_block4, vgetq_lane_f32(rep_scales, 0));
                    const int32x4_t v_i32_block4 = vcvtnq_s32_f32(v_f32_sblock4);

                    const int16x8_t v_full_i16_block2 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block3), vreinterpretq_s16_s32(v_i32_block4));

                    // Row 2 blocks
                    const float32x4_t v_f32_block5 = vld1q_f32(src_ptr + k_idx + read_mem_increment);
                    const float32x4_t v_f32_sblock5 = vmulq_n_f32(v_f32_block5, vgetq_lane_f32(rep_scales, 1));
                    const int32x4_t v_i32_block5 = vcvtnq_s32_f32(v_f32_sblock5);

                    const float32x4_t v_f32_block6 = vld1q_f32(src_ptr + k_idx + 4 + read_mem_increment);
                    const float32x4_t v_f32_sblock6 = vmulq_n_f32(v_f32_block6, vgetq_lane_f32(rep_scales, 1));
                    const int32x4_t v_i32_block6 = vcvtnq_s32_f32(v_f32_sblock6);

                    const int16x8_t v_full_i16_block3 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block5), vreinterpretq_s16_s32(v_i32_block6));

                    const float32x4_t v_f32_block7 = vld1q_f32(src_ptr + k_idx + 8 + read_mem_increment);
                    const float32x4_t v_f32_sblock7 = vmulq_n_f32(v_f32_block7, vgetq_lane_f32(rep_scales, 1));
                    const int32x4_t v_i32_block7 = vcvtnq_s32_f32(v_f32_sblock7);

                    const float32x4_t v_f32_block8 = vld1q_f32(src_ptr + k_idx + 12 + read_mem_increment);
                    const float32x4_t v_f32_sblock8 = vmulq_n_f32(v_f32_block8, vgetq_lane_f32(rep_scales, 1));
                    const int32x4_t v_i32_block8 = vcvtnq_s32_f32(v_f32_sblock8);

                    const int16x8_t v_full_i16_block4 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block7), vreinterpretq_s16_s32(v_i32_block8));

                    // Row 3 blocks
                    const float32x4_t v_f32_block9 = vld1q_f32(src_ptr + k_idx + 2 * read_mem_increment);
                    const float32x4_t v_f32_sblock9 = vmulq_n_f32(v_f32_block9, vgetq_lane_f32(rep_scales, 2));
                    const int32x4_t v_i32_block9 = vcvtnq_s32_f32(v_f32_sblock9);

                    const float32x4_t v_f32_blockA = vld1q_f32(src_ptr + k_idx + 4 + 2 * read_mem_increment);
                    const float32x4_t v_f32_sblockA = vmulq_n_f32(v_f32_blockA, vgetq_lane_f32(rep_scales, 2));
                    const int32x4_t v_i32_blockA = vcvtnq_s32_f32(v_f32_sblockA);

                    const int16x8_t v_full_i16_block5 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block9), vreinterpretq_s16_s32(v_i32_blockA));

                    const float32x4_t v_f32_blockB = vld1q_f32(src_ptr + k_idx + 8 + 2 * read_mem_increment);
                    const float32x4_t v_f32_sblockB = vmulq_n_f32(v_f32_blockB, vgetq_lane_f32(rep_scales, 2));
                    const int32x4_t v_i32_blockB = vcvtnq_s32_f32(v_f32_sblockB);

                    const float32x4_t v_f32_blockC = vld1q_f32(src_ptr + k_idx + 12 + 2 * read_mem_increment);
                    const float32x4_t v_f32_sblockC = vmulq_n_f32(v_f32_blockC, vgetq_lane_f32(rep_scales, 2));
                    const int32x4_t v_i32_blockC = vcvtnq_s32_f32(v_f32_sblockC);

                    const int16x8_t v_full_i16_block6 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_blockB), vreinterpretq_s16_s32(v_i32_blockC));

                    // Row 4 blocks
                    const float32x4_t v_f32_blockD = vld1q_f32(src_ptr + k_idx + 3 * read_mem_increment);
                    const float32x4_t v_f32_sblockD = vmulq_n_f32(v_f32_blockD, vgetq_lane_f32(rep_scales, 3));
                    const int32x4_t v_i32_blockD = vcvtnq_s32_f32(v_f32_sblockD);

                    const float32x4_t v_f32_blockE = vld1q_f32(src_ptr + k_idx + 4 + 3 * read_mem_increment);
                    const float32x4_t v_f32_sblockE = vmulq_n_f32(v_f32_blockE, vgetq_lane_f32(rep_scales, 3));
                    const int32x4_t v_i32_blockE = vcvtnq_s32_f32(v_f32_sblockE);

                    const int16x8_t v_full_i16_block7 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_blockD), vreinterpretq_s16_s32(v_i32_blockE));

                    const float32x4_t v_f32_blockF = vld1q_f32(src_ptr + k_idx + 8 + 3 * read_mem_increment);
                    const float32x4_t v_f32_sblockF = vmulq_n_f32(v_f32_blockF, vgetq_lane_f32(rep_scales, 3));
                    const int32x4_t v_i32_blockF = vcvtnq_s32_f32(v_f32_sblockF);

                    const float32x4_t v_f32_block0 = vld1q_f32(src_ptr + k_idx + 12 + 3 * read_mem_increment);
                    const float32x4_t v_f32_sblock0 = vmulq_n_f32(v_f32_block0, vgetq_lane_f32(rep_scales, 3));
                    const int32x4_t v_i32_block0 = vcvtnq_s32_f32(v_f32_sblock0);

                    const int16x8_t v_full_i16_block8 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_blockF), vreinterpretq_s16_s32(v_i32_block0));

                    const int8x16_t v_i8_block1_3 =
                        vuzp1q_s8(vreinterpretq_s8_s16(v_full_i16_block1), vreinterpretq_s8_s16(v_full_i16_block3));
                    vst1q_s8(dst_ptr, v_i8_block1_3);
                    dst_ptr += write_mem_increment;

                    const int8x16_t v_i8_block5_7 =
                        vuzp1q_s8(vreinterpretq_s8_s16(v_full_i16_block5), vreinterpretq_s8_s16(v_full_i16_block7));
                    vst1q_s8(dst_ptr, v_i8_block5_7);
                    dst_ptr += write_mem_increment;

                    const int8x16_t v_i8_block2_4 =
                        vuzp1q_s8(vreinterpretq_s8_s16(v_full_i16_block2), vreinterpretq_s8_s16(v_full_i16_block4));
                    vst1q_s8(dst_ptr, v_i8_block2_4);
                    dst_ptr += write_mem_increment;

                    const int8x16_t v_i8_block6_8 =
                        vuzp1q_s8(vreinterpretq_s8_s16(v_full_i16_block6), vreinterpretq_s8_s16(v_full_i16_block8));
                    vst1q_s8(dst_ptr, v_i8_block6_8);
                    dst_ptr += write_mem_increment;
                }
                src_ptr += local_bl;
            }
            lhs_packed = (void*)((int8_t*)lhs_packed + lhs_packed_stride);
        }
    }
    if (num_rows % 4 != 0) {
        for (; row_idx < num_rows; ++row_idx) {
            const float* src_ptr = (const float*)((const uint8_t*)lhs + (row_idx + m_idx_start) * lhs_stride);

            for (size_t b = 0; b < num_blocks_per_row; ++b) {
                float abs_max = 0.0F;

                const size_t dst_x = ((row_idx + m_idx_start) % local_mr);
                int8_t* dst_ptr = (int8_t*)lhs_packed + (b * local_mr) * num_bytes_per_block;

                float32x4_t v_f32_abs_values;
                float32x4_t v_f32_maxvals;
                float32x4_t v_currentmax = vdupq_n_f32(0);

                for (size_t idx_v = 0; idx_v < local_bl; idx_v += 4) {
                    v_f32_maxvals = vld1q_f32(src_ptr + idx_v);
                    v_f32_abs_values = vabsq_f32(v_f32_maxvals);
                    v_currentmax = vmaxq_f32(v_f32_abs_values, v_currentmax);
                }
                abs_max = vmaxvq_f32(v_currentmax);

                // Calculate scale and reciprocal
                const float scale = abs_max / ((1 << 7) - 1);
                const float rep_scale = scale ? 1.0F / scale : 0.0F;

                *((uint16_t*)(dst_ptr + dst_x * kai_num_bytes_multiplier)) = kai_cast_f16_f32(scale);
                dst_ptr += local_mr * kai_num_bytes_multiplier;

                dst_ptr += dst_x * k_block_len * sizeof(int8_t);

                // Quantize and pack the block
                for (size_t k_idx = 0; k_idx < local_bl; k_idx += k_block_len * 2) {
                    const float32x4_t v_f32_block1 = vld1q_f32(src_ptr + k_idx);
                    const float32x4_t v_f32_sblock1 = vmulq_n_f32(v_f32_block1, rep_scale);
                    const int32x4_t v_i32_block1 = vcvtnq_s32_f32(v_f32_sblock1);

                    const float32x4_t v_f32_block2 = vld1q_f32(src_ptr + k_idx + 4);
                    const float32x4_t v_f32_sblock2 = vmulq_n_f32(v_f32_block2, rep_scale);
                    const int32x4_t v_i32_block2 = vcvtnq_s32_f32(v_f32_sblock2);

                    const int16x8_t v_full_i16_block1 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block1), vreinterpretq_s16_s32(v_i32_block2));

                    const float32x4_t v_f32_block3 = vld1q_f32(src_ptr + k_idx + 8);
                    const float32x4_t v_f32_sblock3 = vmulq_n_f32(v_f32_block3, rep_scale);
                    const int32x4_t v_i32_block3 = vcvtnq_s32_f32(v_f32_sblock3);

                    const float32x4_t v_f32_block4 = vld1q_f32(src_ptr + k_idx + 12);
                    const float32x4_t v_f32_sblock4 = vmulq_n_f32(v_f32_block4, rep_scale);
                    const int32x4_t v_i32_block4 = vcvtnq_s32_f32(v_f32_sblock4);

                    const int16x8_t v_full_i16_block2 =
                        vuzp1q_s16(vreinterpretq_s16_s32(v_i32_block3), vreinterpretq_s16_s32(v_i32_block4));

                    const int8x16_t v_full_i8_block =
                        vuzp1q_s8(vreinterpretq_s8_s16(v_full_i16_block1), vreinterpretq_s8_s16(v_full_i16_block2));

                    vst1_s8(dst_ptr, vget_low_s8(v_full_i8_block));
                    dst_ptr += 8 * sizeof(int8_t);
                    dst_ptr += (local_mr - 1) * k_block_len * sizeof(int8_t);

                    vst1_s8(dst_ptr, vget_high_s8(v_full_i8_block));
                    dst_ptr += 8 * sizeof(int8_t);
                    dst_ptr += (local_mr - 1) * k_block_len * sizeof(int8_t);
                }
                src_ptr += local_bl;
            }
            // Move to the next row if we have interleaved all Mr rows
            if ((((row_idx + 1) + m_idx_start) % local_mr) == 0) {
                lhs_packed = (void*)((int8_t*)lhs_packed + lhs_packed_stride);
            }
        }
    }
}
