//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"

#include <arm_neon.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum = sizeof(float);
static const size_t kai_num_bytes_multiplier = sizeof(float);
static const size_t kai_bl_multiple_of = 32;

inline static size_t kai_get_num_bytes_per_block(size_t bl) {
    return bl * sizeof(int8_t) + kai_num_bytes_multiplier + kai_num_bytes_sum;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSERT((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t bl) {
    KAI_UNUSED(kr);

    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block(bl);
}

size_t kai_get_m_step_lhs_quant_pack_qsi8d32pscalef32_f32_neon(size_t mr) {
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % mr) == 0);

    KAI_UNUSED(sr);
    return (m_idx / mr) * kai_get_lhs_packed_stride(k, mr, kr, bl);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return (num_rows * kai_get_lhs_packed_stride(k, mr, kr, bl));
}
void kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(((kr / sr) == 8) || ((kr / sr) == 4));

    if (m == 0) {
        return;
    }
    const size_t lhs_packed_stride = kai_get_lhs_packed_stride(k, mr, kr, bl);
    const size_t num_rows = m;
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl);
    const size_t mr_block_size = mr * num_bytes_per_block;

    const int32_t k_block_len = (int32_t)(kr / sr);

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const float* row_src_ptr = (const float*)((const uint8_t*)lhs + (row_idx + m_idx_start) * lhs_stride);
        const size_t dst_idx = ((row_idx + m_idx_start) % mr);

        for (size_t blk_idx = 0; blk_idx < num_blocks_per_row; ++blk_idx) {
            const float* src_ptr = row_src_ptr + blk_idx * bl;
            int8_t* dst_ptr = (int8_t*)lhs_packed + dst_idx * k_block_len * sizeof(int8_t) + blk_idx * mr_block_size;
            int8_t* param_ptr = (int8_t*)lhs_packed + blk_idx * mr_block_size + bl * mr + dst_idx * kai_num_bytes_sum;

            // Find absmax for each block
            float absmax = -FLT_MAX;
            int32_t k_idx = 0;
            float32x4_t vabsmax = vdupq_n_f32(-FLT_MAX);
            for (; k_idx < ((int32_t)bl); k_idx += 8) {
                const float32x4_t src0_0 = vld1q_f32(src_ptr + 0 + (size_t)k_idx);
                const float32x4_t src0_1 = vld1q_f32(src_ptr + 4 + (size_t)k_idx);
                // Calculate the max
                vabsmax = vmaxq_f32(vabsq_f32(src0_0), vmaxq_f32(vabsmax, vabsq_f32(src0_1)));
            }
            // Get the absmax
            absmax = vmaxvq_f32(vabsmax);

            // Maximum/minimum int8 values
            const float qmax = (float)INT8_MAX;

            // Get the scale and reciprocal to quantize
            const float scale0 = absmax == 0.0F ? 0.0F : qmax / absmax;
            const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;

            int32_t qsum = 0;
            // Quantize the blocks
            for (k_idx = 0; k_idx <= (int32_t)bl - k_block_len; k_idx += k_block_len) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = KAI_MIN((size_t)k_idx, k - 1);
                if (k_block_len == 8) {
                    const float32x4_t vsrc_0 = vld1q_f32(src_ptr + k_idx_start);
                    const float32x4_t vsrc_1 = vld1q_f32(src_ptr + k_idx_start + 4);

                    // Scale the values
                    float32x4_t v0_f32 = vmulq_n_f32(vsrc_0, scale0);
                    float32x4_t v1_f32 = vmulq_n_f32(vsrc_1, scale0);

                    int16x4_t v0_s16 = vqmovn_s32(vcvtnq_s32_f32(v0_f32));
                    int16x4_t v1_s16 = vqmovn_s32(vcvtnq_s32_f32(v1_f32));
                    int16x8_t v_s16 = vcombine_s16(v0_s16, v1_s16);

                    v_s16 = vmaxq_s16(v_s16, vdupq_n_s16(INT8_MIN));
                    v_s16 = vminq_s16(v_s16, vdupq_n_s16(INT8_MAX));

                    // Update the sum
                    qsum += vaddvq_s16(v_s16);

                    int8x8_t v0_s8 = vqmovn_s16(v_s16);
                    vst1_s8(dst_ptr, v0_s8);
                    dst_ptr += 8 * sizeof(int8_t);
                } else if (k_block_len == 4) {
                    const float32x2_t vsrc_0 = vld1_f32(src_ptr + k_idx_start);
                    const float32x2_t vsrc_1 = vld1_f32(src_ptr + k_idx_start + 2);

                    // Scale the values
                    float32x2_t v0_f32 = vmul_n_f32(vsrc_0, scale0);
                    float32x2_t v1_f32 = vmul_n_f32(vsrc_1, scale0);

                    int32x2_t v0_s32 = vcvtn_s32_f32(v0_f32);
                    int32x2_t v1_s32 = vcvtn_s32_f32(v1_f32);
                    int16x4_t v_s16 = vqmovn_s32(vcombine_s32(v0_s32, v1_s32));

                    v_s16 = vmax_s16(v_s16, vdup_n_s16(INT8_MIN));
                    v_s16 = vmin_s16(v_s16, vdup_n_s16(INT8_MAX));

                    // Update the sum
                    qsum += vaddv_s16(v_s16);

                    dst_ptr[0] = vqmovnh_s16(vget_lane_s16(v_s16, 0));
                    dst_ptr[1] = vqmovnh_s16(vget_lane_s16(v_s16, 1));
                    dst_ptr[2] = vqmovnh_s16(vget_lane_s16(v_s16, 2));
                    dst_ptr[3] = vqmovnh_s16(vget_lane_s16(v_s16, 3));
                    dst_ptr += 4 * sizeof(int8_t);
                }
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }
            *((float*)(param_ptr)) = ((float)qsum) * recip_scale0;
            param_ptr += mr * kai_num_bytes_sum;
            *((float*)(param_ptr)) = recip_scale0;
        }
        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((int8_t*)lhs_packed + lhs_packed_stride);
        }
    }
}
#endif  // Architectural features check.
