//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64 and FEAT_FP16.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qsi8d32pscalef32_f16_neon.h"

#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#define FLT16_MAX 65504.0F
#define FLT16_MIN (-65504.0F)
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
size_t kai_get_m_step_lhs_quant_pack_qsi8d32pscalef32_f16_neon(size_t mr) {
    return mr;
}
size_t kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_UNUSED(sr);
    return (m_idx / mr) * kai_get_lhs_packed_stride(k, mr, kr, bl);
}
size_t kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return (num_rows * kai_get_lhs_packed_stride(k, mr, kr, bl));
}
void kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride, void* lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);

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
        const float16_t* row_src_ptr = (const float16_t*)((const uint8_t*)lhs + (row_idx + m_idx_start) * lhs_stride);
        const size_t dst_x = ((row_idx + m_idx_start) % mr);
        for (size_t b = 0; b < num_blocks_per_row; ++b) {
            const float16_t* src_ptr = row_src_ptr + b * bl;
            int8_t* dst_ptr = (int8_t*)lhs_packed + dst_x * k_block_len * sizeof(int8_t) + b * mr_block_size;
            int8_t* param_ptr = (int8_t*)lhs_packed + b * mr_block_size + bl * mr + dst_x * kai_num_bytes_sum;
            // Find absmax for each block
            float16_t absmax = (float16_t)(-FLT16_MAX);
            int32_t k_idx = 0;
            float16x8_t vabsmax = vdupq_n_f16(-FLT16_MAX);
            for (; k_idx < ((int32_t)bl); k_idx += 8) {
                const float16x8_t src = vabsq_f16(vld1q_f16(src_ptr + (size_t)k_idx));
                vabsmax = vmaxq_f16(vabsmax, src);
            }
            // Get the absmax
            absmax = vmaxvq_f16(vabsmax);
            // Maximum/minimum int8 values
            const float qmax = (float)INT8_MAX;
            const float scale0 = absmax == 0.0F ? 0.0F : qmax / absmax;
            // Reciprocal to quantize
            const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;
            int32_t qsum = 0;
            // Quantize the blocks
            for (k_idx = 0; k_idx < (int32_t)bl; k_idx += k_block_len) {
                for (size_t k_block_idx = 0; k_block_idx < (size_t)k_block_len; ++k_block_idx) {
                    // Clamp at the last valid k-index
                    const size_t k_idx_start = KAI_MIN((size_t)k_idx + k_block_idx, k - 1);

                    const float16_t src0_0 = *(src_ptr + k_idx_start);

                    // Scale the values
                    int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

                    v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                    v0_s32 = KAI_MIN(v0_s32, INT8_MAX);
                    qsum += v0_s32;

                    *(dst_ptr) = (int8_t)v0_s32;
                    dst_ptr += sizeof(int8_t);
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
