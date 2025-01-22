//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_lhs_quant_pack_qsi8d32p_f32.h"

#include <math.h>
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

size_t kai_get_m_step_lhs_quant_pack_qsi8d32p_f32(size_t mr) {
    KAI_UNUSED(mr);
    return 1;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, bl);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, bl);
}

void kai_run_lhs_quant_pack_qsi8d32p_f32(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed) {
    if (m == 0) {
        return;
    }

    const size_t num_rows = m;
    const size_t k_block_len = kr / sr;
    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, mr, kr, bl);
    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const float* src_ptr = (const float*)((const uint8_t*)lhs + (row_idx + m_idx_start) * lhs_stride);

        for (size_t b = 0; b < num_blocks_per_row; ++b) {
            float abs_max = 0.0F;

            const size_t dst_x = ((row_idx + m_idx_start) % mr);
            int8_t* dst_ptr = (int8_t*)lhs_packed + (b * mr) * num_bytes_per_block;

            for (size_t idx_v = 0; idx_v < bl; ++idx_v) {
                const float val = src_ptr[idx_v];
                abs_max = KAI_MAX(abs_max, fabsf(val));
            }

            // Calculate scale and reciprocal
            const float scale = abs_max / ((1 << 7) - 1);
            const float rep_scale = scale ? 1.0F / scale : 0.0F;

            *((uint16_t*)(dst_ptr + dst_x * kai_num_bytes_multiplier)) = kai_cast_f16_f32(scale);
            dst_ptr += mr * kai_num_bytes_multiplier;

            dst_ptr += dst_x * k_block_len * sizeof(int8_t);

            // Quantize and pack the block
            for (size_t k_idx = 0; k_idx < bl; k_idx += k_block_len) {
                for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    // Clamp at the last valid k-index
                    const size_t k_idx_start = KAI_MIN(k_idx + k_block_idx, k - 1);

                    const float src0_0 = *(src_ptr + k_idx_start);

                    // Scale the values
                    int32_t v0_s32 = (int32_t)(roundf(src0_0 * rep_scale));

                    *dst_ptr = (int8_t)v0_s32;
                    dst_ptr += sizeof(int8_t);
                }
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }

            src_ptr += bl;
        }
        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((int8_t*)lhs_packed + lhs_packed_stride);
        }
    }
}
