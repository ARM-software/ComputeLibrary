//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qsi8d32p_f32_neon.h"

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

size_t kai_get_m_step_lhs_quant_pack_qsi8d32p_f32_neon(size_t mr) {
    KAI_UNUSED(mr);
    return 1;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32_neon(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % mr) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    // The scales are stored after all the mr packed quantized values
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, bl);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return (num_rows * kai_lhs_packed_stride(k, mr, kr, bl));
}

void kai_run_lhs_quant_pack_qsi8d32p_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed) {
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME(kr == 4);
    KAI_ASSUME(bl == 32);
    KAI_UNUSED(sr);
    KAI_UNUSED(m_idx_start);
    KAI_UNUSED(lhs_stride);

    if (m == 0) {
        return;
    }

    const size_t num_blocks = kai_num_blocks_per_row(k, bl);
    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, mr, kr, bl);

    const float* lhs_ptr = lhs;
    int8_t* lhs_packed_start_ptr = lhs_packed;

    for (size_t m_idx = 0; m_idx < m; m_idx++) {
        int8_t* lhs_packed_ptr = lhs_packed_start_ptr;
        uint16_t* lhs_packed_scales =
            (uint16_t*)(lhs_packed_ptr + lhs_packed_stride - ((mr * num_blocks) * kai_num_bytes_multiplier));

        lhs_packed_ptr += (m_idx % mr) * kr;
        lhs_packed_scales += (m_idx % mr);

        for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
            // Maximum absolute value of the block elements
            float amax = 0.0F;

            for (size_t bl_idx = 0; bl_idx < bl; bl_idx++) {
                amax = KAI_MAX(amax, fabsf(lhs_ptr[bl_idx]));
            }

            const float sf = amax / ((1 << 7) - 1);

            const float sf_inv = sf ? 1.0F / sf : 0.0F;

            for (size_t bl_idx = 0; bl_idx < bl; bl_idx += kr) {
                for (size_t kr_idx = 0; kr_idx < kr; ++kr_idx) {
                    int32_t v0_s32 = (int32_t)(roundf(lhs_ptr[kr_idx] * sf_inv));
                    lhs_packed_ptr[kr_idx] = (int8_t)v0_s32;
                }
                lhs_ptr += kr;
                lhs_packed_ptr += mr * kr;
            }

            // Num_blocks (rows) x Mr (cols)
            lhs_packed_scales[0] = kai_cast_f16_f32(sf);

            lhs_packed_scales += mr;
        }
        if (((m_idx + 1) % mr) == 0) {
            lhs_packed_start_ptr += lhs_packed_stride;
        }
    }
}
#endif  // Architectural features check.
