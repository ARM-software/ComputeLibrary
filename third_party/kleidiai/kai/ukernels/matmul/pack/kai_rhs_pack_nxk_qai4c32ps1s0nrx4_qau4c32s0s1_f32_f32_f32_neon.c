//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.
#include "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_offset_rhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_bl_multiple_of = 32;
static const size_t kai_nr_multiple_of = 4;

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_num_bytes_per_block(size_t bl) {
    return (bl / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_offset_rhs;
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl);
    return nr * (num_bytes_per_block * num_blocks_per_row + kai_num_bytes_bias);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);
    KAI_UNUSED(kr);
    return (n_idx / nr) * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_UNUSED(kr);
    const size_t num_rows = kai_roundup(n, nr) / nr;
    return num_rows * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

void kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const void* zero, const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qai4c32p_params* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    KAI_ASSUME((nr % kai_nr_multiple_of) == 0);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr / sr == 4);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(zero != NULL);
    KAI_ASSUME(scale != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t block_length = kr / sr;
    const size_t num_blocks_per_row = k / bl;
    const size_t rhs_stride = k / 2;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride(k, nr, kr, bl);

    const size_t dst_packed_block_size = kai_get_num_bytes_per_block(bl) * nr;
    const size_t dst_block_data_size = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_bias_offset = num_blocks_per_row * dst_packed_block_size;
    const size_t k_block_length_in_bytes = (block_length * sizeof(uint8_t)) / 2;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;
        float* dst_row_bias = (float*)(dst_row + dst_bias_offset);
        size_t row_idx = dst_row_idx * nr;
        size_t rows_left = n - row_idx;
        for (size_t block_idx = 0; block_idx < num_blocks_per_row; block_idx++) {
            uint8_t* block_dst_row = dst_row + block_idx * dst_packed_block_size;
            float* block_dst_zp = (float*)(block_dst_row + nr * dst_block_data_size);
            float* block_dst_scale = block_dst_zp + nr;
            size_t k_idx = block_idx * bl;
            for (size_t dst_byte_idx = 0; dst_byte_idx < dst_block_data_size; dst_byte_idx += 8) {
                for (size_t nr_idx = 0; nr_idx <= nr - 4; nr_idx += 4) {
                    const size_t n0_idx = KAI_MIN(dst_row_idx * nr + nr_idx, n - 1);
                    const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                    const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                    const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);
                    const uint8_t* src_addr_byte = rhs + (k_idx / 2) + dst_byte_idx;

                    const uint8x8_t vec0_u8 = vld1_u8(src_addr_byte + n0_idx * rhs_stride);
                    const uint8x8_t vec1_u8 = vld1_u8(src_addr_byte + n1_idx * rhs_stride);
                    const uint8x8_t vec2_u8 = vld1_u8(src_addr_byte + n2_idx * rhs_stride);
                    const uint8x8_t vec3_u8 = vld1_u8(src_addr_byte + n3_idx * rhs_stride);

                    const uint16x4_t vec0_u16 = vreinterpret_u16_u8(vec0_u8);
                    const uint16x4_t vec1_u16 = vreinterpret_u16_u8(vec1_u8);
                    const uint16x4_t vec2_u16 = vreinterpret_u16_u8(vec2_u8);
                    const uint16x4_t vec3_u16 = vreinterpret_u16_u8(vec3_u8);

                    const uint16x4_t vec01_lo_u16 = vzip1_u16(vec0_u16, vec1_u16);
                    const uint16x4_t vec01_hi_u16 = vzip2_u16(vec0_u16, vec1_u16);
                    const uint16x4_t vec23_lo_u16 = vzip1_u16(vec2_u16, vec3_u16);
                    const uint16x4_t vec23_hi_u16 = vzip2_u16(vec2_u16, vec3_u16);

                    const uint32x2_t vec01_lo_u32 = vreinterpret_u32_u16(vec01_lo_u16);
                    const uint32x2_t vec01_hi_u32 = vreinterpret_u32_u16(vec01_hi_u16);
                    const uint32x2_t vec23_lo_u32 = vreinterpret_u32_u16(vec23_lo_u16);
                    const uint32x2_t vec23_hi_u32 = vreinterpret_u32_u16(vec23_hi_u16);

                    const uint32x2_t vin0_u32 = vzip1_u32(vec01_lo_u32, vec23_lo_u32);
                    const uint32x2_t vin1_u32 = vzip2_u32(vec01_lo_u32, vec23_lo_u32);
                    const uint32x2_t vin2_u32 = vzip1_u32(vec01_hi_u32, vec23_hi_u32);
                    const uint32x2_t vin3_u32 = vzip2_u32(vec01_hi_u32, vec23_hi_u32);

                    uint8x8_t vin0_u8 = vreinterpret_u8_u32(vin0_u32);
                    uint8x8_t vin1_u8 = vreinterpret_u8_u32(vin1_u32);
                    uint8x8_t vin2_u8 = vreinterpret_u8_u32(vin2_u32);
                    uint8x8_t vin3_u8 = vreinterpret_u8_u32(vin3_u32);

                    const uint8x8_t vin0_s1s = vshr_n_u8(vin0_u8, 4);
                    const uint8x8_t vin1_s1s = vshr_n_u8(vin1_u8, 4);
                    const uint8x8_t vin2_s1s = vshr_n_u8(vin2_u8, 4);
                    const uint8x8_t vin3_s1s = vshr_n_u8(vin3_u8, 4);

                    vin0_u8 = vshl_n_u8(vin0_u8, 4);
                    vin1_u8 = vshl_n_u8(vin1_u8, 4);
                    vin2_u8 = vshl_n_u8(vin2_u8, 4);
                    vin3_u8 = vshl_n_u8(vin3_u8, 4);

                    vin0_u8 = vorr_u8(vin0_u8, vin0_s1s);
                    vin1_u8 = vorr_u8(vin1_u8, vin1_s1s);
                    vin2_u8 = vorr_u8(vin2_u8, vin2_s1s);
                    vin3_u8 = vorr_u8(vin3_u8, vin3_s1s);

                    uint8_t* dst_row_offset = block_dst_row + nr_idx * k_block_length_in_bytes;
                    vst1_u8(dst_row_offset, vin0_u8);
                    vst1_u8(dst_row_offset + nr * k_block_length_in_bytes, vin1_u8);
                    vst1_u8(dst_row_offset + 2 * (nr * k_block_length_in_bytes), vin2_u8);
                    vst1_u8(dst_row_offset + 3 * (nr * k_block_length_in_bytes), vin3_u8);
                }
                block_dst_row += nr * sizeof(uint8x8_t);
            }

            // Adjust the zero points and scales
            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(row_idx + i, n - 1);
                const size_t src_idx = src_row_idx * num_blocks_per_row + block_idx;

                block_dst_scale[i] = ((const float*)scale)[src_idx];
                block_dst_zp[i] = ((const float*)zero)[src_idx];
            }
        }
        // Set the bias
        if (bias == NULL) {
            memset(dst_row_bias, 0, nr * kai_num_bytes_bias);
        } else {
            if (rows_left >= nr) {
                memcpy(dst_row_bias, &((const float*)bias)[row_idx], nr * kai_num_bytes_bias);
            } else {
                // Fill remaining values
                memcpy(dst_row_bias, &((const float*)bias)[row_idx], rows_left * kai_num_bytes_bias);
                // Set leftover to 0
                memset(&dst_row_bias[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
            }
        }
    }
}
#endif
