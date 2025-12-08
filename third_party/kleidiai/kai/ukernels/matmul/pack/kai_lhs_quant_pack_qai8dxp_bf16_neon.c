//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if (!defined(__aarch64__) && !defined(_M_ARM64))
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qai8dxp_bf16_neon.h"

#include <arm_neon.h>
#endif
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    static const size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return mr * (k_internal * sizeof(int8_t) + kai_num_bytes_per_multiplier + kai_num_bytes_per_offset);
}

size_t kai_get_m_step_lhs_quant_pack_qai8dxp_bf16_neon(size_t mr) {
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    // It always points to the beginning of the row
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr);
}

// Note: The lhs parameter type has been changed from float* to void*.
// The bfloat16 values (packed in 16 bits) will be converted to float32.
void kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);

    if (m == 0) {
        return;
    }

    // Now lhs is assumed to contain bfloat16 values encoded in uint16_t.
    const uint16_t* src_ptr = (uint16_t const*)lhs;

    const size_t dst_stride = kai_lhs_packed_stride(k, mr);
    const size_t k_internal = kai_k_roundedup(k);
    const int32_t k_block_len = (int32_t)(kr / sr);
    KAI_ASSERT(k_block_len == 8);

    const int32_t num_blocks_k = (int32_t)(k / k_block_len);
    const int32_t num_blocks_k_internal = (int32_t)(k_internal / k_block_len);

    size_t row_idx = 0;

    if (mr == 4) {
        for (; row_idx + 3 < m; row_idx += 4) {
            float max0 = -FLT_MAX;
            float min0 = FLT_MAX;
            float max1 = -FLT_MAX;
            float min1 = FLT_MAX;
            float max2 = -FLT_MAX;
            float min2 = FLT_MAX;
            float max3 = -FLT_MAX;
            float min3 = FLT_MAX;

            // Find min/max for each channel
            int32_t k_idx = 0;
            float32x4_t vmax0 = vdupq_n_f32(-FLT_MAX);
            float32x4_t vmin0 = vdupq_n_f32(FLT_MAX);
            float32x4_t vmax1 = vmax0;
            float32x4_t vmin1 = vmin0;
            float32x4_t vmax2 = vmax0;
            float32x4_t vmin2 = vmin0;
            float32x4_t vmax3 = vmax0;
            float32x4_t vmin3 = vmin0;
            const uint16x8_t zero = vdupq_n_u16(0);
            // Process 8 bfloat16 values per iteration.
            for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
                // Load eight bfloat16 values.
                const uint16x8_t bf16_vec_0 = vld1q_u16(src_ptr + k_idx);
                const uint16x8_t bf16_vec_1 = vld1q_u16(src_ptr + k_idx + (lhs_stride / sizeof(uint16_t)));
                const uint16x8_t bf16_vec_2 = vld1q_u16(src_ptr + k_idx + (2 * (lhs_stride / sizeof(uint16_t))));
                const uint16x8_t bf16_vec_3 = vld1q_u16(src_ptr + k_idx + (3 * (lhs_stride / sizeof(uint16_t))));

                const uint16x8_t bf16_vec1_0 = vzip1q_u16(zero, bf16_vec_0);
                const uint16x8_t bf16_vec2_0 = vzip2q_u16(zero, bf16_vec_0);
                const uint16x8_t bf16_vec1_1 = vzip1q_u16(zero, bf16_vec_1);
                const uint16x8_t bf16_vec2_1 = vzip2q_u16(zero, bf16_vec_1);
                const uint16x8_t bf16_vec1_2 = vzip1q_u16(zero, bf16_vec_2);
                const uint16x8_t bf16_vec2_2 = vzip2q_u16(zero, bf16_vec_2);
                const uint16x8_t bf16_vec1_3 = vzip1q_u16(zero, bf16_vec_3);
                const uint16x8_t bf16_vec2_3 = vzip2q_u16(zero, bf16_vec_3);

                const float32x4_t src0_0 = vreinterpretq_f32_u16(bf16_vec1_0);
                const float32x4_t src0_1 = vreinterpretq_f32_u16(bf16_vec2_0);
                const float32x4_t src1_0 = vreinterpretq_f32_u16(bf16_vec1_1);
                const float32x4_t src1_1 = vreinterpretq_f32_u16(bf16_vec2_1);
                const float32x4_t src2_0 = vreinterpretq_f32_u16(bf16_vec1_2);
                const float32x4_t src2_1 = vreinterpretq_f32_u16(bf16_vec2_2);
                const float32x4_t src3_0 = vreinterpretq_f32_u16(bf16_vec1_3);
                const float32x4_t src3_1 = vreinterpretq_f32_u16(bf16_vec2_3);

                // Calculate the maximum
                vmax0 = vmaxq_f32(src0_0, vmax0);
                vmax0 = vmaxq_f32(vmax0, src0_1);
                vmax1 = vmaxq_f32(src1_0, vmax1);
                vmax1 = vmaxq_f32(vmax1, src1_1);
                vmax2 = vmaxq_f32(src2_0, vmax2);
                vmax2 = vmaxq_f32(vmax2, src2_1);
                vmax3 = vmaxq_f32(src3_0, vmax3);
                vmax3 = vmaxq_f32(vmax3, src3_1);

                // Calculate the minimum
                vmin0 = vminq_f32(src0_0, vmin0);
                vmin0 = vminq_f32(vmin0, src0_1);
                vmin1 = vminq_f32(src1_0, vmin1);
                vmin1 = vminq_f32(vmin1, src1_1);
                vmin2 = vminq_f32(src2_0, vmin2);
                vmin2 = vminq_f32(vmin2, src2_1);
                vmin3 = vminq_f32(src3_0, vmin3);
                vmin3 = vminq_f32(vmin3, src3_1);
            }
            // Get the max/min scalar values.
            max0 = vmaxvq_f32(vmax0);
            min0 = vminvq_f32(vmin0);
            max1 = vmaxvq_f32(vmax1);
            min1 = vminvq_f32(vmin1);
            max2 = vmaxvq_f32(vmax2);
            min2 = vminvq_f32(vmin2);
            max3 = vmaxvq_f32(vmax3);
            min3 = vminvq_f32(vmin3);
            // Process leftover elements with a scalar loop.
            for (; k_idx < (int32_t)k; ++k_idx) {
                const float src0 = kai_cast_f32_bf16(*(src_ptr + k_idx));
                max0 = fmaxf(src0, max0);
                min0 = fminf(src0, min0);
                const float src1 = kai_cast_f32_bf16(*(src_ptr + k_idx + (lhs_stride / sizeof(uint16_t))));
                max1 = fmaxf(src1, max1);
                min1 = fminf(src1, min1);
                const float src2 = kai_cast_f32_bf16(*(src_ptr + k_idx + (2 * (lhs_stride / sizeof(uint16_t)))));
                max2 = fmaxf(src2, max2);
                min2 = fminf(src2, min2);
                const float src3 = kai_cast_f32_bf16(*(src_ptr + k_idx + (3 * (lhs_stride / sizeof(uint16_t)))));
                max3 = fmaxf(src3, max3);
                min3 = fminf(src3, min3);
            }

            // Maximum/minimum int8 values
            const float qmin = (float)INT8_MIN;
            const float qmax = (float)INT8_MAX;

            const float rmin0 = fminf(0.0F, min0);
            const float rmax0 = fmaxf(0.0F, max0);
            const float scale0 = rmin0 == rmax0 ? 1.F : (qmax - qmin) / (rmax0 - rmin0);
            const float rmin1 = fminf(0.0F, min1);
            const float rmax1 = fmaxf(0.0F, max1);
            const float scale1 = rmin1 == rmax1 ? 1.F : (qmax - qmin) / (rmax1 - rmin1);
            const float rmin2 = fminf(0.0F, min2);
            const float rmax2 = fmaxf(0.0F, max2);
            const float scale2 = rmin2 == rmax2 ? 1.F : (qmax - qmin) / (rmax2 - rmin2);
            const float rmin3 = fminf(0.0F, min3);
            const float rmax3 = fmaxf(0.0F, max3);
            const float scale3 = rmin3 == rmax3 ? 1.F : (qmax - qmin) / (rmax3 - rmin3);

            // Reciprocal to quantize
            const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;
            const float recip_scale1 = scale1 ? 1.0F / scale1 : 0.0F;
            const float recip_scale2 = scale2 ? 1.0F / scale2 : 0.0F;
            const float recip_scale3 = scale3 ? 1.0F / scale3 : 0.0F;

            const float descaled_min0 = rmin0 * scale0;
            const float descaled_max0 = rmax0 * scale0;
            const float descaled_min1 = rmin1 * scale1;
            const float descaled_max1 = rmax1 * scale1;
            const float descaled_min2 = rmin2 * scale2;
            const float descaled_max2 = rmax2 * scale2;
            const float descaled_min3 = rmin3 * scale3;
            const float descaled_max3 = rmax3 * scale3;

            const float zero_point_from_min_error0 = qmin + descaled_min0;
            const float zero_point_from_max_error0 = qmax + descaled_max0;
            const float zero_point_from_min_error1 = qmin + descaled_min1;
            const float zero_point_from_max_error1 = qmax + descaled_max1;
            const float zero_point_from_min_error2 = qmin + descaled_min2;
            const float zero_point_from_max_error2 = qmax + descaled_max2;
            const float zero_point_from_min_error3 = qmin + descaled_min3;
            const float zero_point_from_max_error3 = qmax + descaled_max3;

            float zero_point0 = (zero_point_from_min_error0 + zero_point_from_max_error0 > 0) ? qmin - descaled_min0
                                                                                              : qmax - descaled_max0;
            float zero_point1 = (zero_point_from_min_error1 + zero_point_from_max_error1 > 0) ? qmin - descaled_min1
                                                                                              : qmax - descaled_max1;
            float zero_point2 = (zero_point_from_min_error2 + zero_point_from_max_error2 > 0) ? qmin - descaled_min2
                                                                                              : qmax - descaled_max2;
            float zero_point3 = (zero_point_from_min_error3 + zero_point_from_max_error3 > 0) ? qmin - descaled_min3
                                                                                              : qmax - descaled_max3;

            zero_point0 = fmaxf(zero_point0, qmin);
            zero_point0 = fminf(zero_point0, qmax);
            zero_point1 = fmaxf(zero_point1, qmin);
            zero_point1 = fminf(zero_point1, qmax);
            zero_point2 = fmaxf(zero_point2, qmin);
            zero_point2 = fminf(zero_point2, qmax);
            zero_point3 = fmaxf(zero_point3, qmin);
            zero_point3 = fminf(zero_point3, qmax);

            // Round to nearest integer
            const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);
            const int32_t nudged_zero_point1 = (int32_t)rintf(zero_point1);
            const int32_t nudged_zero_point2 = (int32_t)rintf(zero_point2);
            const int32_t nudged_zero_point3 = (int32_t)rintf(zero_point3);

            const size_t dst_x = ((row_idx + m_idx_start) % mr);

            uint8_t* dst_ptr = (uint8_t*)lhs_packed + (dst_x * k_block_len);

            // Quantize the channels
            int32_t block_idx = 0;

            for (; block_idx < num_blocks_k; ++block_idx) {
                // Clamp at the last valid k-index
                const int32_t k_idx_start = block_idx * k_block_len;

                // Load eight bfloat16 values and convert them to float32.
                const uint16x8_t bf16_vec_0 = vld1q_u16(src_ptr + k_idx_start);
                const uint16x8_t bf16_vec_1 = vld1q_u16(src_ptr + k_idx_start + (lhs_stride / sizeof(uint16_t)));
                const uint16x8_t bf16_vec_2 = vld1q_u16(src_ptr + k_idx_start + (2 * (lhs_stride / sizeof(uint16_t))));
                const uint16x8_t bf16_vec_3 = vld1q_u16(src_ptr + k_idx_start + (3 * (lhs_stride / sizeof(uint16_t))));
                const uint16x8_t bf16_vec1_0 = vzip1q_u16(zero, bf16_vec_0);
                const uint16x8_t bf16_vec2_0 = vzip2q_u16(zero, bf16_vec_0);
                const uint16x8_t bf16_vec1_1 = vzip1q_u16(zero, bf16_vec_1);
                const uint16x8_t bf16_vec2_1 = vzip2q_u16(zero, bf16_vec_1);
                const uint16x8_t bf16_vec1_2 = vzip1q_u16(zero, bf16_vec_2);
                const uint16x8_t bf16_vec2_2 = vzip2q_u16(zero, bf16_vec_2);
                const uint16x8_t bf16_vec1_3 = vzip1q_u16(zero, bf16_vec_3);
                const uint16x8_t bf16_vec2_3 = vzip2q_u16(zero, bf16_vec_3);
                const float32x4_t src0_0 = vreinterpretq_f32_u16(bf16_vec1_0);
                const float32x4_t src0_1 = vreinterpretq_f32_u16(bf16_vec2_0);
                const float32x4_t src1_0 = vreinterpretq_f32_u16(bf16_vec1_1);
                const float32x4_t src1_1 = vreinterpretq_f32_u16(bf16_vec2_1);
                const float32x4_t src2_0 = vreinterpretq_f32_u16(bf16_vec1_2);
                const float32x4_t src2_1 = vreinterpretq_f32_u16(bf16_vec2_2);
                const float32x4_t src3_0 = vreinterpretq_f32_u16(bf16_vec1_3);
                const float32x4_t src3_1 = vreinterpretq_f32_u16(bf16_vec2_3);

                // Scale the values.
                const int16x4_t v0_0 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src0_0, scale0)));
                const int16x4_t v1_0 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src0_1, scale0)));
                const int16x4_t v0_1 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src1_0, scale1)));
                const int16x4_t v1_1 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src1_1, scale1)));
                const int16x4_t v0_2 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src2_0, scale2)));
                const int16x4_t v1_2 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src2_1, scale2)));
                const int16x4_t v0_3 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src3_0, scale3)));
                const int16x4_t v1_3 = vqmovn_s32(vcvtnq_s32_f32(vmulq_n_f32(src3_1, scale3)));

                int16x8_t v0_s16 = vcombine_s16(v0_0, v1_0);
                int16x8_t v1_s16 = vcombine_s16(v0_1, v1_1);
                int16x8_t v2_s16 = vcombine_s16(v0_2, v1_2);
                int16x8_t v3_s16 = vcombine_s16(v0_3, v1_3);

                // Add zero points.
                const int16x8_t vnzp0 = vdupq_n_s16((int16_t)nudged_zero_point0);
                const int16x8_t vnzp1 = vdupq_n_s16((int16_t)nudged_zero_point1);
                const int16x8_t vnzp2 = vdupq_n_s16((int16_t)nudged_zero_point2);
                const int16x8_t vnzp3 = vdupq_n_s16((int16_t)nudged_zero_point3);

                v0_s16 = vaddq_s16(v0_s16, vnzp0);
                v0_s16 = vmaxq_s16(v0_s16, vdupq_n_s16(INT8_MIN));
                v0_s16 = vminq_s16(v0_s16, vdupq_n_s16(INT8_MAX));
                v1_s16 = vaddq_s16(v1_s16, vnzp1);
                v1_s16 = vmaxq_s16(v1_s16, vdupq_n_s16(INT8_MIN));
                v1_s16 = vminq_s16(v1_s16, vdupq_n_s16(INT8_MAX));
                v2_s16 = vaddq_s16(v2_s16, vnzp2);
                v2_s16 = vmaxq_s16(v2_s16, vdupq_n_s16(INT8_MIN));
                v2_s16 = vminq_s16(v2_s16, vdupq_n_s16(INT8_MAX));
                v3_s16 = vaddq_s16(v3_s16, vnzp3);
                v3_s16 = vmaxq_s16(v3_s16, vdupq_n_s16(INT8_MIN));
                v3_s16 = vminq_s16(v3_s16, vdupq_n_s16(INT8_MAX));

                const int8x8_t v0_s8 = vqmovn_s16(v0_s16);
                const int8x8_t v1_s8 = vqmovn_s16(v1_s16);
                const int8x8_t v2_s8 = vqmovn_s16(v2_s16);
                const int8x8_t v3_s8 = vqmovn_s16(v3_s16);

                vst1_s8((int8_t*)(dst_ptr), v0_s8);
                vst1_s8((int8_t*)(dst_ptr + sizeof(int8x8_t)), v1_s8);
                vst1_s8((int8_t*)(dst_ptr + 2 * sizeof(int8x8_t)), v2_s8);
                vst1_s8((int8_t*)(dst_ptr + 3 * sizeof(int8x8_t)), v3_s8);
                dst_ptr += 4 * sizeof(int8x8_t);
            }

            for (; block_idx < num_blocks_k_internal; ++block_idx) {
                // Left over k
                for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    // Clamp at the last valid k-index.
                    const size_t k_idx_start = KAI_MIN((size_t)((block_idx * k_block_len) + k_block_idx), k - 1);

                    const float src0 = kai_cast_f32_bf16(*(src_ptr + k_idx_start));
                    const float src1 = kai_cast_f32_bf16(*(src_ptr + k_idx_start + (lhs_stride / sizeof(uint16_t))));
                    const float src2 =
                        kai_cast_f32_bf16(*(src_ptr + k_idx_start + (2 * (lhs_stride / sizeof(uint16_t)))));
                    const float src3 =
                        kai_cast_f32_bf16(*(src_ptr + k_idx_start + (3 * (lhs_stride / sizeof(uint16_t)))));

                    // Scale the value.
                    int32_t v0_s32 = (int32_t)(roundf(src0 * scale0));
                    int32_t v1_s32 = (int32_t)(roundf(src1 * scale1));
                    int32_t v2_s32 = (int32_t)(roundf(src2 * scale2));
                    int32_t v3_s32 = (int32_t)(roundf(src3 * scale3));

                    v0_s32 = v0_s32 + nudged_zero_point0;
                    v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                    v0_s32 = KAI_MIN(v0_s32, INT8_MAX);

                    v1_s32 = v1_s32 + nudged_zero_point1;
                    v1_s32 = KAI_MAX(v1_s32, INT8_MIN);
                    v1_s32 = KAI_MIN(v1_s32, INT8_MAX);

                    v2_s32 = v2_s32 + nudged_zero_point2;
                    v2_s32 = KAI_MAX(v2_s32, INT8_MIN);
                    v2_s32 = KAI_MIN(v2_s32, INT8_MAX);

                    v3_s32 = v3_s32 + nudged_zero_point3;
                    v3_s32 = KAI_MAX(v3_s32, INT8_MIN);
                    v3_s32 = KAI_MIN(v3_s32, INT8_MAX);

                    *(int8_t*)dst_ptr = (int8_t)v0_s32;
                    *(int8_t*)(dst_ptr + sizeof(int8x8_t)) = (int8_t)v1_s32;
                    *(int8_t*)(dst_ptr + 2 * sizeof(int8x8_t)) = (int8_t)v2_s32;
                    *(int8_t*)(dst_ptr + 3 * sizeof(int8x8_t)) = (int8_t)v3_s32;

                    dst_ptr += sizeof(int8_t);
                }
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }

            uint8_t* dst_base = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

            dst_ptr = dst_base + dst_x * kai_num_bytes_per_offset;

            // LHS offset at the beginning of the row.
            *((int32_t*)(dst_ptr)) = -nudged_zero_point0;
            *((int32_t*)(dst_ptr + kai_num_bytes_per_offset)) = -nudged_zero_point1;
            *((int32_t*)(dst_ptr + 2 * kai_num_bytes_per_offset)) = -nudged_zero_point2;
            *((int32_t*)(dst_ptr + 3 * kai_num_bytes_per_offset)) = -nudged_zero_point3;

            // Assuming the same sizeof() for kai_num_bytes_per_offset and kai_num_bytes_per_multiplier.
            KAI_ASSERT(kai_num_bytes_per_offset == kai_num_bytes_per_multiplier);

            dst_ptr += mr * kai_num_bytes_per_offset;

            // Store the scale quantization params.
            *((float*)(dst_ptr)) = recip_scale0;
            *((float*)(dst_ptr + kai_num_bytes_per_multiplier)) = recip_scale1;
            *((float*)(dst_ptr + 2 * kai_num_bytes_per_multiplier)) = recip_scale2;
            *((float*)(dst_ptr + 3 * kai_num_bytes_per_multiplier)) = recip_scale3;

            // Update src_ptr. Note: now lhs contains bfloat16 values (2 bytes each).
            src_ptr += (4 * lhs_stride / sizeof(uint16_t));

            // Move to the next row as we have interleaved all Mr rows.
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }

    for (; row_idx < m; ++row_idx) {
        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        int32_t k_idx = 0;
        float32x4_t vmax0 = vdupq_n_f32(-FLT_MAX);
        float32x4_t vmin0 = vdupq_n_f32(FLT_MAX);
        const uint16x8_t zero = vdupq_n_u16(0);
        // Process 8 bfloat16 values per iteration.
        for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
            // Load eight bfloat16 values.
            const uint16x8_t bf16_vec = vld1q_u16(src_ptr + k_idx);
            const uint16x8_t bf16_vec1 = vzip1q_u16(zero, bf16_vec);
            const uint16x8_t bf16_vec2 = vzip2q_u16(zero, bf16_vec);
            const float32x4_t src0_0 = vreinterpretq_f32_u16(bf16_vec1);
            const float32x4_t src0_1 = vreinterpretq_f32_u16(bf16_vec2);

            // Calculate the maximum
            vmax0 = vmaxq_f32(src0_0, vmax0);
            vmax0 = vmaxq_f32(vmax0, src0_1);

            // Calculate the minimum
            vmin0 = vminq_f32(src0_0, vmin0);
            vmin0 = vminq_f32(vmin0, src0_1);
        }
        // Get the max/min scalar values.
        max0 = vmaxvq_f32(vmax0);
        min0 = vminvq_f32(vmin0);
        // Process leftover elements with a scalar loop.
        for (; k_idx < (int32_t)k; ++k_idx) {
            const float src0_0 = kai_cast_f32_bf16(*(src_ptr + k_idx));
            max0 = fmaxf(src0_0, max0);
            min0 = fminf(src0_0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT8_MIN;
        const float qmax = (float)INT8_MAX;

        const float rmin0 = fminf(0.0F, min0);
        const float rmax0 = fmaxf(0.0F, max0);
        const float scale0 = rmin0 == rmax0 ? 1.F : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;

        const float descaled_min0 = rmin0 * scale0;
        const float descaled_max0 = rmax0 * scale0;

        const float zero_point_from_min_error0 = qmin + descaled_min0;
        const float zero_point_from_max_error0 = qmax + descaled_max0;

        float zero_point0 =
            (zero_point_from_min_error0 + zero_point_from_max_error0 > 0) ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = fmaxf(zero_point0, qmin);
        zero_point0 = fminf(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);

        const size_t dst_x = ((row_idx + m_idx_start) % mr);

        uint8_t* dst_ptr = (uint8_t*)lhs_packed + (dst_x * k_block_len * sizeof(int8_t));

        // Quantize the channels
        int32_t block_idx = 0;

        for (; block_idx < num_blocks_k; ++block_idx) {
            // Clamp at the last valid k-index
            const int32_t k_idx_start = block_idx * k_block_len;

            // Load eight bfloat16 values and convert them to float32.
            const uint16x8_t bf16_vec = vld1q_u16(src_ptr + k_idx_start);
            const uint16x8_t bf16_vec1 = vzip1q_u16(zero, bf16_vec);
            const uint16x8_t bf16_vec2 = vzip2q_u16(zero, bf16_vec);
            const float32x4_t src0_0 = vreinterpretq_f32_u16(bf16_vec1);
            const float32x4_t src0_1 = vreinterpretq_f32_u16(bf16_vec2);

            // Scale the values.
            const float32x4_t v0_f32 = vmulq_n_f32(src0_0, scale0);
            const float32x4_t v1_f32 = vmulq_n_f32(src0_1, scale0);
            const int32x4_t v0_s32 = vcvtnq_s32_f32(v0_f32);
            const int32x4_t v1_s32 = vcvtnq_s32_f32(v1_f32);

            const int16x4_t v0_s16 = vqmovn_s32(v0_s32);
            const int16x4_t v1_s16 = vqmovn_s32(v1_s32);
            int16x8_t v_s16 = vcombine_s16(v0_s16, v1_s16);

            // Add zero points.
            int16_t nzp_s16 = (int16_t)nudged_zero_point0;
            int16x8_t vnzp_s16 = vdupq_n_s16(nzp_s16);
            v_s16 = vaddq_s16(v_s16, vnzp_s16);
            v_s16 = vmaxq_s16(v_s16, vdupq_n_s16(INT8_MIN));
            v_s16 = vminq_s16(v_s16, vdupq_n_s16(INT8_MAX));

            const int8x8_t v0_s8 = vqmovn_s16(v_s16);
            vst1_s8((int8_t*)(dst_ptr), v0_s8);
            dst_ptr += 8 * sizeof(int8_t);
            dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
        }

        for (; block_idx < num_blocks_k_internal; ++block_idx) {
            // Left over k
            for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                // Clamp at the last valid k-index.
                const size_t k_idx_start = KAI_MIN((size_t)((block_idx * k_block_len) + k_block_idx), k - 1);

                const float src0_0 = kai_cast_f32_bf16(*(src_ptr + k_idx_start));

                // Scale the value.
                int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

                v0_s32 = v0_s32 + nudged_zero_point0;
                v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                v0_s32 = KAI_MIN(v0_s32, INT8_MAX);

                *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
                dst_ptr += sizeof(int8_t);
            }
            dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
        }

        dst_ptr = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

        dst_ptr += dst_x * kai_num_bytes_per_offset;

        // LHS offset at the beginning of the row.
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;

        // Assuming the same sizeof() for kai_num_bytes_per_offset and kai_num_bytes_per_multiplier.
        KAI_ASSERT(kai_num_bytes_per_offset == kai_num_bytes_per_multiplier);

        dst_ptr += mr * kai_num_bytes_per_offset;

        // Store the scale quantization params.
        *((float*)(dst_ptr)) = recip_scale0;

        // Update src_ptr. Note: now lhs contains bfloat16 values (2 bytes each).
        src_ptr += (lhs_stride / sizeof(uint16_t));

        // Move to the next row if we have interleaved all Mr rows.
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }
}
