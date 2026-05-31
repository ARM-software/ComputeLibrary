//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qai8dxp_f16_neon.h"

#include <arm_fp16.h>
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#define FLT16_MAX 65504.0
#define FLT16_MIN (-65504.0F)

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return mr * (k_internal * sizeof(int8_t) + kai_num_bytes_per_multiplier + kai_num_bytes_per_offset);
}

size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f16_neon(size_t mr) {
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    // It always points to the beginning of the row
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_quant_pack_qai8dxp_f16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSUME((kr / sr == 8) || (kr / sr == 4));

    if (m == 0) {
        return;
    }

    const size_t num_rows = m;

    float16_t const* src_ptr = (float16_t const*)lhs;

    const size_t dst_stride = kai_lhs_packed_stride(k, mr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k);
    const int32_t k_block_len = (int32_t)(kr / sr);

    const int32_t num_blocks_k = (int32_t)(k / k_block_len);
    const int32_t num_blocks_k_internal = (int32_t)(k_internal / k_block_len);
    const size_t lhs_row_length = lhs_stride / sizeof(float16_t);

    const float16x8_t vmax = vdupq_n_f16((float16_t)FLT16_MIN);
    const float16x8_t vmin = vdupq_n_f16((float16_t)FLT16_MAX);

    // As we load 8-element vectors, limit vectorized loop to avoid reading out-of-bounds
    const int32_t blocks_lim_k = num_blocks_k - (8 / k_block_len);

    size_t row_idx = 0;

    // Improved performance with 4x loop unrolling where packing parameters allow
    if (mr == 4) {
        for (; row_idx + 3 < m; row_idx += 4) {
            // Find min/max for each channel
            int32_t k_idx = 0;
            float16x8_t vmax0 = vmax;
            float16x8_t vmin0 = vmin;
            float16x8_t vmax1 = vmax;
            float16x8_t vmin1 = vmin;
            float16x8_t vmax2 = vmax;
            float16x8_t vmin2 = vmin;
            float16x8_t vmax3 = vmax;
            float16x8_t vmin3 = vmin;

            for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
                const float16x8_t src0 = vld1q_f16(src_ptr + k_idx);
                const float16x8_t src1 = vld1q_f16(src_ptr + k_idx + lhs_row_length);
                const float16x8_t src2 = vld1q_f16(src_ptr + k_idx + (2 * lhs_row_length));
                const float16x8_t src3 = vld1q_f16(src_ptr + k_idx + (3 * lhs_row_length));

                vmax0 = vmaxq_f16(src0, vmax0);
                vmax1 = vmaxq_f16(src1, vmax1);
                vmax2 = vmaxq_f16(src2, vmax2);
                vmax3 = vmaxq_f16(src3, vmax3);
                vmin0 = vminq_f16(src0, vmin0);
                vmin1 = vminq_f16(src1, vmin1);
                vmin2 = vminq_f16(src2, vmin2);
                vmin3 = vminq_f16(src3, vmin3);
            }

            float16_t max0 = vmaxvq_f16(vmax0);
            float16_t min0 = vminvq_f16(vmin0);
            float16_t max1 = vmaxvq_f16(vmax1);
            float16_t min1 = vminvq_f16(vmin1);
            float16_t max2 = vmaxvq_f16(vmax2);
            float16_t min2 = vminvq_f16(vmin2);
            float16_t max3 = vmaxvq_f16(vmax3);
            float16_t min3 = vminvq_f16(vmin3);
            // Process leftover elements with a scalar loop.
            for (; k_idx < (int32_t)k; ++k_idx) {
                const float16_t src0 = *(src_ptr + (size_t)k_idx);
                max0 = vmaxh_f16(src0, max0);
                min0 = vminh_f16(src0, min0);
                const float16_t src1 = *(src_ptr + (size_t)k_idx + lhs_row_length);
                max1 = vmaxh_f16(src1, max1);
                min1 = vminh_f16(src1, min1);
                const float16_t src2 = *(src_ptr + (size_t)k_idx + (2 * lhs_row_length));
                max2 = vmaxh_f16(src2, max2);
                min2 = vminh_f16(src2, min2);
                const float16_t src3 = *(src_ptr + (size_t)k_idx + (3 * lhs_row_length));
                max3 = vmaxh_f16(src3, max3);
                min3 = vminh_f16(src3, min3);
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
            const int32_t block_incr = 8 / k_block_len;

            for (; block_idx <= blocks_lim_k; block_idx += block_incr) {
                // Clamp at the last valid k-index
                const int32_t k_idx_start = block_idx * k_block_len;

                const float16x8_t src0 = vld1q_f16(src_ptr + k_idx_start);
                const float16x8_t src1 = vld1q_f16(src_ptr + k_idx_start + lhs_row_length);
                const float16x8_t src2 = vld1q_f16(src_ptr + k_idx_start + (2 * lhs_row_length));
                const float16x8_t src3 = vld1q_f16(src_ptr + k_idx_start + (3 * lhs_row_length));

                // Scale the values.
                const int32x4_t v0_0_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src0)), scale0));
                const int32x4_t v0_1_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_high_f32_f16(src0), scale0));
                const int32x4_t v1_0_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src1)), scale1));
                const int32x4_t v1_1_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_high_f32_f16(src1), scale1));
                const int32x4_t v2_0_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src2)), scale2));
                const int32x4_t v2_1_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_high_f32_f16(src2), scale2));
                const int32x4_t v3_0_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src3)), scale3));
                const int32x4_t v3_1_s32 = vcvtq_s32_f32(vmulq_n_f32(vcvt_high_f32_f16(src3), scale3));

                const int16x4_t v0_0_s16 = vqmovn_s32(v0_0_s32);
                const int16x4_t v0_1_s16 = vqmovn_s32(v0_1_s32);
                const int16x4_t v1_0_s16 = vqmovn_s32(v1_0_s32);
                const int16x4_t v1_1_s16 = vqmovn_s32(v1_1_s32);
                const int16x4_t v2_0_s16 = vqmovn_s32(v2_0_s32);
                const int16x4_t v2_1_s16 = vqmovn_s32(v2_1_s32);
                const int16x4_t v3_0_s16 = vqmovn_s32(v3_0_s32);
                const int16x4_t v3_1_s16 = vqmovn_s32(v3_1_s32);

                int16x8_t v0_s16;
                int16x8_t v1_s16;
                int16x8_t v2_s16;
                int16x8_t v3_s16;
                if (k_block_len == 8) {
                    v0_s16 = vcombine_s16(v0_0_s16, v0_1_s16);
                    v1_s16 = vcombine_s16(v1_0_s16, v1_1_s16);
                    v2_s16 = vcombine_s16(v2_0_s16, v2_1_s16);
                    v3_s16 = vcombine_s16(v3_0_s16, v3_1_s16);
                } else {  // k_block_len == 4
                    v0_s16 = vcombine_s16(v0_0_s16, v1_0_s16);
                    v1_s16 = vcombine_s16(v2_0_s16, v3_0_s16);
                    v2_s16 = vcombine_s16(v0_1_s16, v1_1_s16);
                    v3_s16 = vcombine_s16(v2_1_s16, v3_1_s16);
                }

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

                int8x8_t v0_s8 = vqmovn_s16(v0_s16);
                int8x8_t v1_s8 = vqmovn_s16(v1_s16);
                int8x8_t v2_s8 = vqmovn_s16(v2_s16);
                int8x8_t v3_s8 = vqmovn_s16(v3_s16);

                vst1_s8((int8_t*)(dst_ptr), v0_s8);
                vst1_s8((int8_t*)(dst_ptr + sizeof(int8x8_t)), v1_s8);
                vst1_s8((int8_t*)(dst_ptr + 2 * sizeof(int8x8_t)), v2_s8);
                vst1_s8((int8_t*)(dst_ptr + 3 * sizeof(int8x8_t)), v3_s8);
                dst_ptr += block_incr * mr * k_block_len * sizeof(int8_t);
            }

            for (; block_idx < num_blocks_k_internal; ++block_idx) {
                // left over k
                for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    // Clamp at the last valid k-index.
                    const size_t k_idx_start = KAI_MIN((size_t)((block_idx * k_block_len) + k_block_idx), k - 1);

                    const float src0 = (float)(*(src_ptr + k_idx_start));
                    const float src1 = (float)(*(src_ptr + k_idx_start + lhs_row_length));
                    const float src2 = (float)(*(src_ptr + k_idx_start + (2 * lhs_row_length)));
                    const float src3 = (float)(*(src_ptr + k_idx_start + (3 * lhs_row_length)));

                    // Scale the value.
                    int32_t d0_s32 = (int32_t)(roundf(src0 * scale0));
                    int32_t d1_s32 = (int32_t)(roundf(src1 * scale1));
                    int32_t d2_s32 = (int32_t)(roundf(src2 * scale2));
                    int32_t d3_s32 = (int32_t)(roundf(src3 * scale3));

                    d0_s32 = d0_s32 + nudged_zero_point0;
                    d0_s32 = KAI_MAX(d0_s32, INT8_MIN);
                    d0_s32 = KAI_MIN(d0_s32, INT8_MAX);

                    d1_s32 = d1_s32 + nudged_zero_point1;
                    d1_s32 = KAI_MAX(d1_s32, INT8_MIN);
                    d1_s32 = KAI_MIN(d1_s32, INT8_MAX);

                    d2_s32 = d2_s32 + nudged_zero_point2;
                    d2_s32 = KAI_MAX(d2_s32, INT8_MIN);
                    d2_s32 = KAI_MIN(d2_s32, INT8_MAX);

                    d3_s32 = d3_s32 + nudged_zero_point3;
                    d3_s32 = KAI_MAX(d3_s32, INT8_MIN);
                    d3_s32 = KAI_MIN(d3_s32, INT8_MAX);

                    *(int8_t*)dst_ptr = (int8_t)d0_s32;
                    *(int8_t*)(dst_ptr + k_block_len * sizeof(int8_t)) = (int8_t)d1_s32;
                    *(int8_t*)(dst_ptr + 2 * (k_block_len * sizeof(int8_t))) = (int8_t)d2_s32;
                    *(int8_t*)(dst_ptr + 3 * (k_block_len * sizeof(int8_t))) = (int8_t)d3_s32;
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

            // Update src_ptr. Note: now lhs contains fp16 values (2 bytes each).
            src_ptr += (4 * lhs_row_length);

            // Move to the next row as we have interleaved all Mr rows.
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }

    for (; row_idx < num_rows; ++row_idx) {
        // Find min/max for each channel
        int32_t k_idx = 0;
        float16x8_t vmax0 = vmax;
        float16x8_t vmin0 = vmin;

        for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
            const float16x8_t src0_0 = vld1q_f16(src_ptr + (size_t)k_idx);
            vmax0 = vmaxq_f16(vmax0, src0_0);
            vmin0 = vminq_f16(vmin0, src0_0);
        }
        // Get the max/min
        float16_t max0 = vmaxvq_f16(vmax0);
        float16_t min0 = vminvq_f16(vmin0);

        for (; k_idx < (int32_t)k; ++k_idx) {
            const float16_t src0 = *(src_ptr + (size_t)k_idx);
            max0 = vmaxh_f16(src0, max0);
            min0 = vminh_f16(src0, min0);
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
            zero_point_from_min_error0 + zero_point_from_max_error0 > 0 ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = fmaxf(zero_point0, qmin);
        zero_point0 = fminf(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);

        const size_t dst_x = ((row_idx + m_idx_start) % mr);

        uint8_t* dst_ptr = (uint8_t*)lhs_packed + (dst_x * k_block_len * sizeof(int8_t));

        // Quantize the channels
        int32_t block_idx = 0;

        for (; block_idx <= blocks_lim_k; ++block_idx) {
            const int32_t k_idx_start = block_idx * k_block_len;

            const float16x8_t src_0 = vld1q_f16(src_ptr + k_idx_start);

            // Scale the values
            const float32x4_t v0_f32 = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src_0)), scale0);
            const float32x4_t v1_f32 = vmulq_n_f32(vcvt_high_f32_f16(src_0), scale0);
            const int32x4_t v0_s32 = vcvtnq_s32_f32(v0_f32);
            const int32x4_t v1_s32 = vcvtnq_s32_f32(v1_f32);

            const int16x4_t v0_s16 = vqmovn_s32(v0_s32);
            const int16x4_t v1_s16 = vqmovn_s32(v1_s32);
            int16x8_t v_s16 = vcombine_s16(v0_s16, v1_s16);

            // Add zero points
            int16_t nzp_s16 = (int16_t)nudged_zero_point0;
            int16x8_t vnzp_s16 = vdupq_n_s16(nzp_s16);
            v_s16 = vaddq_s16(v_s16, vnzp_s16);
            v_s16 = vmaxq_s16(v_s16, vdupq_n_s16(INT8_MIN));
            v_s16 = vminq_s16(v_s16, vdupq_n_s16(INT8_MAX));

            int8x8_t v_s8 = vqmovn_s16(v_s16);
            vst1_s8((int8_t*)(dst_ptr), v_s8);
            dst_ptr += mr * k_block_len * sizeof(int8_t);
        }

        for (; block_idx < num_blocks_k_internal; ++block_idx) {
            // left over k
            for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = KAI_MIN((size_t)((block_idx * k_block_len) + k_block_idx), k - 1);

                const float src0 = (float)(*(src_ptr + k_idx_start));

                // Scale the values
                int32_t d0_s32 = (int32_t)(roundf(src0 * scale0));

                d0_s32 = d0_s32 + nudged_zero_point0;
                d0_s32 = KAI_MAX(d0_s32, INT8_MIN);
                d0_s32 = KAI_MIN(d0_s32, INT8_MAX);

                *((int8_t*)(dst_ptr)) = (int8_t)d0_s32;
                dst_ptr += sizeof(int8_t);
            }
            dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
        }

        dst_ptr = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

        dst_ptr += dst_x * kai_num_bytes_per_offset;

        // LHS offset at the beginning of the row
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;

        // Assuming the same sizeof() for kai_num_bytes_per_offset and kai_num_bytes_per_multiplier
        KAI_ASSERT(kai_num_bytes_per_offset == kai_num_bytes_per_multiplier);

        dst_ptr += mr * kai_num_bytes_per_offset;

        // Store the scale quantization params
        *((float*)(dst_ptr)) = recip_scale0;

        src_ptr += lhs_row_length;

        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }
}

#endif  // Architectural features check.
