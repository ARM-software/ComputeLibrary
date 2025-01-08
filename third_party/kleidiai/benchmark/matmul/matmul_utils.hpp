//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#define INT4_MIN (-8)
#define INT4_MAX (7)

static void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (int i = 0; i < num_rows * num_cols; i++) {
        dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
}

static void quant_qs4cx_f32(size_t n, size_t k, const float* rhs_f32, uint8_t* rhs_qs4cx, float* rhs_scales_f32) {
    const size_t dst_stride = (k / 2) * sizeof(int8_t);

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            max0 = std::max(src0_0, max0);
            min0 = std::min(src0_0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT4_MIN;
        const float qmax = (float)INT4_MAX;

        const float rmin0 = std::min(0.0f, min0);
        const float rmax0 = std::max(0.0f, max0);

        const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

        uint8_t* dst_ptr = rhs_qs4cx + row_idx * dst_stride;

        // Quantize the channels
        for (size_t k_idx = 0; k_idx < k; k_idx += 2) {
            const float src0_0 = src_ptr[k_idx + 0];
            const float src0_1 = src_ptr[k_idx + 1];

            // Scale the values
            int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));
            int32_t v1_s32 = (int32_t)(round(src0_1 * scale0));

            // Maximum/minimum int4 values
            v0_s32 = std::clamp(v0_s32, INT4_MIN, INT4_MAX);
            v1_s32 = std::clamp(v1_s32, INT4_MIN, INT4_MAX);

            int32_t v0_u8 = (uint8_t)(v0_s32 + 8);
            int32_t v1_u8 = (uint8_t)(v1_s32 + 8);

            const uint8_t rhs_v0 = (v1_u8 << 4) | v0_u8;

            dst_ptr[0] = rhs_v0;
            dst_ptr += sizeof(uint8_t);
        }

        rhs_scales_f32[row_idx] = recip_scale0;
    }
};
