//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
#error "Dotprod and I8mm extensions required to compile this example"
#else
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

// Include micro-kernel variants
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4cxp_interface.h"
#include "kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)

enum class rhs_format {
    nxk,
    kxn,
};

struct mnk {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
};

mnk matmul_shapes[] = {{1, 33, 32}, {13, 33, 32}, {37, 75, 17}, {16, 32, 64}, {7, 17, 33}, {15, 31, 45}};

// Micro-kernel interface
struct kai_matmul_ukernel_f32_qa8dxp_qs4cxp {
    kai_matmul_clamp_f32_qai8dxp_qsi4cxp_ukernel ukernel;
    std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs4cxp ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod,
     "matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod,
     "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod"}

};

// Number of micro-kernel variants stored in the array
const size_t num_ukernel_variants = sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);

static size_t roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

static void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (int i = 0; i < num_rows * num_cols; i++) {
        dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
}

static void ref_quant_qa8dx_f32(size_t m, size_t k, const float* lhs_f32, int8_t* lhs_qa8dx) {
    const size_t dst_stride = (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t));

    const size_t lhs_qa8dx_stride = k;

    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        const float* src_ptr = lhs_f32 + m_idx * lhs_qa8dx_stride;

        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            max0 = std::max(src0_0, max0);
            min0 = std::min(src0_0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT8_MIN;
        const float qmax = (float)INT8_MAX;

        const float rmin0 = std::min(0.0f, min0);
        const float rmax0 = std::max(0.0f, max0);

        const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

        const float descaled_min0 = rmin0 * scale0;
        const float descaled_max0 = rmax0 * scale0;

        const float zero_point_from_min_error0 = qmin + descaled_min0;
        const float zero_point_from_max_error0 = qmax + descaled_max0;

        float zero_point0 =
            zero_point_from_min_error0 + zero_point_from_max_error0 > 0 ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = std::max(zero_point0, qmin);
        zero_point0 = std::min(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = lrintf(zero_point0);

        int8_t* dst_ptr = (int8_t*)lhs_qa8dx + m_idx * dst_stride;

        // LHS offset at the beginning of the row
        *((float*)(dst_ptr)) = recip_scale0;
        dst_ptr += sizeof(float);
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;
        dst_ptr += sizeof(int32_t);

        // Quantize the channels
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            // Scale the values
            int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));

            v0_s32 = v0_s32 + nudged_zero_point0;
            v0_s32 = std::max(v0_s32, INT8_MIN);
            v0_s32 = std::min(v0_s32, INT8_MAX);
            dst_ptr[0] = (int8_t)v0_s32;
            dst_ptr += sizeof(int8_t);
        }
    }
};

static void quant_nxk_qs4cx_f32(size_t n, size_t k, const float* rhs_f32, uint8_t* rhs_qs4cx, float* rhs_scales_f32) {
    const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

    // Make sure the output is filled with zeros
    std::memset(rhs_qs4cx, 0, n * rhs_qs4cx_stride);

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        const float* src_ptr = rhs_f32 + n_idx * k;

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

        // Quantize the channels
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            // Scale the values
            int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));

            // Maximum/minimum int4 values
            v0_s32 = std::max(v0_s32, INT4_MIN);
            v0_s32 = std::min(v0_s32, INT4_MAX);

            const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

            const size_t dst_addr = (k_idx / 2) + n_idx * rhs_qs4cx_stride;
            uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

            if ((k_idx % 2) == 0) {
                rhs_v0 |= v0_u8;
            } else {
                rhs_v0 |= (v0_u8 << 4);
            }
            rhs_qs4cx[dst_addr] = rhs_v0;
        }

        rhs_scales_f32[n_idx] = recip_scale0;
    }
};

static void quant_kxn_qs4cx_f32(size_t n, size_t k, const float* rhs_f32, uint8_t* rhs_qs4cx, float* rhs_scales_f32) {
    const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

    // Make sure the output is filled with zeros
    std::memset(rhs_qs4cx, 0, k * rhs_qs4cx_stride);

    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        const float* src_ptr = rhs_f32 + n_idx * k;

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

        // Quantize the channels
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            // Scale the values
            int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));

            // Maximum/minimum int4 values
            v0_s32 = std::max(v0_s32, INT4_MIN);
            v0_s32 = std::min(v0_s32, INT4_MAX);

            const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

            const size_t dst_addr = (n_idx / 2) + k_idx * rhs_qs4cx_stride;
            uint8_t rhs_v0 = rhs_qs4cx[dst_addr];

            if ((n_idx % 2) == 0) {
                rhs_v0 |= v0_u8;
            } else {
                rhs_v0 |= (v0_u8 << 4);
            }
            rhs_qs4cx[dst_addr] = rhs_v0;
        }

        rhs_scales_f32[n_idx] = recip_scale0;
    }
};

static void quant_qs4cx_f32(
    size_t n, size_t k, rhs_format format, const float* rhs_f32, uint8_t* rhs_qs4cx, float* rhs_scales_f32) {
    if (rhs_format::nxk == format) {
        quant_nxk_qs4cx_f32(n, k, rhs_f32, rhs_qs4cx, rhs_scales_f32);
    } else {
        quant_kxn_qs4cx_f32(n, k, rhs_f32, rhs_qs4cx, rhs_scales_f32);
    }
};

static void ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
    size_t m, size_t n, size_t k, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4cx, const float* rhs_scales_f32,
    float* dst_f32, float scalar_min, float scalar_max) {
    const size_t lhs_stride = k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

    const size_t rhs_qs4cx_stride = (roundup(k, 2) / 2);

    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            // Main f32 accumulator
            int32_t iacc = 0;

            const int8_t* lhs_ptr = lhs_ptr_start;
            const uint8_t* rhs_ptr = rhs_qs4cx + n_idx * rhs_qs4cx_stride;

            // Get the LHS quantization parameters stored at the
            // beginning of each row
            const float lhs_scale = *(const float*)lhs_ptr;
            lhs_ptr += sizeof(float);

            const int32_t lhs_offset = *(const int32_t*)lhs_ptr;
            lhs_ptr += sizeof(int32_t);

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                // Get the LHS values
                const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

                // Get the RHS values
                const uint8_t rhs_byte = rhs_ptr[0];

                // Unpack the RHS values
                int32_t rhs_v0 = 0;
                if ((k_idx % 2) == 0) {
                    rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                } else {
                    rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
                }

                iacc += lhs_v0 * rhs_v0;
                iacc += lhs_offset * rhs_v0;

                lhs_ptr += 1;

                // Increment only when k_idx is not a multiple of 2
                rhs_ptr += k_idx % 2;
            }

            // Get the RHS scale
            const float rhs_scale = rhs_scales_f32[n_idx];

            float main_acc = iacc * rhs_scale;

            main_acc = main_acc * lhs_scale;

            // Clamp (min-max) operation
            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};

static void ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(
    size_t m, size_t n, size_t k, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4cx, const float* rhs_scales_f32,
    float* dst_f32, float scalar_min, float scalar_max) {
    const size_t lhs_stride = k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

    const size_t rhs_qs4cx_stride = (roundup(n, 2) / 2);

    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8dx + m_idx * lhs_stride;

        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            // Main f32 accumulator
            int32_t iacc = 0;

            const int8_t* lhs_ptr = lhs_ptr_start;
            const uint8_t* rhs_ptr = rhs_qs4cx + (n_idx / 2);

            // Get the LHS quantization parameters stored at the
            // beginning of each row
            const float lhs_scale = *(const float*)lhs_ptr;
            lhs_ptr += sizeof(float);

            const int32_t lhs_offset = *(const int32_t*)lhs_ptr;
            lhs_ptr += sizeof(int32_t);

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                // Get the LHS values
                const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

                // Get the RHS values
                const uint8_t rhs_byte = rhs_ptr[0];

                // Unpack the RHS values
                int32_t rhs_v0 = 0;
                if ((n_idx % 2) == 0) {
                    rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                } else {
                    rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
                }

                iacc += lhs_v0 * rhs_v0;
                iacc += lhs_offset * rhs_v0;

                lhs_ptr += 1;

                // Increment only when k_idx is not a multiple of 2
                rhs_ptr += rhs_qs4cx_stride;
            }

            // Get the RHS scale
            const float rhs_scale = rhs_scales_f32[n_idx];

            float main_acc = iacc * rhs_scale;

            main_acc = main_acc * lhs_scale;

            // Clamp (min-max) operation
            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};

static void ref_matmul_f32_qa8dx_qs4cx(
    size_t m, size_t n, size_t k, rhs_format format, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4cx,
    const float* rhs_scales_f32, float* dst_f32, float scalar_min, float scalar_max) {
    const size_t lhs_stride = k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);

    if (rhs_format::nxk == format) {
        ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4cx(
            m, n, k, lhs_qa8dx, rhs_qs4cx, rhs_scales_f32, dst_f32, scalar_min, scalar_max);
    } else {
        ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4cx(
            m, n, k, lhs_qa8dx, rhs_qs4cx, rhs_scales_f32, dst_f32, scalar_min, scalar_max);
    }
};

static bool is_output_correct(size_t num_rows, size_t num_cols, float tolerance, const float* ref, const float* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;
            printf("ERROR![%ld][%ld]: ref=%.5f vs. act=%.5f\n", y, x, ref[i], act[i]);
            is_valid = false;
            return is_valid;
        }
    }
    return is_valid;
}

int main(int argc, char** argv) {
    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;
    const size_t num_shapes = std::size(matmul_shapes);

    std::cout << "------------" << std::endl;
    for (size_t test_idx = 0; test_idx < num_shapes; ++test_idx) {
        size_t m = matmul_shapes[test_idx].m;
        size_t n = matmul_shapes[test_idx].n;
        size_t k = matmul_shapes[test_idx].k;

        std::cout << "\nTEST[" << m << ", " << n << "," << k << "]"
                  << "\n";
        // Iterate over the RHS format (NxK or KxN)
        for (const rhs_format& format : {rhs_format::nxk, rhs_format::kxn}) {
            std::cout << "Testing RHS format = " << (format == rhs_format::nxk ? "N x K" : "K x N") << std::endl;

            const size_t lhs_native_size_f32 = m * k * sizeof(float);
            const size_t rhs_native_size_f32 = n * k * sizeof(float);
            const size_t rhs_native_size_qs4cx = format == rhs_format::nxk ? n * (roundup(k, 2) / 2) * sizeof(uint8_t)
                                                                           : k * (roundup(n, 2) / 2) * sizeof(uint8_t);
            const size_t rhs_scales_size_f32 = n * sizeof(float);

            // Allocate the memory
            uint8_t* lhs_native_mtx_f32 = new uint8_t[lhs_native_size_f32];
            uint8_t* rhs_native_mtx_f32 = new uint8_t[rhs_native_size_f32];
            uint8_t* rhs_native_mtx_qs4cx = new uint8_t[rhs_native_size_qs4cx];
            uint8_t* rhs_scales_f32 = new uint8_t[rhs_scales_size_f32];

            fill_uniform_random(m, k, (float*)lhs_native_mtx_f32, seed_lhs);
            fill_uniform_random(n, k, (float*)rhs_native_mtx_f32, seed_rhs);

            quant_qs4cx_f32(
                n, k, format, (const float*)rhs_native_mtx_f32, (uint8_t*)rhs_native_mtx_qs4cx, (float*)rhs_scales_f32);

            delete[] rhs_native_mtx_f32;

            //----------- REFERENCE IMPLEMENTATION
            //------------------------------------
            //------------------------------------
            // Memory sizes for the reference implementation
            // After dynamically quantized the LHS matrix, we have the scale and offset for each
            // row. The scale (f32) and offset (int32) are stored at the beginning of each row
            const size_t lhs_ref_size_qa8dx = m * (k + sizeof(int32_t) + sizeof(float));
            const size_t dst_ref_size_f32 = m * n * sizeof(float);

            uint8_t* lhs_ref_mtx_qa8dx = new uint8_t[lhs_ref_size_qa8dx];
            uint8_t* dst_ref_mtx_f32 = new uint8_t[dst_ref_size_f32];

            ref_quant_qa8dx_f32(m, k, (const float*)lhs_native_mtx_f32, (int8_t*)lhs_ref_mtx_qa8dx);

            ref_matmul_f32_qa8dx_qs4cx(
                m, n, k, format, (const int8_t*)lhs_ref_mtx_qa8dx, (const uint8_t*)rhs_native_mtx_qs4cx,
                (const float*)rhs_scales_f32, (float*)dst_ref_mtx_f32, -FLT_MAX, FLT_MAX);

            // Remove the unnecessary buffer
            delete[] lhs_ref_mtx_qa8dx;

            //----------- END REFERENCE IMPLEMENTATION
            //------------------------------------
            //------------------------------------

            //----------- MICRO-KERNELS TESTS
            //------------------------------------
            //------------------------------------
            for (size_t idx_variant = 0; idx_variant < num_ukernel_variants; ++idx_variant) {
                std::cout << "Testing " << ukernel_variants[idx_variant].name << std::endl;

                // Get the packing parameters
                const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
                const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
                const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
                const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

                // Get the size in bytes for the packed matrices
                const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
                size_t rhs_packed_size = 0;

                if (format == rhs_format::nxk) {
                    rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, nr, kr, sr);
                } else {
                    rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(n, k, nr, kr, sr);
                }

                const size_t dst_size = ukernel_variants[idx_variant].ukernel.get_dst_size(m, n);

                // Allocate the matrices
                uint8_t* lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
                uint8_t* rhs_packed_mtx_qs4cx = new uint8_t[rhs_packed_size];
                uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

                // If the RHS matrix contains constant values, the packing can be performed
                // only once
                if (format == rhs_format::nxk) {
                    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params nxk_params;

                    nxk_params.lhs_zero_point = 1;
                    nxk_params.rhs_zero_point = 8;
                    // RHS packing
                    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
                        1, n, k, nr, kr, sr,                     // Packing arguments
                        (const uint8_t*)(rhs_native_mtx_qs4cx),  // RHS
                        NULL,                                    // Bias
                        (const float*)(rhs_scales_f32),          // Scale
                        rhs_packed_mtx_qs4cx,                    // RHS packed
                        0, &nxk_params);
                } else {
                    struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kxn_params;
                    kxn_params.lhs_zero_point = 1;
                    kxn_params.rhs_zero_point = 8;
                    // RHS packing
                    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
                        1, n, k, nr, kr, sr,                     // Packing arguments
                        (const uint8_t*)(rhs_native_mtx_qs4cx),  // RHS
                        NULL,                                    // Bias
                        (const float*)(rhs_scales_f32),          // Scale
                        rhs_packed_mtx_qs4cx,                    // RHS packed
                        0, &kxn_params);
                }

                // LHS packing
                kai_run_lhs_quant_pack_qai8dxp_f32(
                    m, k, mr, kr, sr, 0,               // Packing arguments
                    (const float*)lhs_native_mtx_f32,  // LHS
                    k * sizeof(float),                 // LHS stride
                    lhs_packed_mtx_qa8dx);             // LHS packed

                // Matmul
                {
                    const size_t dst_stride = n * sizeof(float);
                    const size_t lhs_offset = ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
                    const size_t rhs_offset = ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k);
                    const size_t dst_offset = ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

                    const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qa8dx + lhs_offset);
                    const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4cx + rhs_offset);
                    float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 + dst_offset);

                    ukernel_variants[idx_variant].ukernel.run_matmul(
                        m, n, k,           // Dimensions
                        lhs_ptr,           // LHS packed
                        rhs_ptr,           // RHS packed
                        dst_ptr,           // DST
                        dst_stride,        // DST stride (row)
                        sizeof(float),     // DST stride (col)
                        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
                    );
                }

                const bool is_valid =
                    is_output_correct(m, n, 0.0001f, (const float*)dst_ref_mtx_f32, (const float*)dst_act_mtx_f32);

                if (is_valid) {
                    printf("TEST[%ld] = PASSED\n", idx_variant);
                } else {
                    printf("TEST[%ld] = FAILED\n", idx_variant);
                }

                delete[] lhs_packed_mtx_qa8dx;
                delete[] rhs_packed_mtx_qs4cx;
                delete[] dst_act_mtx_f32;
            }
            delete[] lhs_native_mtx_f32;
            delete[] rhs_native_mtx_qs4cx;
            delete[] rhs_scales_f32;
            delete[] dst_ref_mtx_f32;
        }
    }
}

//----------- END MICRO-KERNELS TESTS
//------------------------------------
//------------------------------------

#endif  // Architectural feature check
