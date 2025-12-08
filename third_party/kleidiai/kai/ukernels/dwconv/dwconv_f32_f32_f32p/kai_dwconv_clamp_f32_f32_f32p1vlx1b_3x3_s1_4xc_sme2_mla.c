//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Number of rows iterated through each call.
static const size_t kai_mr = 4;
static const size_t kai_filter_height = 3;
static const size_t kai_filter_width = 3;
static const size_t kai_kr = 1;

typedef struct {
    const void* src;
    size_t pad_top;
    size_t pad_bottom;
    size_t input_cols;
    size_t output_cols;
    void** outptrs;
    const void* output_cols_stride_in_elements;
    size_t input_vl_stride_in_elements;
    const void* output_vls_stride_in_elements;
    size_t pad_left;
    float clamp_min;
    float clamp_max;
    const void* rhs_packed;
    size_t current_channel;
    size_t n_channels;
} KernelArgs;

void kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    const KernelArgs* args, size_t input_row_stride_in_elements, size_t input_col_stride_in_elements);

size_t kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void) {
    return kai_mr;
}

size_t kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void) {
    return kai_filter_height;
}

size_t kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void) {
    return kai_filter_width;
}

size_t kai_get_kr_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void) {
    return kai_kr;
}

size_t kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t dst_height, size_t dst_width, size_t num_channels) {
    return dst_height * dst_width * num_channels * sizeof(float);
}

size_t kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t dst_row_idx, size_t dst_stride_row) {
    KAI_ASSUME(dst_row_idx % kai_mr == 0);
    return (dst_row_idx * dst_stride_row);
}

size_t kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(size_t in_row_idx, size_t in_stride_row) {
    return (in_row_idx * in_stride_row);
}

void kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    const void* src, const void* rhs_packed, void* dst, size_t in_stride_row, size_t in_stride_col,
    size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
    size_t pad_top, float pad_value, float clamp_min, float clamp_max) {
    KAI_ASSUME(src != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(dst != NULL);
    KAI_ASSUME(valid_dst_rows != 0);
    KAI_ASSUME(pad_value == 0.0F);
    KAI_ASSUME(dst_stride_col == in_stride_col);

    // Create padding row.
    float pad_row[KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(float)] = {0};

    // Calculate bottom padding offset.
    const size_t in_rows = 6U;  // Max number of input rows processed in a single kernel call (assuming all are valid)
    size_t pad_bottom = (in_rows < (pad_top + valid_input_rows)) ? (0U) : (in_rows - (pad_top + valid_input_rows));

    // Leading dims calculated using input parameters.
    size_t input_vl_stride_in_elements = kai_get_sme_vector_length_u32();
    size_t input_row_stride_in_elements = in_stride_row / sizeof(float);
    size_t input_col_stride_in_elements = in_stride_col / sizeof(float);

    // Calculate tensor dimensions using the strides provided.
    const size_t num_channels = dst_stride_col / sizeof(float);
    const size_t output_cols = dst_stride_row / (sizeof(float) * num_channels);
    const size_t valid_input_cols = in_stride_row / (sizeof(float) * num_channels);

    // These arrays are initilised as if they were invalid/padded rows, then set if out row is valid
    // Array size corresponds to kai_mr
    void* outptrs[4] = {pad_row, pad_row, pad_row, pad_row};
    size_t outlds[4] = {0};
    size_t outvllds[4] = {0};

    for (size_t i = 0; i < 4; i++) {
        if (i < valid_dst_rows) {
            outptrs[i] = (uint8_t*)dst + (i * dst_stride_row);
            outlds[i] = num_channels;
            outvllds[i] = input_vl_stride_in_elements;
        }
    }

    KernelArgs args;
    args.src = src;
    args.input_vl_stride_in_elements = input_vl_stride_in_elements;
    args.pad_top = pad_top;
    args.pad_bottom = pad_bottom;
    args.pad_left = pad_left;
    args.input_cols = valid_input_cols;
    args.output_cols = output_cols;
    args.outptrs = outptrs;
    args.output_cols_stride_in_elements = outlds;
    args.output_vls_stride_in_elements = outvllds;
    args.current_channel = 0;
    args.n_channels = num_channels;
    args.clamp_min = clamp_min;
    args.clamp_max = clamp_max;
    args.rhs_packed = rhs_packed;

    kai_commit_za();

    kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
        &args, input_row_stride_in_elements, input_col_stride_in_elements);
}
#endif  // Architectural features check.
