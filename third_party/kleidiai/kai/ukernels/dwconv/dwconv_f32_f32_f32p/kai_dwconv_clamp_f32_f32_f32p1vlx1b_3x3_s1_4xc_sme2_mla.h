//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

/// Micro-kernel dependencies
///
/// -# kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme to pack the RHS tensor

/// Gets maximum number of rows of output data produced
///
/// This is the maximum number of rows of output data produced by this kernel when called once.
///
/// @return Maximum number of rows of output data produced by this kernel.
size_t kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void);

/// Gets the height of the filter.
///
/// This is the filter height of the convolution operation supported by this kernel.
///
/// @return The filter height
size_t kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void);

/// Gets the width of the filter.
///
/// This is the filter width of the convolution operation supported by this kernel.
///
/// @return The filter width
size_t kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void);

/// Gets the kr value
///
/// This is the packing parameter which must be used to pack the RHS tensor.
///
/// @return The kr value
size_t kai_get_kr_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(void);

/// Returns the size of the dst buffer in bytes
///
/// @param[in] dst_height Number of rows in the output tensor
/// @param[in] dst_width Number of columns in the output tensor
/// @param[in] num_channels Number of channels in output tensor
///
/// @return output size in bytes.
size_t kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t dst_height, size_t dst_width, size_t num_channels);

/// Returns an offset in bytes to the dst buffer for given row and stride.
///
/// @param[in] dst_row_idx the row index of the output tensor
/// @param[in] dst_stride_row Output row stride in bytes
///
/// @return offset to element in output/destination tensor.
size_t kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t dst_row_idx, size_t dst_stride_row);

/// Return an offset in bytes to the src buffer for a given row and stride
///
/// @param[in] in_row_idx the row index of the input tensor
/// @param[in] in_stride_row Input row stride in bytes
///
/// @return offset to element in source/input tensor.
size_t kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(size_t in_row_idx, size_t in_stride_row);

/// Runs a depthwise convolution operation followed by a clamp operation
///
/// @param[in]  src	Pointer to the start of valid input row to be processed.
/// @param[in]  rhs_packed Pointer to packed weights
/// @param[in]  dst Pointer to the first element of the top output row for this tile (four rows written)
/// @param[in]  in_stride_row Row stride of input tensor in bytes.
///                           Same as input_w * input_channel when row_dilation = 1
/// @param[in]  in_stride_col Column stride within the input tensor, in bytes.
/// @param[in]  dst_stride_row Output row stride in bytes.
/// @param[in]  dst_stride_col Output column stride in bytes.
/// @param[in]  valid_input_rows Count of real input rows available from the start row (identifies bottom padding).
/// @param[in]  valid_dst_rows Number of rows to output to (1-4).
/// @param[in]  pad_left Number of zero pad columns on the left edge.
/// @param[in]  pad_top Number of zero pad rows that precede the first valid input row.
/// @param[in]  pad_value Fill value for padding. This kernel only supports 0.
/// @param[in]  clamp_min	Lower clamp bound applied to every output value.
/// @param[in]  clamp_max	Upper clamp bound applied to every output value.
void kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    const void* src, const void* rhs_packed, void* dst, size_t in_stride_row, size_t in_stride_col,
    size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
    size_t pad_top, float pad_value, float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
