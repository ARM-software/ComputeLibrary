//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Get the size in bytes of the packed rhs data buffer.
///
/// @param[in]  filter_height The height of filter being used in convolution.
/// @param[in]  filter_width  The width of filter being used in convolution.
/// @param[in]  num_channels  Number of channels in input tensor.
///
/// @return  The size in bytes of packed data buffer.
size_t kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    size_t filter_height, size_t filter_width, size_t num_channels);

/// Runs the RHS packing function for the depthwise convolution kernel.
///
/// NOTE: filter_height/filter_width is seperate from height/width of weights intending to allow for padding when using
///       weights shapes different to kernel conv filter size (not yet implemented). These should be the same in typical
///       usecases.
///
/// @param[in]  filter_height The height of filter being used in convolution.
/// @param[in]  filter_width  The width of filter being used in convolution.
/// @param[in]  height Height dimension of rhs tensor. Unused. (Typically equivalent to filter_height)
/// @param[in]  width  Width dimension of rhs tensor. Unused. (Typically equivalent to filter_width)
/// @param[in]  num_channels Number of channels in input tensor.
/// @param[in]  rhs Rhs tensor data buffer
/// @param[in]  bias   Bias data buffer.
/// @param[out] rhs_packed Packed data tensor buffer
void kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    size_t filter_height, size_t filter_width, size_t height, size_t width, size_t num_channels, const void* rhs,
    const void* bias, void* rhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
