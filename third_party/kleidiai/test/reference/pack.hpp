//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kai::test {

class DataFormat;

/// Packs the matrix.
///
/// @param[in] dst_format Data format of the destination matrix.
/// @param[in] src Data buffer of the source matrix.
/// @param[in] src_format Data format of the source matrix.
/// @param[in] height Number of rows of the source matrix.
/// @param[in] width Number of columns of the source matrix.
std::vector<uint8_t> pack(
    const DataFormat& dst_format, const void* src, const void* scales, const void* bias, const DataFormat& src_format,
    size_t height, size_t width);

/// Packs the quantized data and the quantization scale into a single buffer.
///
/// ```
/// Quantized data matrix:
///
///               --->|-----------------|<--- Quantization block width
///                   |                 |
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
///
/// Quantization scale matrix:
///
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// The packed data has each quantization scale followed by the quantized block row.
///
/// ```
/// Packed data:
///
/// +-----+-----------------+-----+-----------------+----- ...
/// | s00 | q00 q01 q02 q03 | s01 | q04 q05 q06 q07 | ........
/// | s10 | q10 q11 q12 q13 | s11 | q14 q15 q16 q17 | ........
/// | s20 | q20 q21 q22 q23 | s21 | q24 q25 q26 q27 | ........
/// | s30 | q30 q31 q32 q33 | s31 | q34 q35 q36 q37 | ........
/// | ... | ............... | ... | ............... | ........
/// : ... : ............... : ... : ............... : ........
/// ```
///
/// @tparam Data The data type of the quantized value.
/// @tparam Scale The data type of the quantization scale.
///
/// @param[in] data The quantized data.
/// @param[in] scales The quantization scales.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The packed data buffer.
template <typename Data, typename Scale>
std::vector<uint8_t> pack_data_scales(
    const void* data, const void* scales, size_t height, size_t width, size_t quant_width);

/// Packs the zero point, data and scale into a single buffer.
///
/// ```
/// Data matrix:
///
/// +-----------------+
/// | q00 q01 q02 q03 |
/// | q10 q11 q12 q13 |
/// | q20 q21 q22 q23 |
/// | q30 q31 q32 q33 |
/// | ............... |
/// : ............... :
///
/// Scales for each row:
///
/// +----+
/// | s0 |
/// | s1 |
/// | s2 |
/// | s3 |
/// | .. |
/// : .. :
///
/// Zero points for each row:
///
/// +----+
/// | z0 |
/// | z1 |
/// | z2 |
/// | z3 |
/// | .. |
/// : .. :
/// ```
///
/// The packed data has each zero point followed by the data row followed by the scale.
///
/// ```
/// Packed data:
///
/// +----+-----------------+----+
/// | z0 | q00 q01 q02 q03 | s0 |
/// | z1 | q10 q11 q12 q13 | s1 |
/// | z2 | q20 q21 q22 q23 | s2 |
/// | z3 | q30 q31 q32 q33 | s3 |
/// | .. | ............... | .. |
/// : .. : ............... : .. :
/// ```
///
/// @tparam Data The data type of the data.
/// @tparam Scale The data type of the scale.
/// @tparam ZeroPoint The data type of the zero point.
///
/// @param[in] data The data buffer.
/// @param[in] scales The scales buffer.
/// @param[in] zero_points The zero points buffer.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
///
/// @return The packed data buffer.
template <typename ZeroPoint, typename Data, typename Scale>
std::vector<uint8_t> pack_zero_points_data_scales_per_block(
    const void* zero_points, const void* data, const void* scales, size_t num_blocks, size_t block_num_zero_points,
    size_t block_num_data, size_t block_num_scales);

/// Packs the quantized data and the quantization scale into a single buffer.
///
/// ```
/// Quantized data matrix:
///
///               --->|-----------------|<--- Quantization block width
///                   |                 |
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
///
/// Quantization scale matrix:
///
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// The packed data has each quantization scale followed by the quantized block row.
///
/// This function is different from @ref pack_data_scales that in this packing method
/// the quantized data row is splitted into two halves and they are interleaved together.
///
/// ```
/// Packed data:
///
/// +-----+-----------------+-----+-----------------+----- ...
/// | s00 | q00 q02 q01 q03 | s01 | q04 q06 q05 q07 | ........
/// | s10 | q10 q12 q11 q13 | s11 | q14 q16 q15 q17 | ........
/// | s20 | q20 q22 q21 q23 | s21 | q24 q26 q25 q27 | ........
/// | s30 | q30 q32 q31 q33 | s31 | q34 q36 q35 q37 | ........
/// | ... | ............... | ... | ............... | ........
/// : ... : ............... : ... : ............... : ........
/// ```
///
/// @tparam Data The data type of the quantized value.
/// @tparam Scale The data type of the quantization scale.
///
/// @param[in] data The quantized data.
/// @param[in] scales The quantization scales.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The packed data buffer.
template <typename Data, typename Scale>
std::vector<uint8_t> pack_data_scales_interleave_block(
    const void* data, const void* scales, size_t height, size_t width, size_t quant_width);

/// Packs the quantized data with two halves of a block interleaved.
///
/// ```
/// Quantized data matrix:
///
///               --->|-----------------|<--- Block width
///                   |                 |
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
///
/// Packed data:
///
/// +-----------------+-----------------+----- ...
/// | q00 q02 q01 q03 | q04 q06 q05 q07 | ........
/// | q10 q12 q11 q13 | q14 q16 q15 q17 | ........
/// | q20 q22 q21 q23 | q24 q26 q25 q27 | ........
/// | q30 q32 q31 q33 | q34 q36 q35 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// @tparam Data The data type of the quantized value.
///
/// @param[in] data The quantized data.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] block_width The number of columns in a block.
///
/// @return The packed data buffer.
template <typename Data>
std::vector<uint8_t> pack_data_interleave_block(const void* data, size_t height, size_t width, size_t block_width) {
    return pack_data_scales_interleave_block<Data, std::nullptr_t>(data, nullptr, height, width, block_width);
}

}  // namespace kai::test
