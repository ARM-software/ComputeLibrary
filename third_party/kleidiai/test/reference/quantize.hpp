//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace kai::test {

/// Quantization method.
enum class QuantizationMethod : uint32_t {
    PER_MATRIX,  ///< Per-matrix, i.e. one quantization scale and zero point for the entire matrix.
    PER_ROW,     ///< Per-row, i.e. one quantization scale and zero point for each row.
};

/// Quantized a float value to an integer datatype using a provided scale.
///
/// @tparam IntType Quantized integer datatype.
///
/// @param[in] float The value to quantize
/// @param[in] scale The scale used to quantize the provided float value.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename IntType>
IntType quantize_symmetric(float value, float scale);

template <typename FloatType, typename IntType, typename ZeroPointType>
IntType quantize_asymmetric(FloatType value, FloatType scale, ZeroPointType zero_point);

/// Computes the quantization information using symmetric per-block quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not affect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// ```
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data           Scale
/// +-----------------+          +-----+
/// | f00 f01 f02 f03 | -------> | s00 |
/// | f10 f11 f12 f13 | -------> | s10 |
/// | f20 f21 f22 f23 | -------> | s20 |
/// | f30 f31 f32 f33 | -------> | s30 |
/// | ............... |          | ... |
/// : ............... :          : ... :
/// ```
///
/// The computed quantization scale matrix:
///
/// ```
/// +-----+-----+-- ...
/// | s00 | s01 | .....
/// | s10 | s11 | .....
/// | s20 | s21 | .....
/// | s30 | s31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantization scale matrix.
template <typename SrcType, typename DstType, typename ScaleType>
std::vector<uint8_t> compute_symmetric_per_block_quantization_info(
    const void* src, size_t height, size_t width, size_t quant_width);

/// Quantizes each block of the matrix using symmetric quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not affect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// The quantization scale matrix can be calculated using
/// @ref compute_symmetric_per_block_quantization_info function.
///
/// The input matrix and the quantization scale matrix:
///
/// ```
///              Floating-point data                            Scale
///
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...       +-----+-----+-- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........       | s00 | s01 | .....
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........       | s10 | s11 | .....
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........       | s20 | s21 | .....
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........       | s30 | s31 | .....
/// | ............... | ............... | ........       | ... | ... | .....
/// : ............... : ............... : ........       : ... : ... : .....
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data        Scale             Quantized data
/// +-----------------+       +-----+          +-----------------+
/// | f00 f01 f02 f03 |       | s00 | -------> | q00 q01 q02 q03 |
/// | f10 f11 f12 f13 |       | s10 | -------> | q10 q11 q12 q13 |
/// | f20 f21 f22 f23 |       | s20 | -------> | q20 q21 q22 q23 |
/// | f30 f31 f32 f33 |       | s30 | -------> | q30 q31 q32 q33 |
/// | ............... |       | ... | -------> | ............... |
/// : ............... :       : ... :          : ............... :
/// ```
///
/// The computed quantized data matrix:
///
/// ```
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] scales The quantization scale matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix.
template <typename SrcType, typename DstType, typename ScaleType>
std::vector<uint8_t> quantize_symmetric_per_block(
    const void* src, const void* scales, size_t height, size_t width, size_t quant_width);

/// Dynamically quantizes each block of the matrix using symmetric quantization method.
///
/// The quantization information is calculated using
/// @ref compute_symmetric_per_block_quantization_info function.
/// The floating-point data is then quantized using
/// @ref quantize_symmetric_per_block function.
///
/// To retain highest quantization accuracy, the data is quantized using the quantization scale
/// with the same data type as the input data.
/// After that the quantization scale can be stored in the buffer using `ScaleType` data type
/// which might have lowest precision than the input data type.
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix and the quantization scale matrix.
template <typename SrcType, typename DstType, typename ScaleType>
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> quantize_symmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width);

/// Computes the quantization information using asymmetric per-block quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not affect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// ```
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data           Scale       Zero point
/// +-----------------+          +-----+       +-----+
/// | f00 f01 f02 f03 | -------> | s00 |       | z00 |
/// | f10 f11 f12 f13 | -------> | s10 |       | z10 |
/// | f20 f21 f22 f23 | -------> | s20 |       | z20 |
/// | f30 f31 f32 f33 | -------> | s30 |       | z30 |
/// | ............... |          | ... |       | ... |
/// : ............... :          : ... :       : ... :
/// ```
///
/// The computed quantization scales and zero points matrices:
///
/// ```
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
/// Quantization zero point matrix:
///
/// +-----+-----+-- ...
/// | z00 | z01 | .....
/// | z10 | z11 | .....
/// | z20 | z21 | .....
/// | z30 | z31 | .....
/// | ... | ... | .....
/// : ... : ... : .....
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
/// @tparam ZeroPointType The data type of the quantization zero points (must be integer).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantization scale matrix and the quantization zero point matrix.
template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> compute_asymmetric_per_block_quantization_info(
    const void* src, size_t height, size_t width, size_t quant_width);

/// Quantizes each block of the matrix using asymmetric quantization method.
///
/// The input matrix is divided into quantization blocks of the same size.
///
/// The height of the block does not affect the behavior of this function hence it is omitted
/// from the function arguments and the figures below.
///
/// The quantization scale and zero point matrix can be calculated using
/// @ref compute_asymmetric_per_block_quantization_info function.
///
/// The input matrix, quantization scale matrix and quantization zero matrix:
///
/// ```
///              Floating-point data                            Scale                  Zero point
///
/// Quantization blocks -------+
///          |                 |
///          |                 |
///          v                 v
/// +-----------------+-----------------+----- ...       +-----+-----+-- ...       +-----+-----+-- ...
/// | f00 f01 f02 f03 | f04 f05 f06 f07 | ........       | s00 | s01 | .....       | z00 | z01 | .....
/// | f10 f11 f12 f13 | f14 f15 f16 f17 | ........       | s10 | s11 | .....       | z10 | z11 | .....
/// | f20 f21 f22 f23 | f24 f25 f26 f27 | ........       | s20 | s21 | .....       | z20 | z21 | .....
/// | f30 f31 f32 f33 | f34 f35 f36 f37 | ........       | s30 | s31 | .....       | z30 | z31 | .....
/// | ............... | ............... | ........       | ... | ... | .....       | ... | ... | .....
/// : ............... : ............... : ........       : ... : ... : .....       | ... | ... | .....
/// ```
///
/// Each row of the quantization block is quantized individually.
///
/// ```
/// Floating-point data        Scale       Zero point          Quantized data
/// +-----------------+       +-----+       +-----+          +-----------------+
/// | f00 f01 f02 f03 |       | s00 |       | z00 | -------> | q00 q01 q02 q03 |
/// | f10 f11 f12 f13 |       | s10 |       | z10 | -------> | q10 q11 q12 q13 |
/// | f20 f21 f22 f23 |       | s20 |       | z20 | -------> | q20 q21 q22 q23 |
/// | f30 f31 f32 f33 |       | s30 |       | z30 | -------> | q30 q31 q32 q33 |
/// | ............... |       | ... |       | ... | -------> | ............... |
/// : ............... :       : ... :       : ... :          : ............... :
/// ```
///
/// The computed quantized data matrix:
///
/// ```
/// +-----------------+-----------------+----- ...
/// | q00 q01 q02 q03 | q04 q05 q06 q07 | ........
/// | q10 q11 q12 q13 | q14 q15 q16 q17 | ........
/// | q20 q21 q22 q23 | q24 q25 q26 q27 | ........
/// | q30 q31 q32 q33 | q34 q35 q36 q37 | ........
/// | ............... | ............... | ........
/// : ............... : ............... : ........
/// ```
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
/// @tparam ZeroPointType The data type of the quantization zero points (must be integer).
///
/// @param[in] src The input matrix.
/// @param[in] scales The quantization scale matrix.
/// @param[in] zero_points The quantization zero point matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix.
template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::vector<uint8_t> quantize_asymmetric_per_block(
    const void* src, const void* scales, const void* zero_points, size_t height, size_t width, size_t quant_width);

/// Dynamically quantizes each block of the matrix using asymmetric quantization method.
///
/// The quantization information is calculated using
/// @ref compute_asymmetric_per_block_quantization_info function.
/// The floating-point data is then quantized using
/// @ref quantize_asymmetric_per_block function.
///
/// To retain highest quantization accuracy, the data is quantized using the quantization scale
/// with the same data type as the input data.
/// After that the quantization scale can be stored in the buffer using `ScaleType` data type
/// which might have lowest precision than the input data type.
///
/// @tparam SrcType The data type of the input data (must be floating-point).
/// @tparam DstType The data type of the output data (must be integer).
/// @tparam ScaleType The data type of the quantization scales (must be floating-point).
/// @tparam ZeroPointType The data type of the quantization zero points (must be integer).
///
/// @param[in] src The input matrix.
/// @param[in] height The number of rows.
/// @param[in] width The number of columns.
/// @param[in] quant_width The number of columns of the quantization block.
///
/// @return The quantized data matrix, the quantization scale matrix and the quantization zero point matrix.
template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, std::vector<uint8_t>> quantize_asymmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width);

}  // namespace kai::test
