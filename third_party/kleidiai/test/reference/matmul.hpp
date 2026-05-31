//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"

namespace kai::test {

class DataFormat;

/// Packs the RHS operand of matrix multiplication.
///
/// @param[in] data Data buffer.
/// @param[in] scales (Optional) Quantization scales.
/// @param[in] zero_points (Optional) Quantization zero points.
/// @param[in] src_format Data format of the RHS matrix.
/// @param[in] dst_format Data format of the packed RHS matrix.
/// @param[in] n Number of non-transposed columns.
/// @param[in] k Number of non-transposed rows.
/// @param[in] transposing Perform transpose then pack.
///
/// @return The packed RHS matrix.
Buffer matmul_pack_rhs(
    const void* data, const void* scales, const void* zero_points, const DataFormat& src_format,
    const DataFormat& dst_format, size_t n, size_t k, bool transposing);

/// Matrix multiplication.
///
/// @param[in] lhs LHS operand data.
/// @param[in] lhs_scales (Optional) LHS operand quantization scales.
/// @param[in] lhs_zero_points (Optional) LHS operand quantization zero point.
/// @param[in] lhs_dt LHS operand data type.
/// @param[in] rhs RHS operand data.
/// @param[in] rhs_scales (Optional) RHS operand quantization scales.
/// @param[in] rhs_zero_points (Optional) RHS operand quantization zero point.
/// @param[in] rhs_dt RHS operand data type.
/// @param[in] bias Bias operand data.
/// @param[in] bias_scales (Optional) Bias operand quantization scales.
/// @param[in] bias_zero_points (Optional) Bias operand quantization zero point.
/// @param[in] bias_dt Bias operand data type.
/// @param[in] dst Output data.
/// @param[in] dst_scales (Optional) Output quantization scales.
/// @param[in] dst_zero_points (Optional) Output quantization zero point.
/// @param[in] dst_dt Output data type.
/// @param[in] m Output height.
/// @param[in] n Output width.
/// @param[in] k Non-transposed LHS width and non-transposed RHS height.
/// @param[in] lhs_transposed `true` if LHS operand is transposed.
/// @param[in] rhs_transposed `true` if RHS operand is transposed.
///
/// @return The result data buffer.
Buffer matmul(
    const void* lhs, const void* lhs_scales, const void* lhs_zero_points, DataType lhs_dt,      //
    const void* rhs, const void* rhs_scales, const void* rhs_zero_points, DataType rhs_dt,      //
    const void* bias, const void* bias_scales, const void* bias_zero_points, DataType bias_dt,  //
    DataType dst_dt,                                                                            //
    size_t m, size_t n, size_t k,                                                               //
    bool lhs_transposed, bool rhs_transposed);

/// Indirect matrix multiplication.
///
/// @param[in] lhs_idata The indirect LHS data matrix.
/// @param[in] lhs_scales (Optional) LHS operand quantization scales.
/// @param[in] lhs_offset The indirection LHS data matrix offset, applied to non-padding pointers
/// @param[in] lhs_padding_ptr The indirection LHS padding chunk pointer
/// @param[in] lhs_zero_points (Optional) LHS operand quantization zero point.
/// @param[in] lhs_dt LHS operand data type.
/// @param[in] rhs RHS operand data.
/// @param[in] rhs_scales (Optional) RHS operand quantization scales.
/// @param[in] rhs_zero_points (Optional) RHS operand quantization zero point.
/// @param[in] rhs_dt RHS operand data type.
/// @param[in] bias Bias operand data.
/// @param[in] bias_scales (Optional) Bias operand quantization scales.
/// @param[in] bias_zero_points (Optional) Bias operand quantization zero point.
/// @param[in] bias_dt Bias operand data type.
/// @param[in] dst_dt Output data type.
/// @param[in] m Output height.
/// @param[in] n Output width.
/// @param[in] k_chunk_count Number pointers per row in lhs_idata
/// @param[in] k_chunk_size Number of elements in each LHS K chunk
///
/// @return The result data buffer.
Buffer indirect_matmul(
    const void* const* lhs_idata, uintptr_t lhs_offset, const void* lhs_padding_ptr, const void* lhs_scales,
    const void* lhs_zero_points, DataType lhs_dt,                                               //
    const void* rhs, const void* rhs_scales, const void* rhs_zero_points, DataType rhs_dt,      //
    const void* bias, const void* bias_scales, const void* bias_zero_points, DataType bias_dt,  //
    DataType dst_dt,                                                                            //
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length);

/// Matrix multiplication with quantized input and floating-point output.
///
/// The LHS matrix is non-transposed and the RHS matrix is transposed.
///
/// @tparam LhsData The data type of the LHS matrix.
/// @tparam LhsScale The data type of the quantization scales of the LHS matrix.
/// @tparam LhsZeroPoint The data type of the quantization zero points of the LHS matrix.
/// @tparam Rhsdata The data type of the RHS matrix.
/// @tparam RhsScale The data type of the quantization scales of the RHS matrix.
/// @tparam RhsZeroPoint The data type of the quantization zero points of the RHS matrix.
/// @tparam Bias The data type of the bias vector.
/// @tparam IntAcc The data type of the intermediate integer accumulator.
/// @tparam DstData The data type of the floating-point accumulator and the output matrix.
///
/// @param[in] m The LHS and output height.
/// @param[in] n The RHS height and output width.
/// @param[in] k The LHS and RHS width.
/// @param[in] lhs_data The LHS data matrix.
/// @param[in] lhs_scales The LHS quantization scales matrix.
/// @param[in] lhs_zero_points The LHS quantization zero points matrix.
/// @param[in] lhs_quant_width The LHS quantization block width.
/// @param[in] rhs_data The RHS data matrix.
/// @param[in] rhs_scales The RHS quantization scales matrix.
/// @param[in] rhs_zero_points The RHS quantization zero points matrix.
/// @param[in] rhs_quant_width The RHS quantization block width.
/// @param[in] biases The biases vector.
/// @param[in] min_value The minimum output value.
/// @param[in] max_value The maximum output value.
///
/// @return The output matrix.
template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename Bias, typename IntAcc, typename DstData>
Buffer matmul_clamp_nt_t(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    DstData min_value, DstData max_value);

/// Matrix multiplication with quantized input and floating-point output.
///
/// The LHS matrix is non-transposed and the RHS matrix is non-transposed.
///
/// @tparam LhsData The data type of the LHS matrix.
/// @tparam LhsScale The data type of the quantization scales of the LHS matrix.
/// @tparam LhsZeroPoint The data type of the quantization zero points of the LHS matrix.
/// @tparam Rhsdata The data type of the RHS matrix.
/// @tparam RhsScale The data type of the quantization scales of the RHS matrix.
/// @tparam RhsZeroPoint The data type of the quantization zero points of the RHS matrix.
/// @tparam Bias The data type of the bias vector.
/// @tparam IntAcc The data type of the intermediate integer accumulator.
/// @tparam DstData The data type of the floating-point accumulator and the output matrix.
///
/// @param[in] m The LHS and output height.
/// @param[in] n The RHS height and output width.
/// @param[in] k The LHS and RHS width.
/// @param[in] k_chunk_count Number of K chunk pointers per row in lhs_ptrs matrix
/// @param[in] k_chunk_length Lenght of each K chunk pointed to in lhs_ptrs matrix
/// @param[in] lhs_data The LHS data matrix.
/// @param[in] lhs_ptrs The indirect LHS data matrix.
/// @param[in] lhs_offset The indirection LHS data matrix offset, applied to non-padding pointers
/// @param[in] lhs_padding_ptr The indirection LHS padding chunk pointer
/// @param[in] lhs_scales The LHS quantization scales matrix.
/// @param[in] lhs_zero_points The LHS quantization zero points matrix.
/// @param[in] lhs_quant_width The LHS quantization block width.
/// @param[in] rhs_data The RHS data matrix.
/// @param[in] rhs_scales The RHS quantization scales matrix.
/// @param[in] rhs_zero_points The RHS quantization zero points matrix.
/// @param[in] rhs_quant_width The RHS quantization block width.
/// @param[in] biases The biases vector.
/// @param[in] min_value The minimum output value.
/// @param[in] max_value The maximum output value.
///
/// @return The output matrix.
template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename Bias, typename IntAcc, typename DstData>
Buffer matmul_clamp_nt_nt(
    size_t m, size_t n, size_t k,                                                                       //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_width,  //
    const void* biases,                                                                                 //
    DstData min_value, DstData max_value);

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename BiasData, typename BiasScale, typename BiasZeroPoint, typename DstData>
Buffer matmul_nt_t_quantized(
    size_t m, size_t n, size_t k,  //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_height,
    size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_height,
    size_t rhs_quant_width,  //
    const void* bias_data, const void* bias_scales, const void* bias_zero_points, size_t bias_quant_width);

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename BiasData, typename BiasScale, typename BiasZeroPoint, typename DstData>
Buffer matmul_nt_nt_quantized(
    size_t m, size_t n, size_t k,  //
    const void* lhs_data, const void* lhs_scales, const void* lhs_zero_points, size_t lhs_quant_height,
    size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_height,
    size_t rhs_quant_width,  //
    const void* bias_data, const void* bias_scales, const void* bias_zero_points, size_t bias_quant_width);

template <
    typename LhsData, typename LhsScale, typename LhsZeroPoint, typename RhsData, typename RhsScale,
    typename RhsZeroPoint, typename BiasData, typename BiasScale, typename BiasZeroPoint, typename DstData>
Buffer indirect_matmul_nt_t_quantized(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length,  //
    const void* const* lhs_ptrs, uintptr_t lhs_offset, const void* lhs_padding_ptr, const void* lhs_scales,
    const void* lhs_zero_points, size_t lhs_quant_height,
    size_t lhs_quant_width,  //
    const void* rhs_data, const void* rhs_scales, const void* rhs_zero_points, size_t rhs_quant_height,
    size_t rhs_quant_width,  //
    const void* bias_data, const void* bias_scales, const void* bias_zero_points, size_t bias_quant_width);

}  // namespace kai::test
