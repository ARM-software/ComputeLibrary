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

/// Packs the RHS buffer for static quantized GeMM.
///
/// The RHS matrix must be transposed.
///
/// This function can be used when the following conditions are met:
///   * LHS, RHS and DST data types have the same size and are quantized.
///   * LHS is asymmetric per-tensor, RHS is symmetric per-channel and DST is asymmetric per-tensor.
///
/// @tparam Data The data type of the RHS matrix.
/// @tparam Scale The data type of the quantization scales.
/// @tparam ZeroPoint The data type of the quantization zero points and the operator biases.
///
/// @param[in] data The data buffer of the RHS matrix.
/// @param[in] scales The quantization scales of the RHS matrix.
/// @param[in] lhs_scale The quantization scale of the LHS matrix.
/// @param[in] dst_scale The quantization scale of the DST matrix.
/// @param[in] biases The biases of the operator.
/// @param[in] lhs_zero_point The quantization zero point of the LHS matrix.
/// @param[in] n The number of columns of the non-transposed RHS matrix.
/// @param[in] k The number of rows of the non-transposed RHS matrix.
/// @param[in] block_height The number of rows of a data block (N dimension).
/// @param[in] block_width The number of columns of a data block (K dimension).
///
/// @return The packed RHS.
template <typename Data, typename Scale, typename ZeroPoint>
std::vector<uint8_t> matmul_pack_rhs_nxk_static_quantized(
    const void* data, const void* scales, Scale lhs_scale, Scale dst_scale, const void* biases,
    ZeroPoint lhs_zero_point, size_t n, size_t k, size_t block_height, size_t block_width);

}  // namespace kai::test
