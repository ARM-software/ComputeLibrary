//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <test/common/data_type.hpp>

#include "imatmul_interface.hpp"
#include "kai/kai_common.h"

namespace kai::benchmark {

using DataType = test::DataType;

/// Runner for the indirect matrix multiplication micro-kernel (imatmul).
///
/// Prepares and executes the run method of the imatmul micro-kernel.
///
/// @tparam IndirectMatMulInterface Interface of the indirect matrix multiplication micro-kernel.
template <typename IndirectMatMulInterface>
class ImatmulRunner {
public:
    /// Constructs an ImatmulRunner object.
    ///
    /// @param imatmul_interface Abstraction containing the micro-kernel to run.
    /// @param dst_type Output type of the micro-kernel. Required for the micro-kernel to make certain assumptions
    /// internally about the stride of the data.
    ImatmulRunner(const IndirectMatMulInterface& imatmul_interface, const DataType dst_type) :
        m_imatmul_interface(imatmul_interface), m_dst_type(dst_type) {
        set_mnk_chunked(m_m, m_n, m_k_chunk_count, m_k_chunk_length);
    }

    /// Sets the M, N and chunked K dimensions for imatmul micro-kernels.
    ///
    /// @param m Number of rows in the LHS and DST matrices.
    /// @param n Number of columns in the RHS and DST matrices.
    /// @param k_chunk_count Number of K chunks (for chunked K dimension).
    /// @param k_chunk_length Length of each K chunk.
    void set_mnk_chunked(size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length) {
        m_m = m;
        m_n = n;
        m_k_chunk_count = k_chunk_count;
        m_k_chunk_length = k_chunk_length;
        m_dst_stride_row = m_n * data_type_size_in_bits(m_dst_type) / 8;
        m_dst_stride_col = data_type_size_in_bits(m_dst_type) / 8;
    }

    /// Runs the indirect matrix multiplication micro-kernel.
    ///
    /// @param lhs Buffer containing LHS matrix data.
    /// @param rhs Buffer containing RHS matrix data.
    /// @param dst Destination buffer to write to.
    void run(const void* lhs, const void* rhs, void* dst);

private:
    IndirectMatMulInterface m_imatmul_interface = {};
    DataType m_dst_type = DataType::FP32;
    size_t m_m = 1;
    size_t m_n = 1;
    size_t m_k_chunk_count = 1;
    size_t m_k_chunk_length = 1;
    size_t m_dst_stride_row = 1;
    size_t m_dst_stride_col = 1;
};

/// Default run method for imatmul micro-kernels (ImatmulBaseInterface)
template <typename IndirectMatMulInterface>
void ImatmulRunner<IndirectMatMulInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_imatmul_interface.run_imatmul(
        m_m, m_n, m_k_chunk_count, m_k_chunk_length, lhs, rhs, dst, m_dst_stride_row, -FLT_MAX, FLT_MAX);
}

/// Specialized run method for static quantization interface (ImatmulStaticQuantInterface)
template <>
inline void ImatmulRunner<ImatmulStaticQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    constexpr kai_matmul_requantize32_params params = {INT8_MIN, INT8_MAX, 0};
    m_imatmul_interface.run_imatmul(
        m_m, m_n, m_k_chunk_count, m_k_chunk_length, lhs, rhs, dst, m_dst_stride_row, &params);
}

}  // namespace kai::benchmark
