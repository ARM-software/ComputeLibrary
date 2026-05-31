//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <test/common/data_type.hpp>

#include "kai/kai_common.h"
#include "matmul_interface.hpp"

namespace kai::benchmark {

using DataType = test::DataType;

/// Runner for the matrix multiplication micro-kernel.
///
/// Prepares and executes the run method of the micro-kernel.
///
/// @tparam MatMulInterface Interface of the matrix multiplication micro-kernel.
template <typename MatMulInterface>
class MatMulRunner {
public:
    /// Constructs a MatMulRunner object.
    ///
    /// @param matmul_interface Abstraction containing the micro-kernel to run.
    /// @param dst_type Output type of the micro-kernel. Required for the micro-kernel to make certain assumptions
    /// internally about the stride of the data.
    MatMulRunner(const MatMulInterface& matmul_interface, const DataType dst_type) :
        matmul_interface_(matmul_interface), dst_type_(dst_type) {
    }

    /// Sets the M, N and K dimensions to describe the operand and result matrices.
    ///
    /// @param m Rows in a non-transposed LHS and DST matrix.
    /// @param n Columns in a non-transposed RHS and DST matrix.
    /// @param k Columns in a non-transposed LHS matrix, and rows in a non-transposed RHS matrix.
    void set_mnk(const size_t m, const size_t n, const size_t k) {
        m_ = m;
        n_ = n;
        k_ = k;

        lhs_stride_ = k_ * data_type_size_in_bits(dst_type_) / 8;
        dst_stride_row_ = n_ * data_type_size_in_bits(dst_type_) / 8;
        dst_stride_col_ = data_type_size_in_bits(dst_type_) / 8;
    }

    /// Sets the block size to use.
    ///
    /// @param bl Block size. Used for micro-kernels with dynamic blockwise quantization.
    void set_bl(const size_t bl) {
        bl_ = bl;
    }

    /// Runs the matrix multiplication micro-kernel.
    ///
    /// @param lhs Buffer containing LHS matrix data.
    /// @param rhs Buffer containing RHS matrix data.
    /// @param dst Destination buffer to write to.
    void run(const void* lhs, const void* rhs, void* dst);

private:
    MatMulInterface matmul_interface_ = {};

    DataType dst_type_ = DataType::FP32;

    size_t m_ = 1;
    size_t n_ = 1;
    size_t k_ = 1;
    size_t bl_ = 32;

    size_t lhs_stride_ = 1;
    size_t dst_stride_row_ = 1;
    size_t dst_stride_col_ = 1;
};

/// Runs the matrix multiplication micro-kernel.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <typename MatMulInterface>
void MatMulRunner<MatMulInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the strided LHS interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulStridedLhsInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, lhs_stride_, rhs, dst,        //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the interface with a floating point destination buffer.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulFloatInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_,                          //
        lhs, rhs, static_cast<float*>(dst),  //
        dst_stride_row_, dst_stride_col_,    //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the static quantization interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulStaticQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    constexpr kai_matmul_requantize32_params params = {INT8_MIN, INT8_MAX, 0};
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        &params                            //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface with
/// generic destination buffer.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantGenericDstInterface>::run(
    const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_, bl_,                   //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_, bl_,                     //
        lhs, rhs, static_cast<float*>(dst),  //
        dst_stride_row_, dst_stride_col_,    //
        -FLT_MAX, FLT_MAX                    //
    );
}

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP
