//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>
#include <tuple>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/data_format.hpp"
#include "test/common/float16.hpp"
#include "test/common/matrix_portion.hpp"

namespace kai::test {
/// Matrix multiplication shape.
struct MatMulShape {
    size_t m;  ///< LHS height.
    size_t n;  ///< RHS width.
    size_t k;  ///< LHS width and RHS height.

    struct Hash {
        size_t operator()(const MatMulShape& shape) const {
            return                                     //
                (std::hash<size_t>{}(shape.m) << 0) ^  //
                (std::hash<size_t>{}(shape.n) << 1) ^  //
                (std::hash<size_t>{}(shape.k) << 2);   //
        }
    };

private:
    friend bool operator==(const MatMulShape& lhs, const MatMulShape& rhs) {
        return                 //
            lhs.m == rhs.m &&  //
            lhs.n == rhs.n &&  //
            lhs.k == rhs.k;
    }
    friend std::ostream& operator<<(std::ostream& os, const MatMulShape& shape);
};

/// Value range
template <typename T>
struct Range {
    T min;
    T max;

    [[nodiscard]] T range() const {
        return max - min;
    }
};

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)

/// Matrix multiplication method.
struct MatMulMethod {
    std::string_view name{};  ///< Name of matmul method.

    size_t m0{0};  ///< Block size in M dimension.
    size_t n0{0};  ///< Block size in N dimension.
    size_t k0{0};  ///< Block size in K dimension.

    DataFormat dst_format{};         ///< Data format of the destination matrix.
    DataFormat lhs_format{};         ///< Data format of the LHS matrix.
    DataFormat packed_lhs_format{};  ///< Data format of the packed LHS matrix.
    DataFormat rhs_format{};         ///< Data format of the RHS matrix.
    DataFormat packed_rhs_format{};  ///< Data format of the packed RHS matrix.
    DataFormat bias_format{};        ///< Data format of the bias vector.
    bool nb_support{};               ///< Does the kernel support null_bias.

    /// Generate LHS matrix.
    ///
    /// @param[in] m Number of rows in the LHS matrix.
    /// @param[in] k Number of columns in the LHS matrix.
    ///
    /// @return LHS matrix data buffer.
    std::function<Buffer(size_t, size_t)> fn_generate_lhs{nullptr};

    /// Generate RHS matrix.
    ///
    /// @param[in] k Number of rows in the RHS matrix.
    /// @param[in] n Number of columns in the RHS matrix.
    ///
    /// @return RHS matrix data buffer.
    std::function<Buffer(size_t, size_t)> fn_generate_rhs{nullptr};

    /// Generate bias.
    ///
    /// @param[in] n Number of rows in the bias.
    /// @param[in] k Number of columns in the bias.
    ///
    /// @return Bias data buffer.
    std::function<Buffer(size_t, size_t)> fn_generate_bias{nullptr};

    /// Check if CPU supports required features.
    ///
    /// @return Supported (true) or not supported (false).
    std::function<bool(void)> fn_is_supported{nullptr};

    /// Gets mr value.
    ///
    /// This is the packing parameter which must be used to pack the LHS matrix (if necessary).
    ///
    /// @return The mr value.
    std::function<size_t(void)> fn_get_mr{nullptr};

    /// Gets nr value.
    ///
    /// This is the packing parameter which must be used to pack the RHS matrix (if necessary).
    ///
    /// @return The nr value.
    std::function<size_t(void)> fn_get_nr{nullptr};

    /// Gets kr value.
    ///
    /// This is the packing parameter which must be used to pack the LHS and RHS matrix (if necessary).
    ///
    /// @return The kr value.
    std::function<size_t(void)> fn_get_kr{nullptr};

    /// Gets sr value.
    ///
    /// This is the packing parameter which must be used to pack the RHS matrix.
    ///
    /// @return The sr value.
    std::function<size_t(void)> fn_get_sr{nullptr};

    /// Gets m step value for main kernel.
    ///
    /// The starting row index must be divisible by `m_step`.
    ///
    /// @return The m step value.
    std::function<size_t(void)> fn_get_main_m_step{nullptr};

    /// Gets n step value for RHS packing micro-kernel.
    ///
    /// The starting row index must be divisible by `n_step`.
    ///
    /// @return The n step value.
    std::function<size_t(void)> fn_get_pack_rhs_n_step{nullptr};

    /// Gets n step value for main kernel.
    ///
    /// The starting column index must be divisible by `n_step`.
    ///
    /// @return The n step value.
    std::function<size_t(void)> fn_get_main_n_step{nullptr};

    /// Gets the offset in bytes of the LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t stride)> fn_get_lhs_offset{nullptr};

    /// Gets the size in bytes of the packed LHS matrix.
    ///
    /// @param[in] m Number of rows in the unpacked LHS matrix.
    /// @param[in] k Number of columns in the unpacked LHS matrix.
    /// @param[in] mr Number of rows to be interleaved.
    /// @param[in] kr Unused. Must be 1.
    /// @param[in] sr Unused. Must be 1.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t m, size_t k, size_t mr, size_t kr, size_t sr)> fn_get_packed_lhs_size{nullptr};

    /// Gets the offset in bytes of the packed LHS matrix.
    ///
    /// @param[in] m_idx Coordinate of the matrix in M dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t m_idx, size_t k)> fn_get_packed_lhs_offset{nullptr};

    /// Preprocesses the LHS matrix.
    ///
    /// @param[in] m Number of rows of the unpacked LHS matrix.
    /// @param[in] k Common dimension between the LHS and RHS matrix.
    /// @param[in] mr Block size in M dimension. It must be {{ kernel.interleave_by }}VL.
    /// @param[in] kr Block size in K dimension. It must be {{ kernel.block_by }}.
    /// @param[in] sr Number of kr splits. It must be 1.
    /// @param[in] m_idx_start Unused. Must be 0.
    /// @param[in] lhs LHS matrix data buffer.
    /// @param[in] lhs_stride Row stride in bytes of the LHS matrix.
    /// @param[out] lhs_packed Packed RHS matrix.
    std::function<void(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
        void* lhs_packed)>
        fn_pack_lhs{nullptr};

    /// Gets a value indicating whether LHS packing is needed.
    [[nodiscard]] bool is_pack_lhs_needed() const {
        return fn_pack_lhs != nullptr;
    }

    /// Gets the offset in bytes of the RHS matrix.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx)> fn_get_rhs_offset{nullptr};

    /// Gets the size in bytes of the packed RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t n, size_t k)> fn_get_packed_rhs_size{nullptr};

    /// Gets the size in bytes of the packed RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] nr Block size in N dimension.
    /// @param[in] kr Block size in K dimension.
    ///
    /// @return The size in bytes.
    std::function<size_t(size_t n, size_t k, size_t nr, size_t kr)> fn_get_packed_rhs_size_generic_block_size = nullptr;

    /// Gets the offset in bytes of the packed RHS matrix in the RHS packing micro-kernel
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t k)> fn_get_pack_rhs_packed_rhs_offset{nullptr};

    /// Gets the offset in bytes of the packed RHS matrix in the main kernel.
    ///
    /// @param[in] n_idx Coordinate of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    ///
    /// @return The offset in bytes.
    std::function<size_t(size_t n_idx, size_t k)> fn_get_main_packed_rhs_offset{nullptr};

    std::function<void(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params)>
        fn_pack_rhs{nullptr};

    /// Gets n step value.
    ///
    /// The starting row index must be divisible by `n_step`.
    ///
    /// @return The n step value.
    std::function<size_t()> fn_pack_rhs_nxk_get_n_step{nullptr};

    /// Gets the offset in bytes to the data element in the RHS matrix buffer.
    ///
    /// @param[in] n_idx Column index.
    /// @param[in] rhs_offset Row stride in bytes of the RHS matrix.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t n_idx, size_t rhs_stride)> fn_pack_rhs_nxk_get_rhs_offset{nullptr};

    /// Gets the offset in bytes to the data element in the bias buffer.
    ///
    /// @param[in] n_idx Column index.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t n_idx)> fn_pack_rhs_nxk_get_bias_offset{nullptr};

    /// Gets the offset in bytes to the data element in the packed RHS buffer.
    ///
    /// @param[in] n_idx Row index.
    /// @param[in] k Number of columns.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t n_idx, size_t k)> fn_pack_rhs_nxk_get_packed_rhs_offset{nullptr};

    /// Gets the size in bytes of the packed RHS buffer.
    ///
    /// @param[in] n Number of rows.
    /// @param[in] k Number of columns.
    ///
    /// @return The size in bytes of the packed RHS buffer.
    std::function<size_t(size_t n, size_t k)> fn_pack_rhs_nxk_get_packed_rhs_size{nullptr};

    /// Runs the RHS packing micro-kernel for matrix multiplication.
    ///
    /// The pointer of each buffers (RHS, bias and packed RHS) needs to be added with offset
    /// calculated using the following functions:
    ///
    ///   * RHS: @ref kai_get_rhs_offset_rhs_pack_nxk_f32p2vlx1b_f32_f32_sme.
    ///   * Bias: @ref kai_get_bias_offset_rhs_pack_nxk_f32p2vlx1b_f32_f32_sme.
    ///   * Output: @ref kai_get_rhs_packed_offset_rhs_pack_nxk_f32p2vlx1b_f32_f32_sme.
    ///
    /// @param[in] num_groups Number of groups. It must be 1.
    /// @param[in] n Number of columns of the output matrix.
    /// @param[in] k Common dimension between the LHS and RHS matrix.
    /// @param[in] nr Block size in N dimension. It must be 2 * kai_get_sme_vector_length_u32().
    /// @param[in] kr Block size in K dimension. It must be 1.
    /// @param[in] sr Number of kr splits. It must be 1.
    /// @param[in] rhs_stride Row stride in bytes of the RHS matrix.
    /// @param[in] rhs RHS matrix data buffer.
    /// @param[in] bias Bias matrix data buffer.
    /// @param[in] scale Scale data buffer. It must be NULL.
    /// @param[out] rhs_packed Packed RHS matrix.
    /// @param[in] extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix. It must be 0.
    /// @param[in] params Extra packing parameters. It must be NULL.
    std::function<void(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
        const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params)>
        fn_pack_rhs_nxk{nullptr};

    /// Gets the offset in bytes to the data element in the bias buffer.
    ///
    /// @param[in] n_idx Column index.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t n_idx)> fn_get_bias_offset{nullptr};

    /// Gets the offset in bytes to the data element in the destination matrix buffer.
    ///
    /// @param[in] m_idx Row index.
    /// @param[in] n_idx Column index.
    /// @param[in] stride Row stride in bytes.
    ///
    /// @return The offset in bytes to the data element.
    std::function<size_t(size_t m_idx, size_t n_idx, size_t stride)> fn_get_dst_offset{nullptr};

    /// Gets the size in bytes of the destination matrix buffer.
    ///
    /// @param[in] m Number of rows.
    /// @param[in] n Number of columns.
    ///
    /// @return The size in bytes of the destination matrix buffer.
    std::function<size_t(size_t m, size_t n)> fn_get_dst_size{nullptr};

    /// Performs F16 or F32 matrix multiplication with RHS packing
    /// followed by clamp operation.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] lhs LHS data buffer.
    /// @param[in] packed_rhs Packed RHS data buffer.
    /// @param[out] dst Output data buffer.
    /// @param[in] lhs_stride LHS row stride.
    /// @param[in] dst_stride_row Output row stride.
    /// @param[in] dst_stride_col Output column stride.
    /// @param[in] clamp_min Lower bound of the output data.
    /// @param[in] clamp_max Upper bound of the output data.
    std::function<void(
        size_t m, size_t n, size_t k,                             //
        const void* lhs, size_t lhs_stride,                       //
        const void* packed_rhs,                                   //
        void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max)>
        fn_matmul_f16_f16_f16p = nullptr;

    std::function<void(
        size_t m, size_t n, size_t k,                             //
        const void* lhs, size_t lhs_stride,                       //
        const void* packed_rhs,                                   //
        void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max)>
        fn_matmul_f32_f32_f32p = nullptr;

    /// Performs BF16 matrix multiplication with LHS and RHS packing
    /// followed by clamp operation.
    ///
    /// @param[in] m Size of the matrix in M dimension.
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] packed_lhs Packed LHS data buffer.
    /// @param[in] packed_rhs Packed RHS data buffer.
    /// @param[out] dst Output data buffer.
    /// @param[in] dst_stride_row Output row stride.
    /// @param[in] dst_stride_col Output column stride.
    /// @param[in] clamp_min Lower bound of the output data.
    /// @param[in] clamp_max Upper bound of the output data.
    std::function<void(
        size_t m, size_t n, size_t k,                             //
        const void* packed_lhs,                                   //
        const void* packed_rhs,                                   //
        void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max)>
        fn_matmul_f32_bf16p_bf16p = nullptr;

    std::function<void(
        size_t m, size_t n, size_t k,                             //
        const void* packed_lhs,                                   //
        const void* packed_rhs,                                   //
        void* dst, size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max)>
        fn_matmul_f16_bf16p_bf16p = nullptr;

    /// Performs F16 or F32 matrix multiplication with LHS & RHS packing
    /// followed by clamp operation.
    ///
    /// @param[in] m Number of output rows to be computed.
    /// @param[in] n Number of output columns to be computed.
    /// @param[in] k Common dimension of the LHS and RHS operands.
    /// @param[in] packed_lhs Packed LHS matrix buffer.
    /// @param[in] packed_rhs Packed RHS matrix buffer.
    /// @param[out] dst Output matrix buffer.
    /// @param[in] dst_stride_row Row stride in bytes of the output matrix.
    /// @param[in] dst_stride_col Column stride in bytes of the output matrix.
    /// @param[in] clamp_min Minimum value to clamp the final result.
    /// @param[in] clamp_max Maximum value to clamp the final result.
    std::function<void(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, float clamp_min, float clamp_max)>
        fn_matmul_f16_f16p_f16p = nullptr;

    std::function<void(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
        size_t dst_stride_col, float clamp_min, float clamp_max)>
        fn_matmul_f32_f32p_f32p = nullptr;

    /// Gets a value indicating whether pre-processing the RHS matrix is needed.
    [[nodiscard]] bool is_pack_rhs_needed() const {
        return fn_pack_rhs != nullptr;
    }

    /// Gets a value indicating whether pre-processing the transposed RHS matrix is needed.
    [[nodiscard]] bool is_pack_rhs_nxk_needed() const {
        return fn_pack_rhs_nxk != nullptr;
    }

    /// Preprocesses the RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] rhs RHS data buffer.
    /// @param[in] rhs_row_stride RHS row stride.
    /// @param[in] bias Bias data buffer.
    /// @param[in] scale Quantization scales data buffer.
    /// @param[out] packed_rhs Packed RHS data buffer.
    void pack_rhs(
        size_t n, size_t k, const void* rhs, size_t rhs_row_stride, const void* bias, const void* scale,
        void* packed_rhs) const {
        KAI_UNUSED(scale);

        if (fn_pack_rhs != nullptr) {
            fn_pack_rhs(
                1, n, k, fn_get_nr(), fn_get_kr(), fn_get_sr(), rhs_row_stride, rhs, bias, nullptr, packed_rhs, 0,
                nullptr);
        } else {
            KAI_ERROR("RHS pre-processing is not supported!");
        }
    }

    /// Preprocesses the transposed RHS matrix.
    ///
    /// @param[in] n Size of the matrix in N dimension.
    /// @param[in] k Size of the matrix in K dimension.
    /// @param[in] rhs RHS data buffer.
    /// @param[in] rhs_row_stride RHS row stride.
    /// @param[in] bias Bias data buffer.
    /// @param[in] scale Quantization scales data buffer.
    /// @param[out] packed_rhs Packed RHS data buffer.
    void pack_rhs_nxk(
        size_t n, size_t k, const void* rhs, size_t rhs_row_stride, const void* bias, const void* scale,
        void* packed_rhs) const {
        KAI_UNUSED(scale);

        if (fn_pack_rhs_nxk != nullptr) {
            fn_pack_rhs_nxk(
                1, n, k, fn_get_nr(), fn_get_kr(), fn_get_sr(), rhs_row_stride, rhs, bias, nullptr, packed_rhs, 0,
                nullptr);
        } else {
            KAI_ERROR("RHS pre-processing is not supported!");
        }
    }

    [[nodiscard]] bool has_main_kernel() const {
        return fn_matmul_f16_f16_f16p != nullptr ||  //
            fn_matmul_f16_f16p_f16p != nullptr ||    //
            fn_matmul_f32_f32p_f32p != nullptr ||    //
            fn_matmul_f32_f32_f32p != nullptr ||     //
            fn_matmul_f32_bf16p_bf16p != nullptr ||  //
            fn_matmul_f16_bf16p_bf16p != nullptr;
    }

    void main_kernel(
        size_t m, size_t n, size_t k, const void* lhs, const void* rhs, const void* bias, void* dst, size_t lhs_stride,
        size_t rhs_stride, size_t dst_stride, float clamp_min, float clamp_max) const {
        KAI_UNUSED(bias);
        KAI_UNUSED(rhs_stride);

        if (fn_matmul_f16_f16_f16p) {
            fn_matmul_f16_f16_f16p(
                m, n, k, lhs, lhs_stride, rhs, dst, dst_stride, sizeof(uint16_t), clamp_min, clamp_max);
        } else if (fn_matmul_f32_f32_f32p) {
            fn_matmul_f32_f32_f32p(m, n, k, lhs, lhs_stride, rhs, dst, dst_stride, sizeof(float), clamp_min, clamp_max);
        } else if (fn_matmul_f16_f16p_f16p) {
            fn_matmul_f16_f16p_f16p(m, n, k, lhs, rhs, dst, dst_stride, sizeof(Float16), clamp_min, clamp_max);
        } else if (fn_matmul_f32_f32p_f32p) {
            fn_matmul_f32_f32p_f32p(m, n, k, lhs, rhs, dst, dst_stride, sizeof(float), clamp_min, clamp_max);
        } else if (fn_matmul_f32_bf16p_bf16p) {
            fn_matmul_f32_bf16p_bf16p(
                m, n, k, reinterpret_cast<const uint16_t*>(lhs), rhs, reinterpret_cast<float*>(dst), dst_stride,
                sizeof(float), clamp_min, clamp_max);
        } else if (fn_matmul_f16_bf16p_bf16p) {
            fn_matmul_f16_bf16p_bf16p(m, n, k, lhs, rhs, dst, dst_stride, sizeof(uint16_t), clamp_min, clamp_max);
        } else {
            KAI_ERROR("Main kernel is not available!");
        }
    }
};

// NOLINTEND(misc-non-private-member-variables-in-classes)

/// Describes bias handling
enum class BiasMode {
    INTERNAL,  // Zero bias internally generated in kernel
    PROVIDED,  // Bias provided by kernel caller
};

/// Matrix multiplication test information.
using MatMulTestParams = std::tuple<MatMulMethod, MatMulShape, MatrixPortion, BiasMode>;
using MatMulTestPortionedParams = std::tuple<size_t, MatMulShape, MatrixPortion>;
using MatMulTestPortionedParamsWithBias = std::tuple<size_t, MatMulShape, MatrixPortion, bool>;
using MatMulTestPortionedParamsWithBias_WithBL = std::tuple<size_t, MatMulShape, size_t, MatrixPortion, bool>;

/// Prints the test information.
void PrintTo(const MatMulTestParams& param, std::ostream* os);
void PrintTo(const MatMulShape& shape, std::ostream* os);
void PrintTo(const MatrixPortion& portion, std::ostream* os);
void PrintTo(const BiasMode& bias_mode, std::ostream* os);

/// Generate test information.
std::string test_description(
    const std::string_view& name, const MatMulShape& shape, const MatrixPortion& portion, bool bias);

}  // namespace kai::test

template <>
struct std::hash<kai::test::MatMulShape> {
    size_t operator()(const kai::test::MatMulShape& ms) const {
        return kai::test::MatMulShape::Hash{}(ms);
    }
};
