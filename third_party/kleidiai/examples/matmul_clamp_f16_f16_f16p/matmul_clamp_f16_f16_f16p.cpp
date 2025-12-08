//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two half precision floating-point (FP16) matrices and the accumulation of
// the result into an FP16 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using floating-point fused multiply-add to accumulator (FMLA)
// vector instructions present in the FEAT_FP16 Arm® architecture feature.
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else
#include <arm_neon.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

// Include micro-kernel variants
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

namespace {
/// Micro-kernel interface
constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
    kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

/// Reference implementation of matrix multiplication
void run_matmul_ref(
    size_t m, size_t n, size_t k, const float16_t* lhs, const float16_t* rhs, const float16_t* bias, float16_t* dst,
    float scalar_min, float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float16_t acc = bias[col_idx];

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += lhs[row_idx * k + k_idx] * rhs[col_idx + n * k_idx];
            }
            acc = std::max(acc, static_cast<float16_t>(scalar_min));
            acc = std::min(acc, static_cast<float16_t>(scalar_max));

            dst[row_idx * n + col_idx] = acc;
        }
    }
}

/// Fills the matrix with incremental values
void fill_matrix(size_t num_rows, size_t num_cols, float16_t* dst, const float16_t weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = float16_t(i * weight);
    }
}

/// Print the matrix
void print_matrix(size_t num_rows, size_t num_cols, const char* name, const float16_t* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << src[y * num_cols + x] << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

/// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float16_t tolerance, const float16_t* ref, const float16_t* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;

            std::cout << std::setprecision(5) << std::fixed << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";

            is_valid = false;
        }
    }
    return is_valid;
}
}  // namespace

int main() {
    int ret = 0;
    // 1x1 Convolution operator in NHWC format.
    const size_t nhwc_n = 2;
    const size_t nhwc_h = 2;
    const size_t nhwc_w = 4;
    const size_t nhwc_c_in = 4;    // Input channels
    const size_t nhwc_c_out = 24;  // Output channels

    // Map NHWC of operator to GEMM terminology
    const size_t M = nhwc_h * nhwc_w * nhwc_n;  // Rows of LHS and DST matrices
    const size_t N = nhwc_c_out;                // Columns of RHS and DST matrices
    const size_t K = nhwc_c_in;                 // Columns of LHS, rows of RHS matrices

    const size_t lhs_size = M * K;
    const size_t rhs_size = N * K;
    const size_t bias_size = N;
    const size_t dst_size = M * N;

    // Allocate the memory
    float16_t* lhs = new float16_t[lhs_size];
    float16_t* rhs = new float16_t[rhs_size];
    float16_t* bias = new float16_t[bias_size];

    fill_matrix(M, K, lhs, 0.1);
    fill_matrix(K, N, rhs, 0.1);
    fill_matrix(1, N, bias, 10);

#ifdef KAI_DEBUG
    print_matrix(M, K, "lhs", lhs);
    print_matrix(K, N, "rhs", rhs);
    print_matrix(1, N, "bias", bias);
#endif  // KAI_DEBUG

    //----------- REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------
    float16_t* dst_ref = new float16_t[dst_size];

    run_matmul_ref(
        M, N, K,           // Dimensions
        lhs,               // LHS buffer
        rhs,               // RHS buffer
        bias,              // Bias buffer
        dst_ref,           // DST
        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
    );
    //----------- END REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
    const size_t rhs_packed_cols = nr + K * nr;
    const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

    uint8_t* rhs_packed = new uint8_t[rhs_packed_size];

    const size_t lhs_stride = K * sizeof(float16_t);
    const size_t rhs_stride = N * sizeof(float16_t);
    const size_t dst_stride_row = N * sizeof(float16_t);
    const size_t dst_stride_col = sizeof(float16_t);

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    // The RHS and Bias buffers can be freed after packing, however we reuse them for the reference test below

#ifdef KAI_DEBUG
    print_matrix(rhs_packed_rows, rhs_packed_cols, "rhs_packed", rhs_packed);
#endif  // KAI_DEBUG

    float16_t* dst = new float16_t[dst_size];

    // Framework scheduling params

    // Example alternative values to try. ukernel.get_m_step() * 2 or M;
    const size_t m_step = ukernel.get_m_step();  // Scheduling along M

    // Example alternative values to try.  n_step = N;
    const size_t n_step = ukernel.get_n_step();  // Scheduling along N

    for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step) {
        for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
            // Support functions return offset in bytes
            const uint8_t* lhs_ptr =
                (const uint8_t*)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
            const uint8_t* rhs_ptr = (const uint8_t*)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
            uint8_t* dst_ptr = (uint8_t*)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));
#ifdef KAI_DEBUG
            printf("Processing a %zux%zu ouptut block starting at (%zu, %zu)\n", m_step, n_step, i_m_step, i_n_step);
#endif
            const size_t actual_m = std::min(M - i_m_step, m_step);
            const size_t actual_n = std::min(N - i_n_step, n_step);

            ukernel.run_matmul(
                actual_m, actual_n, K,  // Dimensions
                lhs_ptr,                // LHS
                lhs_stride,             // LHS stride
                rhs_ptr,                // RHS packed
                dst_ptr,                // DST
                dst_stride_row,         // DST stride (row)
                dst_stride_col,         // DST stride (col)
                -FLT_MAX, FLT_MAX       // Min and max for the clamp operation
            );
        }
    }

#ifdef KAI_DEBUG
    print_matrix(M, N, "dst", dst);
#endif  // KAI_DEBUG

    const bool is_valid = is_output_correct(M, N, 0.0001, dst_ref, dst);

    std::cout << "TEST[matmul_clamp_f16_f16_f16p]\n";
    std::cout << "- ukernel: matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla\n";
    if (is_valid) {
        std::cout << "- Status: PASSED\n";
    } else {
        std::cout << "- Status: FAILED\n";
        ret = 1;
    }

    //----------- END MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------

    delete[] lhs;
    delete[] rhs;
    delete[] bias;
    delete[] rhs_packed;
    delete[] dst;
    delete[] dst_ref;

    return ret;
}
#endif  // Architectural features check.
