//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP

#include <cstddef>

#include "kai/kai_common.h"

namespace kai::benchmark {

/// Abstraction for the unspecialized Matrix Multiplication microkernel interface
struct MatMulBaseInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the unspecialized Matrix Multiplication microkernel interface with a strided LHS matrix
struct MatMulStridedLhsInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        size_t lhs_stride,                             //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication microkernel interface with a floating point destination buffer
struct MatMulFloatInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        float* dst,                                    //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication micro-kernel with static quantization
struct MatMulStaticQuantInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        const kai_matmul_requantize32_params* params);
};

/// Abstraction for the Matrix Multiplication micro-kernel with dynamic blockwise quantization and generic destination
/// buffer
struct MatMulBlockwiseDynamicQuantGenericDstInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl,       //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication micro-kernel with dynamic blockwise quantization and float destination
/// buffer
struct MatMulBlockwiseDynamicQuantInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl,       //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        float* dst,                                    //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP
