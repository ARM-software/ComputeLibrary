//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "kai/kai_common.h"

namespace kai::benchmark {

/// Abstraction for the unspecialized Indirect Matrix Multiplication micro-kernel interface
struct ImatmulBaseInterface {
    void (*run_imatmul)(
        size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length,  //
        const void* lhs_packed,                                           //
        const void* rhs_packed,                                           //
        void* dst,                                                        //
        size_t dst_stride_row,                                            //
        float clamp_min, float clamp_max);
};

/// Abstraction for the unspecialized Indirect Matrix Multiplication micro-kernel interface with static quantization
struct ImatmulStaticQuantInterface {
    void (*run_imatmul)(
        size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length,  //
        const void* lhs_packed,                                           //
        const void* rhs_packed,                                           //
        void* dst,                                                        //
        size_t dst_stride_row,                                            //
        const kai_matmul_requantize32_params* params);
};

}  // namespace kai::benchmark
