//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Interface for matrix multiplication kernel with dynamic quantization.
struct MatMulDqInterface {
    size_t (*get_m_step)();
    size_t (*get_n_step)();
    size_t (*get_mr)();
    size_t (*get_nr)();
    size_t (*get_kr)();
    size_t (*get_sr)();
    size_t (*get_lhs_packed_offset)(size_t m_idx, size_t k);
    size_t (*get_rhs_packed_offset)(size_t n_idx, size_t k);
    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t dst_stride);
    size_t (*get_dst_size)(size_t m, size_t n);
    void (*run)(
        size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, float* dst, size_t dst_stride_row,
        size_t dst_stride_col, float scalar_min, float scalar_max);
};

}  // namespace kai::test
