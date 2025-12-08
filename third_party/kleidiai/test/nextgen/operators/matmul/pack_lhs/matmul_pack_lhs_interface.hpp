//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Interface for LHS packing kernel with dynamic quantization.
struct MatMulPackLhsDqInterface {
    size_t (*get_m_step)(size_t mr);
    size_t (*get_lhs_offset)(size_t m_idx, size_t lhs_stride);
    size_t (*get_lhs_packed_offset)(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
    size_t (*get_lhs_packed_size)(size_t m, size_t k, size_t mr, size_t kr, size_t sr);
    void (*run)(
        size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs, size_t lhs_stride,
        void* lhs_packed);
};

}  // namespace kai::test
