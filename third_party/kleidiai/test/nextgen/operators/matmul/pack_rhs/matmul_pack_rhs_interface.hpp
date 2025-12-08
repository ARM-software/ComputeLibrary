//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "kai/kai_common.h"

namespace kai::test {

/// Interface for RHS packing with per-channel quantization.
struct MatMulPackRhsQuantInterface {
    size_t (*get_n_step)(size_t nr);
    size_t (*get_rhs_offset)(size_t n_idx, size_t rhs_stride);
    size_t (*get_rhs_packed_stride)(size_t k, size_t nr, size_t kr, size_t sr);
    size_t (*get_rhs_packed_offset)(size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr);
    size_t (*get_rhs_packed_size)(size_t n, size_t k, size_t nr, size_t kr, size_t sr);
    void (*run)(
        size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
        const float* scale, void* rhs_packed, size_t extra_bytes, const kai_rhs_pack_qs4cxs1s0_param* params);
};

}  // namespace kai::test
