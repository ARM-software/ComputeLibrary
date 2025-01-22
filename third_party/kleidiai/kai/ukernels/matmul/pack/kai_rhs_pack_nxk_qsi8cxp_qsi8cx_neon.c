//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    const size_t k_internal = kai_k_roundedup(k);

    return nr * (k_internal + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
    size_t num_groups,   //
    size_t n,            //
    size_t k,            //
    size_t nr,           //
    size_t kr,           //
    size_t sr,           //
    const int8_t* rhs,   //
    const float* bias,   //
    const float* scale,  //
    void* rhs_packed,    //
    size_t extra_bytes, const struct kai_rhs_pack_qsi8cx_params* params) {
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT(sr == 1);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);

    const int32_t lhs_zero_point = params->lhs_zero_point;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(k, nr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k);
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_num_bytes_per_row = nr * k_internal;
    const size_t rhs_stride = k;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;

        int32_t* sums = (int32_t*)(dst_row + nr * k_internal);

        // Initialize to zero the RHS reduction sums
        memset(sums, 0, nr * sizeof(int32_t));

        for (size_t dst_offset = 0; dst_offset < dst_num_bytes_per_row; dst_offset += kr) {
            const size_t block_idx = dst_offset / kr;
            const size_t nr_idx = block_idx % nr;
            const size_t super_block_idx = block_idx / nr;

            const size_t k0_idx = super_block_idx * kr;
            const size_t n0_idx = dst_row_idx * nr + nr_idx;

            // Clamp the index to avoid out-of-bound reads
            const size_t n0_valid_idx = KAI_MIN(n0_idx, n - 1);

            const size_t src_offset = n0_valid_idx * rhs_stride;

            int32_t partial_sum = 0;

            // Get the partial reduction sum
            for (size_t i = 0; i < kr; i++) {
                const size_t k0_valid_idx = k0_idx + i;
                int8_t v = 0;
                if (k0_valid_idx < k) {
                    v = rhs[src_offset + k0_valid_idx];
                }
                ((int8_t*)dst_row)[i] = v;
                partial_sum += v;
            }

            sums[nr_idx] += partial_sum * lhs_zero_point;

            dst_row += kr;
        }

        // Adjust the reduction sums
        for (size_t i = 0; i < nr; ++i) {
            dst_row += sizeof(int32_t);
        }

        // Adjust the scales
        for (size_t i = 0; i < nr; ++i) {
            // Clamp the row index to avoid out-of-bound reads
            const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
            *((float*)(dst_row)) = scale[src_row_idx];
            dst_row += sizeof(float);
        }

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx * nr + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }
    }
}
