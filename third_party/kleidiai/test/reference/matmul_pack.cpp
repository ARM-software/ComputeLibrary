//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/matmul_pack.hpp"

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/round.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/reduce.hpp"
#include "test/reference/reorder.hpp"

namespace kai::test {

template <typename Data, typename Scale, typename ZeroPoint>
Buffer matmul_pack_rhs_nxk_static_quantized(
    const void* data, const void* scales, Scale lhs_scale, Scale dst_scale, const void* biases,
    ZeroPoint lhs_zero_point, size_t n, size_t k, size_t block_height, size_t block_width) {
    // The RHS data matrix is reordered according to the blocking parameters.
    const auto reordered_data = reorder_block<Data>(data, n, k, block_height, block_width);

    // The effective per-channel scale:
    //   final_scales[n_index] = lhs_scale * rhs_scales[n_index] / dst_scale.
    const auto scale_multiplier = lhs_scale / dst_scale;
    auto combined_scales = mul<Scale>(scales, 1, n, &scale_multiplier, 1, 1);
    combined_scales = pad_matrix<Scale>(
        combined_scales.data(), 1, n, 0, 0, round_up_multiple(n, block_height) - n, 0, 0);  // Pads with 0s.

    // The effective per-channel biases:
    //   final_biases[n_index] = biases[n_index] - lhs_zero_point * sum(data[n_index, :]).
    const auto row_sum_reduced = reduce_add_x<Data, ZeroPoint>(data, n, k);
    // Reduced across width earlier, so lhs width is now 1
    const auto row_sum_times_lhs_zp = mul<ZeroPoint>(row_sum_reduced.data(), n, 1, &lhs_zero_point, 1, 1);
    auto combined_biases = sub<ZeroPoint>(biases, 1, n, row_sum_times_lhs_zp.data(), 1, n);
    combined_biases = pad_matrix<ZeroPoint>(
        combined_biases.data(), 1, n, 0, 0, round_up_multiple(n, block_height) - n, 0, 0);  // Pads with 0s.

    // Packs the effective biases followed by the data block followed by the effective scales for the block.
    auto packed_rhs = pack_zero_points_data_scales_per_block<ZeroPoint, Data, Scale>(
        combined_biases.data(), reordered_data.data(), combined_scales.data(), round_up_division(n, block_height),
        block_height, block_height * round_up_multiple(k, block_width), block_height);

    return packed_rhs;
}

template Buffer matmul_pack_rhs_nxk_static_quantized<int8_t>(
    const void* data, const void* scales, float lhs_scale, float dst_scale, const void* biases, int32_t lhs_zero_point,
    size_t n, size_t k, size_t block_height, size_t block_width);

}  // namespace kai::test
