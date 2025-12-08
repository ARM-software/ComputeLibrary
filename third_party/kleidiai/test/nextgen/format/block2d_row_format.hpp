//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"

namespace kai::test {

/// 2D blocked data with optional per-row values.
///
/// Example:
///   Shape: (5, 8)
///   Block size: (2, 3)
///   Prefix per-row data 0:
///     a0 a1 a2 a3 a4
///   Prefix per-row data 1:
///     b0 b1 b2 b3 b4
///   Data:
///     v00 v01 v02 v03 v04 v05 v06 v07
///     v10 v11 v12 v13 v14 v15 v16 v17
///     v20 v21 v22 v23 v24 v25 v26 v27
///     v30 v31 v32 v33 v34 v35 v36 v37
///     v40 v41 v42 v43 v44 v45 v46 v47
///   Postfix per-row data 0:
///     c0 c1 c2 c3 c4
///   Postfix per-row data 1:
///     d0 d1 d2 d3 d4
///
///   Combined blocked data with per-row data:
///     +----+----+-------------+--------------+------------+----+----+
///     | a0 | b0 | v00 v01 v02 | v03 v04 v05 | v06 v07 ___ | c0 | d0 |
///     | a1 | b1 | v10 v11 v12 | v13 v14 v15 | v16 v17 ___ | c1 | d1 |
///     +----+----+-------------+--------------+------------+----+----+
///     | a2 | b2 | v20 v21 v22 | v23 v24 v25 | v26 v27 ___ | c2 | d2 |
///     | a3 | b3 | v30 v31 v32 | v33 v34 v35 | v36 v37 ___ | c3 | d3 |
///     +----+----+-------------+--------------+------------+----+----+
///     | a4 | b4 | v40 v41 v42 | v43 v44 v45 | v46 v47 ___ | c4 | d4 |
///     | __ | __ | ___ ___ ___ | ___ ___ ___ | ___ ___ ___ | __ | __ |
///     +----+----+-------------+--------------+------------+----+----+
///
///   Packed data stream:
///     +-------+-------+-------------------------+-------------------------+-------------------------+-------+-------+
///     | a0 a1 | b0 b1 | v00 v01 v02 v10 v11 v12 | v03 v04 v05 v13 v14 v15 | v06 v07  0  v16 v17  0  | c0 c1 | d0 d1 |
///     +-------+-------+-------------------------+-------------------------+-------------------------+-------+-------+
///     | a2 a3 | b2 b3 | v20 v21 v22 v30 v31 v32 | v23 v24 v25 v33 v34 v35 | v26 v27  0  v36 v37  0  | c2 c3 | d2 d3 |
///     +-------+-------+-------------------------+-------------------------+-------------------------+-------+-------+
///     | a4  0 | b4  0 | v40 v41 v42  0   0   0  | v43 v44 v45  0   0   0  | v46 v47  0   0   0   0  | c4  0 | d4  0 |
///     +-------+-------+-------------------------+-------------------------+-------------------------+-------+-------+
class Block2dRowFormat : public Format {
public:
    /// Creates a 2D blocked data  with optional per-row values.
    ///
    /// @param[in] block_height The block height.
    /// @param[in] block_width The block width.
    /// @param[in] width_align The input data is padded so that the width is multiple of this value
    ///                        before the data is packed. This value must be divisible by block width.
    /// @param[in] pad_right_same Right padding with the last element instead of 0.
    /// @param[in] dtype The data type.
    /// @param[in] pre_dtypes The data type of each prefix per-row component.
    /// @param[in] post_dtypes The data type of each postfix per-row component.
    Block2dRowFormat(
        size_t block_height, size_t block_width, size_t width_align, bool pad_right_same, DataType dtype,
        Span<const DataType> pre_dtypes, Span<const DataType> post_dtypes) :
        m_block_height(block_height),
        m_block_width(block_width),
        m_width_align(width_align),
        m_pad_right_same(pad_right_same),
        m_dtype(dtype),
        m_pre_dtypes(pre_dtypes.begin(), pre_dtypes.end()),
        m_post_dtypes(post_dtypes.begin(), post_dtypes.end()) {
        KAI_TEST_ASSERT(width_align % block_width == 0);
        KAI_TEST_ASSERT(block_height * block_width * data_type_size_in_bits(dtype) % 8 == 0);

        for (const DataType pre_dtype : pre_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(pre_dtype) % 8 == 0);
        }

        for (const DataType post_dtype : post_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(post_dtype) % 8 == 0);
        }
    }

    [[nodiscard]] size_t compute_offset(Span<const size_t> shape, Span<const size_t> indices) const override;
    [[nodiscard]] size_t compute_size(Span<const size_t> shape) const override;
    [[nodiscard]] Buffer generate_random(Span<const size_t> shape, Rng& rng) const override;
    [[nodiscard]] Buffer pack(Span<const size_t> shape, Span<const Span<const std::byte>> buffers) const override;
    [[nodiscard]] bool compare(
        Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer, MismatchHandler& handler) const override;
    void print(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data) const override;
    [[nodiscard]] bool operator==(const Format& other) const override;

private:
    size_t m_block_height;
    size_t m_block_width;
    size_t m_width_align;
    bool m_pad_right_same;
    DataType m_dtype;
    std::vector<DataType> m_pre_dtypes;
    std::vector<DataType> m_post_dtypes;
};

}  // namespace kai::test
