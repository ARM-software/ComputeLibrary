//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/data_type.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// Packs the data into 2D blocks.
///
/// Example:
///   Shape: (5, 8)
///   Block size: (2, 3)
///   Data:
///     v00 v01 v02 v03 v04 v05 v06 v07
///     v10 v11 v12 v13 v14 v15 v16 v17
///     v20 v21 v22 v23 v24 v25 v26 v27
///     v30 v31 v32 v33 v34 v35 v36 v37
///     v40 v41 v42 v43 v44 v45 v46 v47
///
///   Blocked data:
///     +-------------+--------------+------------+
///     | v00 v01 v02 | v03 v04 v05 | v06 v07 ___ |
///     | v10 v11 v12 | v13 v14 v15 | v16 v17 ___ |
///     +-------------+--------------+------------+
///     | v20 v21 v22 | v23 v24 v25 | v26 v27 ___ |
///     | v30 v31 v32 | v33 v34 v35 | v36 v37 ___ |
///     +-------------+--------------+------------+
///     | v40 v41 v42 | v43 v44 v45 | v46 v47 ___ |
///     | ___ ___ ___ | ___ ___ ___ | ___ ___ ___ |
///     +-------------+--------------+------------+
///
///   Packed data stream:
///     +-------------------------+-------------------------+-------------------------+
///     | v00 v01 v02 v10 v11 v12 | v03 v04 v05 v13 v14 v15 | v06 v07  0  v16 v17  0  |
///     +-------------------------+-------------------------+-------------------------+
///     | v20 v21 v22 v30 v31 v32 | v23 v24 v25 v33 v34 v35 | v26 v27  0  v36 v37  0  |
///     +-------------------------+-------------------------+-------------------------+
///     | v40 v41 v42  0   0   0  | v43 v44 v45  0   0   0  | v46 v47  0   0   0   0  |
///     +-------------------------+-------------------------+-------------------------+
///
/// @param[in] block_height The block height.
/// @param[in] block_width The block width.
/// @param[in] width_align The input data is padded so that the width is multiple of this value
///                        before the data is packed. This value must be divisible by block width.
/// @param[in] pad_right_same Right padding with the last element instead of 0.
/// @param[in] height The data height.
/// @param[in] width The data width.
/// @param[out] packed_data The packed data buffer.
/// @param[in] data The input data buffer.
///
/// @return The size of packed data.
using PackBlock2dFn = size_t (*)(
    size_t block_height, size_t block_width, size_t width_align, bool pad_right_same, size_t height, size_t width,
    Span<std::byte> packed_data, Span<const std::byte> data);

/// Gets the 2D block packing function for the specified data type.
///
/// @param[in] dtype The data type.
///
/// @return The function pointer.
[[nodiscard]] PackBlock2dFn make_pack_block2d(DataType dtype);

}  // namespace kai::test
