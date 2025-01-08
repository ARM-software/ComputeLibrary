//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kai::test {

/// Reorders the input matrix block by block.
///
/// Example:
///
/// The input matrix: 5x7.
///
/// ```
/// +-----------------------------+
/// | a00 a01 a02 a03 a04 a05 a06 |
/// | a10 a11 a12 a13 a14 a15 a16 |
/// | a20 a21 a22 a23 a24 a25 a26 |
/// | a30 a31 a32 a33 a34 a35 a36 |
/// | a40 a41 a42 a43 a44 a45 a46 |
/// +-----------------------------+
/// ```
///
/// The matrix is divided into blocks of 2x3.
/// At the right and bottom edges, the partial blocks are padded with 0s.
///
/// ```
//  +-------------+-------------+-------------+
/// | a00 a01 a02 | a03 a04 a05 | a06  0   0  |
/// | a10 a11 a12 | a13 a14 a15 | a16  0   0  |
/// +-------------+-------------+-------------+
/// | a20 a21 a22 | a23 a24 a25 | a26  0   0  |
/// | a30 a31 a32 | a33 a34 a35 | a36  0   0  |
/// +-------------+-------------+-------------+
/// | a40 a41 a42 | a43 a44 a45 | a46  0   0  |
/// |  0   0   0  |  0   0   0  |  0   0   0  |
/// +-------------+-------------+-------------+
/// ```
///
/// Each block is then flatten to get the final reordered matrix:
///
/// ```
/// +-------------------------+-------------------------+-------------------------+
/// | a00 a01 a02 a10 a11 a12 | a03 a04 a05 a13 a14 a15 | a06  0   0  a16  0   0  |
/// +-------------------------+-------------------------+-------------------------+
/// | a20 a21 a22 a30 a31 a32 | a23 a24 a25 a33 a34 a35 | a26  0   0  a36  0   0  |
/// +-------------------------+-------------------------+-------------------------+
/// | a40 a41 a42  0   0   0  | a43 a44 a45  0   0   0  | a46  0   0   0   0   0  |
/// +-------------------------+-------------------------+-------------------------+
///
/// @tparam T The data type.
///
/// @param[in] src The input data.
/// @param[in] height The number of rows of the input matrix.
/// @param[in] width The number of columns of the input matrix.
/// @param[in] block_height The number of rows of a block.
/// @param[in] block_width The number of columns of a block.
///
/// @param[in] The reordered matrix.
/// ```
template <typename T>
std::vector<uint8_t> reorder_block(
    const void* src, size_t height, size_t width, size_t block_height, size_t block_width);

}  // namespace kai::test
