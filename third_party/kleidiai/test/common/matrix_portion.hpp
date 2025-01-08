//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/rect.hpp"

namespace kai::test {

/// Portion of a matrix.
///
/// This class is used to define the sub-matrix under test.
///
/// This is the relative version of @ref Rect.
class MatrixPortion {
public:
    /// Creates a new matrix portion.
    ///
    /// @param[in] start_row Starting row as the ratio to the height of the matrix.
    /// @param[in] start_col Starting column as the ratio to the width of the matrix.
    /// @param[in] height Portion height as the ratio to the height of the matrix.
    /// @param[in] width Portion width as the ratio to the width of the matrix.
    MatrixPortion(float start_row, float start_col, float height, float width);

    /// Gets the starting row as the ratio to the height of the matrix.
    [[nodiscard]] float start_row() const;

    /// Gets the starting column as the ratio to the width of the matrix.
    [[nodiscard]] float start_col() const;

    /// Gets the portion height as the ratio to the height of the matrix.
    [[nodiscard]] float height() const;

    /// Gets the portion width as the ratio to the width of the matrix.
    [[nodiscard]] float width() const;

    /// Computes the starting coordinate and the shape of the sub-matrix.
    ///
    /// Requirements:
    ///
    ///   * The starting coordinate of the sub-matrix shall be aligned with the scheduling block boundary.
    ///   * If it is not the scheduling block at the right and/or bottom edge of the full matrix, the height and width
    ///     of the sub-matrix shall be rounded up to multiple of the scheduling block height and width.
    ///   * If it is the scheduling block at the right and/or bottom edge of the full matrix, the height and width
    ///     of the sub-matrix shall be the rounded up to the edge of the matrix.
    ///
    /// @param[in] full_height Matrix height.
    /// @param[in] full_width Matrix width.
    /// @param[in] scheduler_block_height Block height for scheduling purpose.
    /// @param[in] scheduler_block_width Block width for scheduling purpose.
    ///
    /// @return The rectangular region of the matrix.
    [[nodiscard]] Rect compute_portion(
        size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width) const;

private:
    float _start_row;
    float _start_col;
    float _height;
    float _width;
};

}  // namespace kai::test
