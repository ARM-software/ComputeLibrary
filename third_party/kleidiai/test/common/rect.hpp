//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <iosfwd>

namespace kai::test {

/// Rectangular region of a matrix.
///
/// This is the absolute version of @ref MatrixPortion.
class Rect {
public:
    /// Creates a new rectangular region of a matrix.
    ///
    /// @param[in] start_row Starting row index.
    /// @param[in] start_col Starting column index.
    /// @param[in] height Number of rows.
    /// @param[in] width Number of columns.
    Rect(size_t start_row, size_t start_col, size_t height, size_t width);

    /// Gets the starting row index.
    [[nodiscard]] size_t start_row() const;

    /// Gets the starting column index.
    [[nodiscard]] size_t start_col() const;

    /// Gets the number of rows.
    [[nodiscard]] size_t height() const;

    /// Gets the number of columns.
    [[nodiscard]] size_t width() const;

    /// Gets the end (exclusive) row index.
    [[nodiscard]] size_t end_row() const;

    /// Gets the end (exclusive) column index.
    [[nodiscard]] size_t end_col() const;

    /// Check if position is within rect
    [[nodiscard]] bool contains(size_t row, size_t col) const;

private:
    friend std::ostream& operator<<(std::ostream& os, const Rect& rect);

    size_t _start_row;
    size_t _start_col;
    size_t _height;
    size_t _width;
};

}  // namespace kai::test
