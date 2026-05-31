//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"

namespace kai::test {

/// Multidimensional array with elements being stored in row-major order.
///
/// For example:
///   A data buffer with the shape of (2, 3), i.e. 2 rows and 3 columns:
///     a00 a01 a02
///     a10 a11 a12
///   It is stored in the memory as follows:
///     a00 a01 a02 a10 a11 a12
class PlainFormat final : public Format {
public:
    /// Creates a new plain data format.
    ///
    /// @param[in] dtype The element data type.
    explicit PlainFormat(DataType dtype) : m_dtype(dtype) {
    }

    [[nodiscard]] DataType dtype() const override {
        return m_dtype;
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
    DataType m_dtype;
};

}  // namespace kai::test
