//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"

namespace kai::test {

/// Data format.
///
/// A data format contains the description of how the data is stored in the memory,
/// including data type, data reordering rule, multi-component data packing, etc.
///
/// Data format does not own data nor has any information about the size of the data
/// as well as the underlining meaning of the data (e.g. data, scale, bias, etc.).
class Format {
public:
    Format() = default;                          ///< Default constructor.
    virtual ~Format() = default;                 ///< Destructor.
    Format(const Format&) = default;             ///< Copy constructor.
    Format& operator=(const Format&) = default;  ///< Copy assignment.
    Format(Format&&) = default;                  ///< Move constructor.
    Format& operator=(Format&&) = default;       ///< Move assignment.

    /// Gets the data type of data format.
    ///
    /// Only @ref PlainFormat supports this method.
    [[nodiscard]] virtual DataType dtype() const {
        KAI_TEST_ERROR("Not supported.");
    }

    /// Calculates the offset in bytes to locate data of this format in the memory.
    ///
    /// @param[in] shape The size of the multidimensional data.
    /// @param[in] indices The coordinate to the data element.
    ///
    /// @return The offset in bytes.
    [[nodiscard]] virtual size_t compute_offset(Span<const size_t> shape, Span<const size_t> indices) const = 0;

    /// Calculates the size in bytes of a data buffer of this format with the specified shape.
    ///
    /// @param[in] shape The size of the multidimensional data.
    ///
    /// @return The size in bytes.
    [[nodiscard]] virtual size_t compute_size(Span<const size_t> shape) const = 0;

    /// Generates random data with this format.
    ///
    /// @param[in] shape The size of the multidimensional data.
    /// @param[in, out] rng The random number generator.
    ///
    /// @return The data buffer.
    [[nodiscard]] virtual Buffer generate_random(Span<const size_t> shape, Rng& rng) const = 0;

    /// Packs the data with this format.
    ///
    /// Depending on the actual format, the list of source data buffers can be different.
    ///
    /// @param[in] buffers The list of source data buffers.
    ///
    /// @return The packed data buffer.
    [[nodiscard]] virtual Buffer pack(Span<const size_t> shape, Span<const Span<const std::byte>> buffers) const = 0;

    /// Compares a portion of two data buffers with this format.
    ///
    /// The data inside the tile of interests of the two buffers are compared.
    /// The data in the buffer under test that is outside the tile of intersts must be 0.
    ///
    /// @param[in] shape The size of the multidimensional data.
    /// @param[in] tile_coords The starting coordinate of the tile to be compared.
    /// @param[in] tile_shape The size of the tile to be compared.
    /// @param[in] imp_buffer The data buffer under test.
    /// @param[in] ref_buffer The reference data buffer.
    /// @param[in] handler The mismatch handler.
    ///
    /// @return `true` if the two data buffers are considered matched.
    [[nodiscard]] virtual bool compare(
        Span<const size_t> shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer, MismatchHandler& handler) const = 0;

    /// Prints the content of the data buffer with this format to the output stream.
    ///
    /// @param[in] os The output stream to write to.
    /// @param[in] shape The size of the multidimensional data.
    /// @param[in] data The data buffer.
    virtual void print(std::ostream& os, Span<const size_t> shape, Span<const std::byte> data) const = 0;

    /// Equal operator.
    [[nodiscard]] virtual bool operator==(const Format& other) const = 0;

    /// Not equal operator.
    [[nodiscard]] bool operator!=(const Format& other) const {
        return !(*this == other);
    }
};

}  // namespace kai::test
