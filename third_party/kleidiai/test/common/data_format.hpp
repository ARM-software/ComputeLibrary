//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

#include "test/common/data_type.hpp"

namespace kai::test {

/// Data format.
class DataFormat {
public:
    /// Packing format.
    enum class PackFormat : uint32_t {
        NONE,              ///< No quantization information is included.
        BIAS_PER_ROW,      ///< Per-row bias.
        QUANTIZE_PER_ROW,  ///< Per-row quantization.
    };

    /// Creates a new data format.
    ///
    /// @param[in] data_type Data type of data value.
    /// @param[in] block_height Block height.
    /// @param[in] block_width Block width.
    /// @param[in] pack_format Packing format.
    /// @param[in] zero_point_dt Data type of zero point value.
    /// @param[in] scale_dt Data type of scale value.
    /// @param[in] subblock_height Sub-block height.
    /// @param[in] subblock_width Sub-block width.
    DataFormat(
        DataType data_type = DataType::UNKNOWN, size_t block_height = 0, size_t block_width = 0,
        PackFormat pack_format = PackFormat::NONE, DataType zero_point_dt = DataType::UNKNOWN,
        DataType scale_dt = DataType::UNKNOWN, size_t subblock_height = 0, size_t subblock_width = 0) noexcept;

    /// Equality operator.
    [[nodiscard]] bool operator==(const DataFormat& rhs) const;

    /// Unequality operator.
    [[nodiscard]] bool operator!=(const DataFormat& rhs) const;

    /// Gets the packing format.
    [[nodiscard]] PackFormat pack_format() const;

    /// Gets the data type of data value.
    [[nodiscard]] DataType data_type() const;

    /// Gets the data type of scale value.
    [[nodiscard]] DataType scale_data_type() const;

    /// Gets the data type of zero point value.
    [[nodiscard]] DataType zero_point_data_type() const;

    /// Gets a value indicating whether this format has no blocking or packing information.
    [[nodiscard]] bool is_raw() const;

    /// Gets the block height.
    [[nodiscard]] size_t block_height() const;

    /// Gets the block width.
    [[nodiscard]] size_t block_width() const;

    /// Gets the sub-block height.
    [[nodiscard]] size_t subblock_height() const;

    /// Gets the sub-block width.
    [[nodiscard]] size_t subblock_width() const;

    /// Gets the block height given the full height of the matrix.
    ///
    /// @param[in] full_height Height of the full matrix.
    ///
    /// @return The block height.
    [[nodiscard]] size_t actual_block_height(size_t full_height) const;

    /// Gets the block width given the full width of the matrix.
    ///
    /// @param[in] full_width Width of the full matrix.
    ///
    /// @return The block width.
    [[nodiscard]] size_t actual_block_width(size_t full_width) const;

    /// Gets the sub-block height given the full height of the matrix.
    ///
    /// @param[in] full_height Height of the full matrix.
    ///
    /// @return The sub-block height.
    [[nodiscard]] size_t actual_subblock_height(size_t full_height) const;

    /// Gets the sub-block width given the full width of the matrix.
    ///
    /// @param[in] full_width Width of the full matrix.
    ///
    /// @return The sub-block width.
    [[nodiscard]] size_t actual_subblock_width(size_t full_width) const;

    /// Gets the scheduling block height.
    ///
    /// @param[in] full_height Height of the full matrix.
    ///
    /// @return The block height for scheduling purpose.
    [[nodiscard]] size_t scheduler_block_height(size_t full_height) const;

    /// Gets the scheduling block width.
    ///
    /// @param[in] full_width Width of the full matrix.
    ///
    /// @return The block width for scheduling purpose.
    [[nodiscard]] size_t scheduler_block_width(size_t full_width) const;

    /// Gets the row stride in bytes given the data is stored continuously without any gap in the memory.
    ///
    /// In case of per-row bias or quantization, the row stride is the number of bytes from one row group
    /// to the next. One row group consists of `block_height` rows.
    ///
    /// @param[in] width Width of the full matrix.
    ///
    /// @return The default row stride in bytes of the matrix.
    [[nodiscard]] uintptr_t default_row_stride(size_t width) const;

    /// Gets the offsets in bytes in the data buffer given the data is stored continuously
    /// without any gap in the memory.
    ///
    /// @param[in] row Row coordinate.
    /// @param[in] col Colum coordinate.
    /// @param[in] width Width of the full matrix.
    ///
    /// @return The default offset in bytes.
    [[nodiscard]] uintptr_t default_offset_in_bytes(size_t row, size_t col, size_t width) const;

    /// Gets the size in bytes of the matrix given the data is stored continuously without any gap in the memory.
    ///
    /// @param[in] height Height of the full matrix.
    /// @param[in] width Width of the full matrix.
    ///
    /// @return The size in bytes of the matrix.
    [[nodiscard]] size_t default_size_in_bytes(size_t height, size_t width) const;

    /// Hash functor
    struct Hash {
        size_t operator()(const DataFormat& format) const;
    };

private:
    DataType _data_type;
    PackFormat _pack_format;
    DataType _scale_dt;
    DataType _zero_point_dt;
    size_t _block_height;
    size_t _block_width;
    size_t _subblock_height;
    size_t _subblock_width;
};

}  // namespace kai::test

template <>
struct std::hash<kai::test::DataFormat> {
    size_t operator()(const kai::test::DataFormat& df) const {
        return kai::test::DataFormat::Hash{}(df);
    }
};

template <>
struct std::hash<kai::test::DataFormat::PackFormat> {
    size_t operator()(const kai::test::DataFormat::PackFormat& pf) const {
        using PF = std::underlying_type_t<kai::test::DataFormat::PackFormat>;
        return std::hash<PF>{}(static_cast<PF>(pf));
    }
};
