//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_format.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>

#include "kai/kai_common.h"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"

namespace kai::test {

DataFormat::DataFormat(
    DataType data_type, size_t block_height, size_t block_width, PackFormat pack_format, DataType zero_point_dt,
    DataType scale_dt, size_t subblock_height, size_t subblock_width) noexcept :
    _data_type(data_type),
    _pack_format(pack_format),
    _scale_dt(scale_dt),
    _zero_point_dt(zero_point_dt),
    _block_height(block_height),
    _block_width(block_width),
    _subblock_height(subblock_height),
    _subblock_width(subblock_width) {
}

bool DataFormat::operator==(const DataFormat& rhs) const {
    return _data_type == rhs._data_type && _pack_format == rhs._pack_format && _scale_dt == rhs._scale_dt &&
        _zero_point_dt == rhs._zero_point_dt && _block_height == rhs._block_height && _block_width == rhs._block_width;
}

bool DataFormat::operator!=(const DataFormat& rhs) const {
    return !(*this == rhs);
}

DataType DataFormat::data_type() const {
    return _data_type;
}

DataFormat::PackFormat DataFormat::pack_format() const {
    return _pack_format;
}

DataType DataFormat::scale_data_type() const {
    return _scale_dt;
}

DataType DataFormat::zero_point_data_type() const {
    return _zero_point_dt;
}

bool DataFormat::is_raw() const {
    return _pack_format == PackFormat::NONE &&  //
        _block_height == 0 && _block_width == 0 && _subblock_height == 0 && _subblock_width == 0;
}

size_t DataFormat::block_height() const {
    return _block_height;
}

size_t DataFormat::block_width() const {
    return _block_width;
}

size_t DataFormat::subblock_height() const {
    return _subblock_height;
}

size_t DataFormat::subblock_width() const {
    return _subblock_width;
}

size_t DataFormat::actual_block_height(size_t full_height) const {
    return _block_height > 0 ? _block_height
                             : round_up_multiple(full_height, _subblock_height > 0 ? _subblock_height : 1);
}

size_t DataFormat::actual_block_width(size_t full_width) const {
    return _block_width > 0 ? _block_width : round_up_multiple(full_width, _subblock_width > 0 ? _subblock_width : 1);
}

size_t DataFormat::actual_subblock_height(size_t full_height) const {
    return _subblock_height > 0 ? _subblock_height : actual_block_height(full_height);
}

size_t DataFormat::actual_subblock_width(size_t full_width) const {
    return _subblock_width > 0 ? _subblock_width : actual_block_width(full_width);
}

size_t DataFormat::scheduler_block_height([[maybe_unused]] size_t full_height) const {
    const auto padded_block_height = round_up_multiple(_block_height, _subblock_height > 0 ? _subblock_height : 1);

    switch (_pack_format) {
        case PackFormat::NONE:
            return _block_height > 0 ? padded_block_height : 1;

        case PackFormat::BIAS_PER_ROW:
        case PackFormat::QUANTIZE_PER_ROW:
            KAI_ASSUME_ALWAYS(_block_height > 0);
            return padded_block_height;

        default:
            KAI_ERROR("Unsupported packing format!");
    }
}

size_t DataFormat::scheduler_block_width(size_t full_width) const {
    const auto padded_block_width = round_up_multiple(_block_width, _subblock_width > 0 ? _subblock_width : 1);

    switch (_pack_format) {
        case PackFormat::NONE:
            return _block_width > 0 ? padded_block_width : 1;

        case PackFormat::BIAS_PER_ROW:
        case PackFormat::QUANTIZE_PER_ROW:
            return full_width;

        default:
            KAI_ERROR("Unsupported packing format!");
    }
}

uintptr_t DataFormat::default_row_stride(size_t width) const {
    const auto padded_width = round_up_multiple(width, actual_block_width(width));

    switch (_pack_format) {
        case PackFormat::NONE:
            return (_block_height > 0 ? _block_height : 1) * padded_width * data_type_size_in_bits(_data_type) / 8;

        case PackFormat::BIAS_PER_ROW:
            KAI_ASSUME_ALWAYS(_block_height > 0);
            return _block_height * data_type_size_in_bits(_zero_point_dt) / 8 +  //
                _block_height * padded_width * data_type_size_in_bits(_data_type) / 8;

        case PackFormat::QUANTIZE_PER_ROW:
            KAI_ASSUME_ALWAYS(_block_height > 0);
            return _block_height * data_type_size_in_bits(_zero_point_dt) / 8 +          //
                _block_height * padded_width * data_type_size_in_bits(_data_type) / 8 +  //
                _block_height * data_type_size_in_bits(_scale_dt) / 8;

        default:
            KAI_ERROR("Unsupported packing format!");
    }
}

uintptr_t DataFormat::default_offset_in_bytes(size_t row, size_t col, size_t width) const {
    const auto row_stride = default_row_stride(width);
    const auto block_width = scheduler_block_width(width);

    KAI_ASSERT_ALWAYS(col % block_width == 0);

    switch (_pack_format) {
        case PackFormat::NONE:
            return row * row_stride / (_block_height > 0 ? _block_height : 1) +
                col * data_type_size_in_bits(_data_type) / 8;

        case PackFormat::BIAS_PER_ROW:
        case PackFormat::QUANTIZE_PER_ROW:
            KAI_ASSUME_ALWAYS(row % _block_height == 0);
            KAI_ASSUME_ALWAYS(col == 0);
            return (row / _block_height) * row_stride;

        default:
            KAI_ERROR("Unsupported packing format!");
    }
}

size_t DataFormat::default_size_in_bytes(size_t height, size_t width) const {
    const auto num_rows = _block_height > 0 ? (height + _block_height - 1) / _block_height : height;
    const auto block_stride = default_row_stride(width);
    return num_rows * block_stride;
}

size_t DataFormat::Hash::operator()(const DataFormat& format) const {
    return                                                                 //
        (std::hash<DataType>{}(format._data_type) << 0) ^                  //
        (std::hash<DataFormat::PackFormat>{}(format._pack_format) << 1) ^  //
        (std::hash<DataType>{}(format._scale_dt) << 2) ^                   //
        (std::hash<DataType>{}(format._zero_point_dt) << 3) ^              //
        (std::hash<size_t>{}(format._block_height) << 4) ^                 //
        (std::hash<size_t>{}(format._block_width) << 5) ^                  //
        (std::hash<size_t>{}(format._subblock_height) << 6) ^              //
        (std::hash<size_t>{}(format._subblock_width) << 7);                //
}

}  // namespace kai::test
