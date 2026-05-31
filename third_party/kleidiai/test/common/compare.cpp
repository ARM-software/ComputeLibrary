//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/compare.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/logging.hpp"
#include "test/common/memory.hpp"
#include "test/common/printer.hpp"
#include "test/common/rect.hpp"
#include "test/common/round.hpp"

namespace kai::test {

namespace {

/// Calculates the absolute and relative errors.
///
/// @param[in] imp Value under test.
/// @param[in] ref Reference value.
///
/// @return The absolute error and relative error.
template <typename T>
std::tuple<float, float> calculate_error(T imp, T ref) {
    const auto imp_f = static_cast<float>(imp);
    const auto ref_f = static_cast<float>(ref);

    const auto abs_error = std::abs(imp_f - ref_f);
    const auto rel_error = ref_f != 0 ? abs_error / std::abs(ref_f) : 0.0F;

    return {abs_error, rel_error};
}

/// Compares matrices with per-row quantization.
template <typename Data>
bool compare_raw(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler) {
    const size_t block_height = format.actual_block_height(full_height);
    const size_t block_width = format.actual_block_width(full_width);
    const size_t subblock_height = format.actual_subblock_height(full_height);
    const size_t subblock_width = format.actual_subblock_width(full_width);

    size_t idx = 0;

    bool block_heading_written = false;
    bool subblock_heading_written = false;
    bool row_heading_written = false;
    std::ostringstream sstream;
    for (size_t y_block = 0; y_block < full_height; y_block += block_height) {
        for (size_t x_block = 0; x_block < full_width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y_element = 0; y_element < subblock_height; ++y_element) {
                        for (size_t x_element = 0; x_element < subblock_width; ++x_element) {
                            const auto y = y_block + y_subblock + y_element;
                            const auto x = x_block + x_subblock + x_element;

                            const auto in_roi = rect.contains(y, x);
                            const auto imp_value = read_array<Data>(imp_data, idx);
                            const auto ref_value = in_roi ? read_array<Data>(ref_data, idx) : static_cast<Data>(0);

                            const auto [abs_err, rel_err] = calculate_error(imp_value, ref_value);

                            if (abs_err != 0 || rel_err != 0) {
                                if (!in_roi) {
                                    handler.mark_as_failed();
                                }

                                const auto notifying = !in_roi || handler.handle_data(abs_err, rel_err);

                                if (notifying) {
                                    if (!block_heading_written) {
                                        sstream << "block @ (" << y_block << ", " << x_block << "):\n";
                                        block_heading_written = true;
                                    }
                                    if (!subblock_heading_written) {
                                        sstream << "  sub-block @ (" << y_subblock << ", " << x_subblock << "):\n";
                                        subblock_heading_written = true;
                                    }
                                    if (!row_heading_written) {
                                        sstream << "    row=" << y_element << ": ";
                                        row_heading_written = true;
                                    }
                                    sstream << x_element << ", ";
                                }
                            }

                            ++idx;
                        }
                        if (row_heading_written) {
                            sstream << "\n";
                        }
                        row_heading_written = false;
                    }
                    subblock_heading_written = false;
                }
            }
            block_heading_written = false;
        }
    }

    const bool success = handler.success(full_height * full_width);
    if (!success) {
        KAI_LOGE("mismatches:\n", sstream.str());
    }
    return success;
}

/// Compares matrices with per-row bias or per-row quantization.
template <typename Data, typename Scale, typename Offset>
bool compare_per_row(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler) {
    static constexpr auto has_scale = !std::is_null_pointer_v<Scale>;

    const auto block_height = format.actual_block_height(full_height);
    const auto block_width = format.actual_block_width(full_width);
    const auto subblock_height = format.actual_subblock_height(full_height);
    const auto subblock_width = format.actual_subblock_width(full_width);

    KAI_ASSUME_ALWAYS(format.scheduler_block_height(full_height) == block_height);
    KAI_ASSUME_ALWAYS(format.scheduler_block_width(full_width) == full_width);
    KAI_ASSUME_ALWAYS(rect.start_col() == 0);
    KAI_ASSUME_ALWAYS(rect.width() == full_width);

    const size_t row_block_zero_points_bytes = block_height * sizeof(Offset);
    const size_t row_block_scales_bytes = has_scale ? block_height * sizeof(Scale) : 0;
    const size_t row_block_data_bytes = block_height * round_up_multiple(full_width, block_width) * sizeof(Data);

    const auto* imp_ptr = reinterpret_cast<const uint8_t*>(imp_data);
    const auto* ref_ptr = reinterpret_cast<const uint8_t*>(ref_data);

    for (size_t y_block = 0; y_block < full_height; y_block += block_height) {
        const auto in_roi = y_block >= rect.start_row() && y_block < rect.end_row();

        // Checks the zero points.
        for (size_t i = 0; i < block_height; ++i) {
            const auto imp_zero_point = reinterpret_cast<const Offset*>(imp_ptr)[i];
            const Offset ref_zero_point = in_roi ? reinterpret_cast<const Offset*>(ref_ptr)[i] : static_cast<Offset>(0);
            const auto [abs_err, rel_err] = calculate_error(imp_zero_point, ref_zero_point);

            if (abs_err != 0 || rel_err != 0) {
                handler.mark_as_failed();

                const auto raw_row = y_block + i;
                KAI_LOGE(
                    "Mismatched zero point ", raw_row, ": actual = ", imp_zero_point, ", expected: ", ref_zero_point);
            }
        }

        imp_ptr += row_block_zero_points_bytes;
        ref_ptr += row_block_zero_points_bytes;

        // Checks the data.
        for (size_t x_block = 0; x_block < full_width; x_block += block_width) {
            for (size_t y_subblock = 0; y_subblock < block_height; y_subblock += subblock_height) {
                for (size_t x_subblock = 0; x_subblock < block_width; x_subblock += subblock_width) {
                    for (size_t y = 0; y < subblock_height; ++y) {
                        for (size_t x = 0; x < subblock_width; ++x) {
                            const auto offset = (y_subblock + y) * full_width + x_block + x_subblock + x;
                            const auto imp_value = read_array<Data>(imp_ptr, offset);
                            const Data ref_value = in_roi ? read_array<Data>(ref_ptr, offset) : static_cast<Data>(0);
                            const auto [abs_err, rel_err] = calculate_error(imp_value, ref_value);

                            if (abs_err != 0 || rel_err != 0) {
                                if (!in_roi) {
                                    handler.mark_as_failed();
                                }

                                const auto notifying = !in_roi || handler.handle_data(abs_err, rel_err);

                                if (notifying) {
                                    const auto raw_index = y_block * block_height * block_width + offset;
                                    KAI_LOGE(
                                        "Mismatched data ", raw_index, ": actual = ", imp_value,
                                        ", expected: ", ref_value);
                                }
                            }
                        }
                    }
                }
            }
        }

        imp_ptr += row_block_data_bytes;
        ref_ptr += row_block_data_bytes;

        // Checks the scales (if exists).
        if constexpr (has_scale) {
            for (size_t i = 0; i < block_height; ++i) {
                const auto imp_scale = reinterpret_cast<const Scale*>(imp_ptr)[i];
                const Scale ref_scale = in_roi ? reinterpret_cast<const Scale*>(ref_ptr)[i] : 0;
                const auto [abs_err, rel_err] = calculate_error(imp_scale, ref_scale);

                if (abs_err != 0 || rel_err != 0) {
                    handler.mark_as_failed();

                    const auto raw_row = y_block + i;
                    KAI_LOGE(
                        "Mismatched quantization scale ", raw_row, ": actual = ", imp_scale, ", expected: ", ref_scale);
                }
            }

            imp_ptr += row_block_scales_bytes;
            ref_ptr += row_block_scales_bytes;
        }
    }

    return handler.success(rect.height() * full_width);
}

}  // namespace

bool compare(
    const void* imp_data, const void* ref_data, const DataFormat& format, size_t full_height, size_t full_width,
    const Rect& rect, MismatchHandler& handler) {
    const auto data_type = format.data_type();
    const auto scale_dt = format.scale_data_type();
    const auto offset_dt = format.zero_point_data_type();

    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            switch (data_type) {
                case DataType::FP32:
                    return compare_raw<float>(imp_data, ref_data, format, full_height, full_width, rect, handler);

                case DataType::FP16:
                    return compare_raw<Float16>(imp_data, ref_data, format, full_height, full_width, rect, handler);

                case DataType::BF16:
                    return compare_raw<BFloat16<>>(imp_data, ref_data, format, full_height, full_width, rect, handler);

                default:
                    break;
            }

            break;

        case DataFormat::PackFormat::BIAS_PER_ROW:
            if (data_type == DataType::FP16 && offset_dt == DataType::FP16) {
                return compare_per_row<Float16, std::nullptr_t, Float16>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            } else if (data_type == DataType::FP32 && offset_dt == DataType::FP32) {
                return compare_per_row<float, std::nullptr_t, float>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            } else if (data_type == DataType::BF16 && offset_dt == DataType::FP32) {
                return compare_per_row<BFloat16<>, std::nullptr_t, float>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            }

            break;

        case DataFormat::PackFormat::QUANTIZE_PER_ROW:
            if (data_type_is_quantized_int8(data_type) && scale_dt == DataType::FP32 && offset_dt == DataType::I32) {
                return compare_per_row<int8_t, float, int32_t>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            } else if (
                data_type_is_quantized_int4(data_type) && scale_dt == DataType::FP32 && offset_dt == DataType::I32) {
                return compare_per_row<Int4, float, int32_t>(
                    imp_data, ref_data, format, full_height, full_width, rect, handler);
            }

            break;

        default:
            break;
    }

    KAI_ERROR("Unsupported format!");
}

// =====================================================================================================================

DefaultMismatchHandler::DefaultMismatchHandler(
    float abs_error_threshold, float rel_error_threshold, size_t abs_mismatched_threshold,
    float rel_mismatched_threshold) :
    _abs_error_threshold(abs_error_threshold),
    _rel_error_threshold(rel_error_threshold),
    _abs_mismatched_threshold(abs_mismatched_threshold),
    _rel_mismatched_threshold(rel_mismatched_threshold),
    _num_mismatches(0),
    _failed(false) {
}

DefaultMismatchHandler::DefaultMismatchHandler(const DefaultMismatchHandler& rhs) :
    _abs_error_threshold(rhs._abs_error_threshold),
    _rel_error_threshold(rhs._rel_error_threshold),
    _abs_mismatched_threshold(rhs._abs_mismatched_threshold),
    _rel_mismatched_threshold(rhs._rel_mismatched_threshold),
    _num_mismatches(0),
    _failed(false) {
    // Cannot copy mismatch handler that is already in use.
    KAI_ASSUME_ALWAYS(rhs._num_mismatches == 0);
    KAI_ASSUME_ALWAYS(!rhs._failed);
}

DefaultMismatchHandler& DefaultMismatchHandler::operator=(const DefaultMismatchHandler& rhs) {
    if (this != &rhs) {
        // Cannot copy mismatch handler that is already in use.
        KAI_ASSUME_ALWAYS(rhs._num_mismatches == 0);
        KAI_ASSUME_ALWAYS(!rhs._failed);

        _abs_error_threshold = rhs._abs_error_threshold;
        _rel_error_threshold = rhs._rel_error_threshold;
        _abs_mismatched_threshold = rhs._abs_mismatched_threshold;
        _rel_mismatched_threshold = rhs._rel_mismatched_threshold;
    }

    return *this;
}

bool DefaultMismatchHandler::handle_data(float absolute_error, float relative_error) {
    const auto mismatched = absolute_error > _abs_error_threshold && relative_error > _rel_error_threshold;

    if (mismatched) {
        ++_num_mismatches;
    }

    return mismatched;
}

void DefaultMismatchHandler::mark_as_failed() {
    _failed = true;
}

bool DefaultMismatchHandler::success(size_t num_checks) const {
    if (_failed) {
        return false;
    }

    const auto mismatched_rate = static_cast<float>(_num_mismatches) / static_cast<float>(num_checks);
    return _num_mismatches <= _abs_mismatched_threshold || mismatched_rate <= _rel_mismatched_threshold;
}

}  // namespace kai::test
