//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "kai/kai_common.h"
#include "test/common/assert.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"

namespace kai::test {

/// The size in bits of type `T`.
template <typename T>
inline constexpr size_t size_in_bits = sizeof(T) * 8;

/// The size in bits of type `T`.
template <>
inline constexpr size_t size_in_bits<UInt4> = 4;

/// The size in bits of type `T`.
template <>
inline constexpr size_t size_in_bits<Int4> = 4;

/// Reads the array at the specified index.
///
/// @param[in] array Data buffer.
/// @param[in] index Array index.
///
/// @return The array value at the specified index.
template <typename T>
T read_array(const void* array, size_t index) {
    if constexpr (std::is_same_v<T, UInt4>) {
        const auto [lo, hi] = UInt4::unpack_u8(reinterpret_cast<const uint8_t*>(array)[index / 2]);
        return index % 2 == 0 ? lo : hi;
    } else if constexpr (std::is_same_v<T, Int4>) {
        const auto [lo, hi] = Int4::unpack_u8(reinterpret_cast<const uint8_t*>(array)[index / 2]);
        return index % 2 == 0 ? lo : hi;
    } else if constexpr (std::is_same_v<T, BFloat16<false>>) {
        uint16_t raw_value = reinterpret_cast<const uint16_t*>(array)[index];
        return BFloat16<false>(kai_cast_f32_bf16(raw_value));
    } else if constexpr (std::is_same_v<T, BFloat16<true>>) {
        uint16_t raw_value = reinterpret_cast<const uint16_t*>(array)[index];
        return BFloat16<true>(kai_cast_f32_bf16(raw_value));
    } else {
        return reinterpret_cast<const T*>(array)[index];
    }
}

/// Reads the array at the specified index.
///
/// @param[in] array Data buffer.
/// @param[in] index Array index.
///
/// @return The array value at the specified index.
template <typename T>
T read_array(Span<const std::byte> array, size_t index) {
    const size_t min_size = round_up_division((index + 1) * size_in_bits<T>, 8);
    KAI_TEST_ASSERT_MSG(array.size() >= min_size, "The read access is out-of-bound!");
    return read_array<T>(array.data(), index);
}

/// Reads the 2D array at the specified coordinates.
///
/// @param[in] data The data buffer.
/// @param[in] width The array width.
/// @param[in] row The row index.
/// @param[in] col The column index.
///
/// @return The array value at the specified coordinates.
template <typename T>
T read_2d(Span<const std::byte> data, size_t width, size_t row, size_t col) {
    const size_t stride = round_up_division(width * size_in_bits<T>, 8);
    return read_array<T>(data.subspan(row * stride, stride), col);
}

/// Reads the array at the specified index
///
/// @param[in] type Array element data type
/// @param[in] array Data buffer.
/// @param[in] index Array index.
///
/// @return Value at specified index
double read_array(DataType type, const void* array, size_t index);

/// Writes the specified value to the array.
///
/// @param[in] array Data buffer.
/// @param[in] index Array index.
/// @param[in] value Value to be stored.
template <typename T>
void write_array(void* array, size_t index, T value) {
    if constexpr (std::is_same_v<T, UInt4>) {
        auto* arr_value = reinterpret_cast<uint8_t*>(array) + index / 2;
        const auto [lo, hi] = UInt4::unpack_u8(*arr_value);

        if (index % 2 == 0) {
            *arr_value = UInt4::pack_u8(value, hi);
        } else {
            *arr_value = UInt4::pack_u8(lo, value);
        }
    } else if constexpr (std::is_same_v<T, Int4>) {
        auto* arr_value = reinterpret_cast<uint8_t*>(array) + index / 2;
        const auto [lo, hi] = Int4::unpack_u8(*arr_value);

        if (index % 2 == 0) {
            *arr_value = Int4::pack_u8(value, hi);
        } else {
            *arr_value = Int4::pack_u8(lo, value);
        }
    } else {
        reinterpret_cast<T*>(array)[index] = value;
    }
}

/// Writes the specified value to the array.
///
/// @param[in] array Data buffer.
/// @param[in] index Array index.
/// @param[in] value Value to be stored.
template <typename T>
void write_array(Span<std::byte> array, size_t index, T value) {
    const size_t min_size = round_up_division((index + 1) * size_in_bits<T>, 8);
    KAI_TEST_ASSERT_MSG(array.size() >= min_size, "The write access is out-of-bound!");
    write_array<T>(array.data(), index, value);
}

/// Writes the specified value to the 2D array at the specified coordinates.
///
/// @param[out] data The data buffer.
/// @param[in] width The array width.
/// @param[in] row The row index.
/// @param[in] col The column index.
/// @param[in] value The value to be stored.
template <typename T>
void write_2d(Span<std::byte> data, size_t width, size_t row, size_t col, T value) {
    const size_t stride = round_up_division(width * size_in_bits<T>, 8);
    write_array<T>(data.subspan(row * stride, stride), col, value);
}

/// Writes the specified value to the array.
///
/// @param[in] type Array element type.
/// @param[in] array Data buffer.
/// @param[in] index Array index.
/// @param[in] value Value to be stored.
void write_array(DataType type, void* array, size_t index, double value);

}  // namespace kai::test
