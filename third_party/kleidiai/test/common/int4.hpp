//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <tuple>

#include "test/common/buffer.hpp"

namespace kai::test {

/// 4-bit unsigned integer.
class UInt4 {
public:
    /// Creates a new 4-bit unsigned integer value.
    ///
    /// @param[in] value Value.
    constexpr explicit UInt4(uint8_t value) : _value(value) {
    }

    /// Assignment operator.
    UInt4& operator=(uint8_t value);

    /// Assignment operator.
    UInt4& operator=(int value);

    /// Conversion operator.
    operator int32_t() const;

    /// Conversion operator.
    operator float() const;

    /// Addition operator.
    [[nodiscard]] UInt4 operator+(UInt4 rhs) const;

    /// Subtraction operator.
    [[nodiscard]] UInt4 operator-(UInt4 rhs) const;

    /// Multiplication operator.
    [[nodiscard]] UInt4 operator*(UInt4 rhs) const;

    /// Division operator.
    [[nodiscard]] UInt4 operator/(UInt4 rhs) const;

    /// Packs two 4-bit unsigned integer values into one byte.
    ///
    /// @param[in] low Low nibble.
    /// @param[in] high High nibble.
    ///
    /// @return The packed byte.
    [[nodiscard]] static uint8_t pack_u8(UInt4 low, UInt4 high);

    /// Unpacks one byte to two 4-bit unsigned integer values.
    ///
    /// @param[in] value 8-bit packed value.
    ///
    /// @return The low and high nibbles.
    [[nodiscard]] static std::tuple<UInt4, UInt4> unpack_u8(uint8_t value);

private:
    uint8_t _value;
};

/// 4-bit signed integer.
class Int4 {
public:
    /// Creates a new 4-bit signed integer value.
    ///
    /// @param[in] value Value.
    constexpr explicit Int4(int8_t value) : _value(value) {
    }

    /// Assignment operator.
    Int4& operator=(int8_t value);

    /// Assignment operator.
    Int4& operator=(int value);

    /// Conversion operator.
    operator int32_t() const;

    /// Conversion operator.
    operator float() const;

    /// Addition operator.
    [[nodiscard]] Int4 operator+(Int4 rhs) const;

    /// Subtraction operator.
    [[nodiscard]] Int4 operator-(Int4 rhs) const;

    /// Multiplication operator.
    [[nodiscard]] Int4 operator*(Int4 rhs) const;

    /// Division operator.
    [[nodiscard]] Int4 operator/(Int4 rhs) const;

    /// Packs two 4-bit signed integer values into one byte.
    ///
    /// @param[in] low Low nibble.
    /// @param[in] high High nibble.
    ///
    /// @return The packed byte.
    [[nodiscard]] static uint8_t pack_u8(Int4 low, Int4 high);

    /// Unpacks one byte to two 4-bit signed integer values.
    ///
    /// @param[in] value 8-bit packed value.
    ///
    /// @return The low and high nibbles.
    [[nodiscard]] static std::tuple<Int4, Int4> unpack_u8(uint8_t value);

private:
    int8_t _value;
};

/// Reverses the two 4-bit unsigned integer values in the byte in the buffer.
///
/// @param[in] src The data buffer.
///
/// @return The buffer with packed byte, where the high and low nibbles reversed.
Buffer convert_s0s1_s1s0(const Buffer& src);

}  // namespace kai::test
