//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/int4.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

UInt4& UInt4::operator=(uint8_t value) {
    KAI_ASSUME_ALWAYS(value >= 0 && value < 16);
    _value = value;
    return *this;
}

UInt4& UInt4::operator=(int value) {
    KAI_ASSUME_ALWAYS(value >= 0 && value < 16);
    _value = static_cast<uint8_t>(value);
    return *this;
}

UInt4::operator int32_t() const {
    return _value;
}

UInt4::operator float() const {
    return _value;
}

UInt4 UInt4::operator+(UInt4 rhs) const {
    return UInt4(_value + rhs._value);
}

UInt4 UInt4::operator-(UInt4 rhs) const {
    return UInt4(_value - rhs._value);
}

UInt4 UInt4::operator*(UInt4 rhs) const {
    return UInt4(_value * rhs._value);
}

UInt4 UInt4::operator/(UInt4 rhs) const {
    return UInt4(_value / rhs._value);
}

uint8_t UInt4::pack_u8(UInt4 low, UInt4 high) {
    return (low._value & 0x0F) | (high._value << 4);
}

std::tuple<UInt4, UInt4> UInt4::unpack_u8(uint8_t value) {
    const uint8_t low = value & 0x0F;
    const uint8_t high = value >> 4;

    return {UInt4(low), UInt4(high)};
}

// =====================================================================================================================

Int4& Int4::operator=(int8_t value) {
    KAI_ASSUME_ALWAYS(value >= -8 && value < 8);
    _value = value;
    return *this;
}

Int4& Int4::operator=(int value) {
    KAI_ASSUME_ALWAYS(value >= -8 && value < 8);
    _value = static_cast<int8_t>(value);
    return *this;
}

Int4::operator int32_t() const {
    return _value;
}

Int4::operator float() const {
    return _value;
}

Int4 Int4::operator+(Int4 rhs) const {
    return Int4(static_cast<int8_t>(_value + rhs._value));
}

Int4 Int4::operator-(Int4 rhs) const {
    return Int4(static_cast<int8_t>(_value - rhs._value));
}

Int4 Int4::operator*(Int4 rhs) const {
    return Int4(static_cast<int8_t>(_value * rhs._value));
}

Int4 Int4::operator/(Int4 rhs) const {
    return Int4(static_cast<int8_t>(_value / rhs._value));
}

uint8_t Int4::pack_u8(Int4 low, Int4 high) {
    const uint8_t lo = low._value & 0x0F;
    const uint8_t hi = high._value & 0x0F;
    return (lo << 0) | (hi << 4);
}

std::tuple<Int4, Int4> Int4::unpack_u8(uint8_t value) {
    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8_t low = static_cast<int8_t>(value << 4) >> 4;
    const int8_t high = static_cast<int8_t>(value) >> 4;
    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

    return {Int4(low), Int4(high)};
}

// =====================================================================================================================

Buffer convert_s0s1_s1s0(const Buffer& src) {
    const auto length = src.size();
    Buffer dst(length);

    for (size_t i = 0; i < length; ++i) {
        uint8_t val = read_array<uint8_t>(src.data(), i);
        const auto [low, high] = UInt4::unpack_u8(val);
        auto rev_val = UInt4::pack_u8(high, low);
        write_array<uint8_t>(dst.data(), i, rev_val);
    }
    return dst;
}

}  // namespace kai::test
