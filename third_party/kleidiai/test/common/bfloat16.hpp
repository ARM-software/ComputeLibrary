//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <type_traits>

#include "test/common/cpu_info.hpp"
#include "test/common/type_traits.hpp"

extern "C" {

/// Converts single-precision floating-point to half-precision brain floating-point.
///
/// @params[in] value The single-precision floating-point value.
///
/// @return The half-precision brain floating-point value reinterpreted as 16-bit unsigned integer.
uint16_t kai_test_float_to_bfloat16_bfcvt(float value);

}  // extern "C"
namespace kai::test {

/// Half-precision brain floating-point.
template <bool hardware_support>
class BFloat16 {
public:
    /// Constructor.
    BFloat16() = default;

    using p_F32BF16convert = uint16_t (*)(float);

    /// Creates a new object from the specified numeric value.
    explicit BFloat16(float value) : m_data(f32_bf16_convertfn(value)) {
    }

    /// Creates a new half-precision brain floating-point value from the raw data.
    ///
    /// @param[in] data The binary representation of the floating-point value.
    ///
    /// @return The half-precision brain floating-point value.
    static constexpr BFloat16 from_binary(uint16_t data) {
        BFloat16 value{};
        value.m_data = data;
        return value;
    }

    /// Assigns to the specified numeric value which will be converted to `bfloat16_t`.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    BFloat16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
        m_data = f32_bf16_convertfn(value_f32);
        return *this;
    }

    /// Converts to single-precision floating-point.
    explicit operator float() const {
        float value_f32 = 0.0F;
        uint32_t value_u32 = static_cast<uint32_t>(m_data) << 16;

        memcpy(&value_f32, &value_u32, sizeof(float));

        return value_f32;
    }

private:
    /// Equality operator.
    [[nodiscard]] friend bool operator==(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data == rhs.m_data;
    }

    /// Inequality operator.
    [[nodiscard]] friend bool operator!=(BFloat16 lhs, BFloat16 rhs) {
        return lhs.m_data != rhs.m_data;
    }

    /// Writes the value to the output stream.
    ///
    /// @param[in] os Output stream to be written to.
    /// @param[in] value Value to be written.
    ///
    /// @return The output stream.
    friend std::ostream& operator<<(std::ostream& os, BFloat16<> value);

    static uint16_t float_to_bfloat16_round_towards_zero(float value) {
        uint32_t value_u32;

        memcpy(&value_u32, &value, sizeof(value));

        return value_u32 >> 16;
    }

    inline static p_F32BF16convert f32_bf16_convertfn = (hardware_support && cpu_has_bf16())
        ? &kai_test_float_to_bfloat16_bfcvt
        : &float_to_bfloat16_round_towards_zero;

    uint16_t m_data;
};

}  // namespace kai::test
