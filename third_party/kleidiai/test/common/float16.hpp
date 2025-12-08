//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iosfwd>

#include "test/common/type_traits.hpp"

extern "C" {

/// Converts single-precision floating-point to half-precision floating-point.
///
/// @params[in] value The single-precision floating-point value.
///
/// @return The half-precision floating-point value reinterpreted as 16-bit unsigned integer.
uint16_t kai_test_float16_from_float(float value);

/// Converts half-precision floating-point to single-precision floating-point.
///
/// @params[in] The half-precision floating-point value reinterpreted as 16-bit unsigned integer.
///
/// @return The single-precision floating-point value.
float kai_test_float_from_float16(uint16_t value);

/// Adds two half-precision floating-point numbers.
///
/// All half-precision floating-point values are reinterpreted as 16-bit unsigned integer.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return The result of the addition.
uint16_t kai_test_float16_add(uint16_t lhs, uint16_t rhs);

/// Subtracts two half-precision floating-point numbers.
///
/// All half-precision floating-point values are reinterpreted as 16-bit unsigned integer.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return The result of the subtraction.
uint16_t kai_test_float16_sub(uint16_t lhs, uint16_t rhs);

/// Multiplies two half-precision floating-point numbers.
///
/// All half-precision floating-point values are reinterpreted as 16-bit unsigned integer.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return The result of the multiplication.
uint16_t kai_test_float16_mul(uint16_t lhs, uint16_t rhs);

/// Divides two half-precision floating-point numbers.
///
/// All half-precision floating-point values are reinterpreted as 16-bit unsigned integer.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return The result of the division.
uint16_t kai_test_float16_div(uint16_t lhs, uint16_t rhs);

/// Determines whether the first operand is less than the second operand.
///
/// Both operands are half-precision floating-point reinterpreted as 16-bit unsigned integers.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return `true` if the first operand is less than the second operand, otherwise `true`.
bool kai_test_float16_lt(uint16_t lhs, uint16_t rhs);

/// Determines whether the first operand is greater than the second operand.
///
/// Both operands are half-precision floating-point reinterpreted as 16-bit unsigned integers.
///
/// @param[in] lhs The LHS operand.
/// @param[in] rhs The RHS operand.
///
/// @return `true` if the first operand is greater than the second operand, otherwise `true`.
bool kai_test_float16_gt(uint16_t lhs, uint16_t rhs);

}  // extern "C"

namespace kai::test {

/// Half-precision floating-point.
class Float16 {
public:
    /// Constructor.
    constexpr Float16() = default;

    /// Creates a new half-precision floating-point value from the specified
    /// single-precision floating-point value.
    ///
    /// @param[in] value The single-precision floating-point value.
    explicit Float16(float value) : m_data(kai_test_float16_from_float(value)) {
    }

    /// Creates a new half-precision floating-point value from the raw data.
    ///
    /// @param[in] data The binary representation of the floating-point value.
    ///
    /// @return The half-precision floating-point value.
    static constexpr Float16 from_binary(uint16_t data) {
        Float16 value{};
        value.m_data = data;
        return value;
    }

    /// Assigns to the specified numeric value.
    template <typename T, std::enable_if_t<is_arithmetic<T>, bool> = true>
    Float16& operator=(T value) {
        const auto value_f32 = static_cast<float>(value);
        m_data = kai_test_float16_from_float(value_f32);
        return *this;
    }

    /// Converts to single-precision floating-point.
    explicit operator float() const {
        return kai_test_float_from_float16(m_data);
    }

    /// Addition and assignment operator.
    Float16& operator+=(Float16 rhs) {
        m_data = kai_test_float16_add(m_data, rhs.m_data);
        return *this;
    }

    /// Subtraction and assignment operator.
    Float16& operator-=(Float16 rhs) {
        m_data = kai_test_float16_sub(m_data, rhs.m_data);
        return *this;
    }

    /// Multiplication and assignment operator.
    Float16& operator*=(Float16 rhs) {
        m_data = kai_test_float16_mul(m_data, rhs.m_data);
        return *this;
    }

    /// Division and assignment operator.
    Float16& operator/=(Float16 rhs) {
        m_data = kai_test_float16_div(m_data, rhs.m_data);
        return *this;
    }

private:
    /// Addition operator.
    [[nodiscard]] friend Float16 operator+(Float16 lhs, Float16 rhs) {
        Float16 value;
        value.m_data = kai_test_float16_add(lhs.m_data, rhs.m_data);
        return value;
    }

    /// Subtraction operator.
    [[nodiscard]] friend Float16 operator-(Float16 lhs, Float16 rhs) {
        Float16 value;
        value.m_data = kai_test_float16_sub(lhs.m_data, rhs.m_data);
        return value;
    }

    /// Multiplication operator.
    [[nodiscard]] friend Float16 operator*(Float16 lhs, Float16 rhs) {
        Float16 value;
        value.m_data = kai_test_float16_mul(lhs.m_data, rhs.m_data);
        return value;
    }

    /// Division operator.
    [[nodiscard]] friend Float16 operator/(Float16 lhs, Float16 rhs) {
        Float16 value;
        value.m_data = kai_test_float16_div(lhs.m_data, rhs.m_data);
        return value;
    }

    /// Equality operator.
    [[nodiscard]] friend bool operator==(Float16 lhs, Float16 rhs) {
        return lhs.m_data == rhs.m_data;
    }

    /// Unequality operator.
    [[nodiscard]] friend bool operator!=(Float16 lhs, Float16 rhs) {
        return lhs.m_data != rhs.m_data;
    }

    /// Less operator.
    [[nodiscard]] friend bool operator<(Float16 lhs, Float16 rhs) {
        return kai_test_float16_lt(lhs.m_data, rhs.m_data);
    }

    /// Greater operator.
    [[nodiscard]] friend bool operator>(Float16 lhs, Float16 rhs) {
        return kai_test_float16_gt(lhs.m_data, rhs.m_data);
    }

    /// Less-or-equal operator.
    [[nodiscard]] friend bool operator<=(Float16 lhs, Float16 rhs) {
        return !(lhs > rhs);
    }

    /// Greater-or-equal operator.
    [[nodiscard]] friend bool operator>=(Float16 lhs, Float16 rhs) {
        return !(lhs < rhs);
    }

    uint16_t m_data{0};
};

/// Writes the value to the output stream.
///
/// @param[in] os Output stream to be written to.
/// @param[in] value Value to be written.
///
/// @return The output stream.
std::ostream& operator<<(std::ostream& os, Float16 value);

}  // namespace kai::test
