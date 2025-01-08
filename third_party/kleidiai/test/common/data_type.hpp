//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::test {

/// Data type.
enum class DataType : uint16_t {
    // Encoding:
    //
    //    15                                                           0
    //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    //   | i | s | q | a |     RES0      |             bits              |
    //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    //
    //   (RES0: reserved, filled with 0s)
    //
    // Fields:
    //
    //   * i: integer (1) or floating-point (0).
    //   * s: signed (1) or unsigned (0).
    //   * q:
    //     - Integer (i): quantized (1) or non-quantized (0).
    //     - Floating-point (!i): brain (1) or binary (0).
    //   * a:
    //     - Quantized (i && q): asymmetric (1) or symmetric (0).
    //     - Otherwise: RES0.
    //   * bits: size in bits.

    UNKNOWN = 0,  ///< No data.

    FP32 = 0b0'1'0'0'0000'00100000,  ///< Single-precision floating-point.
    FP16 = 0b0'1'0'0'0000'00010000,  ///< Half-precision floating-point.

    BF16 = 0b0'1'1'0'0000'00010000,  ///< Half-precision brain floating-point.

    I32 = 0b1'1'0'0'0000'00100000,  ///< 32-bit signed integer.

    QAI8 = 0b1'1'1'1'0000'00001000,  ///< 8-bit signed asymmetric quantized.

    QSU4 = 0b1'0'1'0'0000'00000100,  ///< 4-bit unsigned symmetric quantized.
    QSI4 = 0b1'1'1'0'0000'00000100,  ///< 4-bit signed symmetric quantized.
};

/// Gets the size in bits of the specified data type.
///
/// @param[in] dt The data type.
///
/// @return The size in bits.
[[nodiscard]] size_t data_type_size_in_bits(DataType dt);

/// Gets a value indicating whether the data type is integral.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is integral.
[[nodiscard]] bool data_type_is_integral(DataType dt);

/// Gets a value indicating whether the data type is floating-point.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is floating-point.
[[nodiscard]] bool data_type_is_float(DataType dt);

/// Gets a value indicating whether the data type is binary floating-point.
///
/// Binary floating point are `half`, `float`, `double`.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is binary floating-point.
[[nodiscard]] bool data_type_is_float_fp(DataType dt);

/// Gets a value indicating whether the data type is brain floating-point.
///
/// Binary floating point are `bfloat16`.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is brain floating-point.
[[nodiscard]] bool data_type_is_float_bf(DataType dt);

/// Gets a value indicating whether the data type is signed.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is signed.
[[nodiscard]] bool data_type_is_signed(DataType dt);

/// Gets a value indicating whether the data type is quantized.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is quantized.
[[nodiscard]] bool data_type_is_quantized(DataType dt);

/// Gets a value indicating whether the data type is asymmetric quantized.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is asymmetric quantized.
[[nodiscard]] bool data_type_is_quantized_asymm(DataType dt);

}  // namespace kai::test
