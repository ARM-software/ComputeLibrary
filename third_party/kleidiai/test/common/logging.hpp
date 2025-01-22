//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

#include "test/common/int4.hpp"

#define KAI_LOGE(...) kai::test::detail::log("ERROR", __VA_ARGS__)

namespace kai::test::detail {

/// Prints the specified value to standard error.
///
/// @tparam T Data type.
///
/// @param[in] value Value to be printed out.
template <typename T>
void write_log_content(T&& value) {
    using TT = std::decay_t<decltype(value)>;

    if constexpr (std::is_same_v<TT, uint8_t>) {
        std::cerr << static_cast<uint32_t>(value);
    } else if constexpr (std::is_same_v<TT, int8_t>) {
        std::cerr << static_cast<int32_t>(value);
    } else if constexpr (std::is_same_v<TT, UInt4>) {
        std::cerr << static_cast<int32_t>(value);
    } else if constexpr (std::is_same_v<TT, Int4>) {
        std::cerr << static_cast<int32_t>(value);
    } else {
        std::cerr << value;
    }
}

/// Prints the specified values to standard error.
///
/// @tparam T Data type of the first value.
/// @tparam Ts Data types of the subsequent values.
///
/// @param[in] value First value to be printed out.
/// @param[in] others Subsequent values to be printed out.
template <typename T, typename... Ts>
void write_log_content(T&& value, Ts&&... others) {
    write_log_content(std::forward<T>(value));
    write_log_content(std::forward<Ts>(others)...);
}

/// Prints the log to standard error.
///
/// @tparam Ts Data types of values to be printed out.
///
/// @param[in] level Severity level.
/// @param[in] args Values to be printed out.
template <typename... Ts>
void log(std::string_view level, Ts&&... args) {
    std::cerr << level << " | ";
    write_log_content(std::forward<Ts>(args)...);
    std::cerr << "\n";
}

}  // namespace kai::test::detail
