//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstdint>

namespace kai::test {

/// Rounding mode.
enum class RoundMode : uint8_t {
    CURRENT,   ///< Using the current rounding mode from fegetround.
    TIE_AWAY,  ///< Rounding to the nearest with halfway rounded away from zero.
};

/// Rounds the value using the specified rounding mode.
template <typename T, RoundMode MODE>
[[nodiscard]] T round(T value) {
    if constexpr (MODE == RoundMode::CURRENT) {
        return nearbyint(value);
    } else {
        static_assert(MODE == RoundMode::TIE_AWAY);
        return std::round(value);
    }
}

}  // namespace kai::test
