//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

namespace kai::test {

/// Fused multiplies and adds.
///
/// @param[in] mul_a The LHS multiplicand.
/// @param[in] mul_b The RHS multiplicand.
/// @param[in] addend The addend.
///
/// @return The fused multiplication and addition result.
template <typename T>
[[nodiscard]] T fused_mul_add(T mul_a, T mul_b, T addend);

template <>
inline float fused_mul_add(float mul_a, float mul_b, float addend) {
    return std::fma(mul_a, mul_b, addend);
}

}  // namespace kai::test
