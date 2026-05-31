//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/bfloat16.hpp"

#include <iostream>
#include <type_traits>

namespace kai::test {

static_assert(sizeof(BFloat16<>) == 2);

static_assert(std::is_trivially_destructible_v<BFloat16<>>);
static_assert(std::is_nothrow_destructible_v<BFloat16<>>);

static_assert(std::is_trivially_copy_constructible_v<BFloat16<>>);
static_assert(std::is_trivially_copy_assignable_v<BFloat16<>>);
static_assert(std::is_trivially_move_constructible_v<BFloat16<>>);
static_assert(std::is_trivially_move_assignable_v<BFloat16<>>);

static_assert(std::is_nothrow_copy_constructible_v<BFloat16<>>);
static_assert(std::is_nothrow_copy_assignable_v<BFloat16<>>);
static_assert(std::is_nothrow_move_constructible_v<BFloat16<>>);
static_assert(std::is_nothrow_move_assignable_v<BFloat16<>>);

std::ostream& operator<<(std::ostream& os, BFloat16<> value) {
    return os << static_cast<float>(value);
}

}  // namespace kai::test
