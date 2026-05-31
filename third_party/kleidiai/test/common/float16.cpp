//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/float16.hpp"

#include <iostream>
#include <type_traits>

namespace kai::test {

static_assert(sizeof(Float16) == 2);

static_assert(std::is_trivially_destructible_v<Float16>);
static_assert(std::is_nothrow_destructible_v<Float16>);

static_assert(std::is_trivially_copy_constructible_v<Float16>);
static_assert(std::is_trivially_copy_assignable_v<Float16>);
static_assert(std::is_trivially_move_constructible_v<Float16>);
static_assert(std::is_trivially_move_assignable_v<Float16>);

static_assert(std::is_nothrow_copy_constructible_v<Float16>);
static_assert(std::is_nothrow_copy_assignable_v<Float16>);
static_assert(std::is_nothrow_move_constructible_v<Float16>);
static_assert(std::is_nothrow_move_assignable_v<Float16>);

std::ostream& operator<<(std::ostream& os, Float16 value) {
    return os << static_cast<float>(value);
}

}  // namespace kai::test
