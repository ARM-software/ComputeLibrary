//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/bfloat16.hpp"

#include <iostream>

namespace kai::test {

std::ostream& operator<<(std::ostream& os, BFloat16 value) {
    return os << static_cast<float>(value);
}

}  // namespace kai::test
