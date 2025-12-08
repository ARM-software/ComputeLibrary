//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::test {

/// Gets the sve vector length.
template <size_t esize>
size_t get_sve_vector_length();

/// Gets the sve vector length.
template <typename T>
size_t get_sve_vector_length() {
    return get_sve_vector_length<sizeof(T)>();
}

}  // namespace kai::test
