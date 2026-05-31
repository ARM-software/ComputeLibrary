//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Packing arguments.
struct MatMulPackArgs {
    size_t mr;
    size_t nr;
    size_t kr;
    size_t sr;
    size_t bl;
};

}  // namespace kai::test
