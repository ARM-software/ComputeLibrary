//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "test/nextgen/common/test_registry.hpp"

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    kai::test::TestRegistry::init();

    return RUN_ALL_TESTS();
}
