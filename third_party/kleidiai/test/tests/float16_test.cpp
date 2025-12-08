//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/float16.hpp"

#include <gtest/gtest.h>

#include "test/common/cpu_info.hpp"
#include "test/common/numeric_limits.hpp"

namespace kai::test {

TEST(Float16, SimpleTest) {
    if (!cpu_has_fp16()) {
        GTEST_SKIP() << "No CPU support for FP16";
    }

    ASSERT_EQ(static_cast<float>(Float16()), 0.0F);
    ASSERT_EQ(static_cast<float>(Float16(1.25F)), 1.25F);
    ASSERT_EQ(static_cast<float>(Float16(3)), 3.0F);
    ASSERT_EQ(Float16(1.25F) + Float16(2.0F), Float16(1.25F + 2.0F));
    ASSERT_EQ(Float16(1.25F) - Float16(2.0F), Float16(1.25F - 2.0F));
    ASSERT_EQ(Float16(1.25F) * Float16(2.0F), Float16(1.25F * 2.0F));
    ASSERT_EQ(Float16(1.25F) / Float16(2.0F), Float16(1.25F / 2.0F));

    ASSERT_FALSE(Float16(1.25F) == Float16(2.0F));
    ASSERT_TRUE(Float16(1.25F) == Float16(1.25F));
    ASSERT_FALSE(Float16(2.0F) == Float16(1.25F));

    ASSERT_TRUE(Float16(1.25F) != Float16(2.0F));
    ASSERT_FALSE(Float16(1.25F) != Float16(1.25F));
    ASSERT_TRUE(Float16(2.0F) != Float16(1.25F));

    ASSERT_TRUE(Float16(1.25F) < Float16(2.0F));
    ASSERT_FALSE(Float16(1.25F) < Float16(1.25F));
    ASSERT_FALSE(Float16(2.0F) < Float16(1.25F));

    ASSERT_FALSE(Float16(1.25F) > Float16(2.0F));
    ASSERT_FALSE(Float16(1.25F) > Float16(1.25F));
    ASSERT_TRUE(Float16(2.0F) > Float16(1.25F));

    ASSERT_TRUE(Float16(1.25F) <= Float16(2.0F));
    ASSERT_TRUE(Float16(1.25F) <= Float16(1.25F));
    ASSERT_FALSE(Float16(2.0F) <= Float16(1.25F));

    ASSERT_FALSE(Float16(1.25F) >= Float16(2.0F));
    ASSERT_TRUE(Float16(1.25F) >= Float16(1.25F));
    ASSERT_TRUE(Float16(2.0F) >= Float16(1.25F));

    Float16 a(1.25F);
    Float16 b(2.0F);

    a += b;
    ASSERT_EQ(a, Float16(1.25F + 2.0F));
    a -= b;
    ASSERT_EQ(a, Float16(1.25F));
    a *= b;
    ASSERT_EQ(a, Float16(1.25F * 2.0F));
    a /= b;
    ASSERT_EQ(a, Float16(1.25F));
}

TEST(Float16, NumericLimitTest) {
    ASSERT_EQ(static_cast<float>(numeric_lowest<Float16>), -65504.0F);
    ASSERT_EQ(static_cast<float>(numeric_highest<Float16>), 65504.0F);
}

}  // namespace kai::test
