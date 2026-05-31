//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/buffer.hpp"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/signal.h>

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>

namespace kai::test {

namespace {
constexpr size_t g_num_runs = 100;
}  // namespace

TEST(BufferTest, NonePolicy) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Store the current buffer policy
    std::string buffer_policy;
    if (const char* buffer_policy_env = getenv(Buffer::buffer_policy_env_name)) {
        buffer_policy = std::string(buffer_policy_env);
    }

    // Overwrite the buffer policy for purpose of the test
    ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, "NONE", 1 /* overwrite */), 0);

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
    }

    if (buffer_policy.empty()) {
        // Remove variable if not present before
        ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
    } else {
        // Restore the buffer policy to its original value
        ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, buffer_policy.c_str(), 1 /* overwrite */), 0);
    }
}

TEST(BufferDeathTest, InvalidPolicy) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Store the current buffer policy
    std::string buffer_policy;
    if (const char* buffer_policy_env = getenv(Buffer::buffer_policy_env_name)) {
        buffer_policy = std::string(buffer_policy_env);
    }

    // Overwrite the buffer policy for purpose of the test
    ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, "INVALID_POLICY_TEST", 1 /* overwrite */), 0);

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

// Ignore missing default case in switch statement in test dependency macro
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
        EXPECT_DEATH({ [[maybe_unused]] const auto buffer = Buffer(buffer_size); }, "");
#pragma GCC diagnostic pop
    }

    if (buffer_policy.empty()) {
        // Remove variable if not present before
        ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
    } else {
        // Restore the buffer policy to its original value
        ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, buffer_policy.c_str(), 1 /* overwrite */), 0);
    }
}

#if defined(__linux__) || defined(__APPLE__)
TEST(BufferDeathTest, ProtectUnderflowPolicy) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Store the current buffer policy
    std::string buffer_policy;
    if (const char* buffer_policy_env = getenv(Buffer::buffer_policy_env_name)) {
        buffer_policy = std::string(buffer_policy_env);
    }

    // Overwrite the buffer policy for purpose of the test
    ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, "PROTECT_UNDERFLOW", 1 /* overwrite */), 0);

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
        ASSERT_NE(data, MAP_FAILED);

// Ignore missing default case in switch statement in test dependency macro
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
        EXPECT_EXIT(
            // Underflow by one byte
            { [[maybe_unused]] const volatile auto val = *--data; },
            [](const size_t exit_status) {
                return testing::KilledBySignal(SIGBUS)(exit_status) ||  //
                    testing::KilledBySignal(SIGSEGV)(exit_status) ||    //
                    testing::KilledBySignal(SIGABRT)(exit_status);      //
            },
            "");
#pragma GCC diagnostic pop
    }

    if (buffer_policy.empty()) {
        // Remove variable if not present before
        ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
    } else {
        // Restore the buffer policy to its original value
        ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, buffer_policy.c_str(), 1 /* overwrite */), 0);
    }
}

TEST(BufferDeathTest, ProtectOverflowPolicy) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Store the current buffer policy
    std::string buffer_policy;
    if (const char* buffer_policy_env = getenv(Buffer::buffer_policy_env_name)) {
        buffer_policy = std::string(buffer_policy_env);
    }

    // Overwrite the buffer policy for purpose of the test
    ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, "PROTECT_OVERFLOW", 1 /* overwrite */), 0);

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
        ASSERT_NE(data, MAP_FAILED);

// Ignore missing default case in switch statement in test dependency macro
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
        EXPECT_EXIT(
            // Overflow by one byte
            { [[maybe_unused]] const volatile auto val = *(data + buffer_size); },
            [](const size_t exit_status) {
                return testing::KilledBySignal(SIGBUS)(exit_status) ||  //
                    testing::KilledBySignal(SIGSEGV)(exit_status) ||    //
                    testing::KilledBySignal(SIGABRT)(exit_status);      //
            },
            "");
#pragma GCC diagnostic pop
    }

    if (buffer_policy.empty()) {
        // Remove variable if not present before
        ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
    } else {
        // Restore the buffer policy to its original value
        ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, buffer_policy.c_str(), 1 /* overwrite */), 0);
    }
}
#endif  // if defined(__linux__) || defined(__APPLE__)

}  // namespace kai::test
