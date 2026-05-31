//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <utility>

namespace kai::test {

#define KAI_REGISTER_TEST(test_suite_type, test_type, test_suite_name, test_name, ...) \
    testing::RegisterTest(                                                             \
        test_suite_name, test_name, nullptr, nullptr, __FILE__, __LINE__,              \
        [=]() -> test_suite_type* { return new test_type(__VA_ARGS__); });

/// A facility to register functions that are called before the main function
/// to setup the list of tests.
///
/// Example:
///
/// ```
/// static auto a_register_function = TestRegistry::register_setup([]() {
///     // Setups the list of tests.
/// });
/// ```
class TestRegistry {
public:
    using Fn = std::function<void()>;

private:
    class Handle {
        friend class TestRegistry;

    private:
        explicit Handle(Fn&& fn);
    };

public:
    /// Registers a function to be called in the main function to setup the list of tests.
    ///
    /// The return value of this function must be kept as a static variable.
    ///
    /// @param[in] fn The function to be called.
    [[nodiscard]] static TestRegistry::Handle register_setup(Fn&& fn) {
        return TestRegistry::Handle(std::move(fn));
    }

    /// Runs all functions registered to be called in the main function to setup the list of tests.
    static void init();
};

}  // namespace kai::test
