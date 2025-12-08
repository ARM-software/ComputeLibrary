//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/common/test_registry.hpp"

#include <utility>
#include <vector>

namespace kai::test {

namespace {

std::vector<TestRegistry::Fn>& call_in_main_fns() {
    static std::vector<TestRegistry::Fn> fns{};

    return fns;
}

}  // namespace

TestRegistry::Handle::Handle(Fn&& fn) {
    call_in_main_fns().emplace_back(std::move(fn));
}

void TestRegistry::init() {
    for (const TestRegistry::Fn& fn : call_in_main_fns()) {
        fn();
    }
}

}  // namespace kai::test
