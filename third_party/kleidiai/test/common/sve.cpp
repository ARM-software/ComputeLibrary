//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/sve.hpp"

#include "kai/kai_common.h"
#include "test/common/cpu_info.hpp"

namespace kai::test {

template <>
size_t get_sve_vector_length<1>() {
    static size_t res = 0;

    if (res == 0) {
        if (cpu_has_sve()) {
            res = kai_get_sve_vector_length_u8();
        } else {
            res = 1;
        }
    }

    return res;
}

template <>
size_t get_sve_vector_length<2>() {
    static size_t res = 0;

    if (res == 0) {
        if (cpu_has_sve()) {
            res = kai_get_sve_vector_length_u16();
        } else {
            res = 1;
        }
    }

    return res;
}

template <>
size_t get_sve_vector_length<4>() {
    static size_t res = 0;

    if (res == 0) {
        if (cpu_has_sve()) {
            res = kai_get_sve_vector_length_u32();
        } else {
            res = 1;
        }
    }

    return res;
}

}  // namespace kai::test
