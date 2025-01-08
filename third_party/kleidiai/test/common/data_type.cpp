//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_type.hpp"

#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"

namespace kai::test {

namespace {

bool has_i(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 15)) != 0;
}

bool has_s(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 14)) != 0;
}

bool has_q(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 13)) != 0;
}

bool has_a(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 12)) != 0;
}

size_t bits(DataType dt) {
    return static_cast<uint16_t>(dt) & 0xFF;
}

}  // namespace

size_t data_type_size_in_bits(DataType dt) {
    return bits(dt);
}

bool data_type_is_integral(DataType dt) {
    return has_i(dt);
}

bool data_type_is_float(DataType dt) {
    KAI_ASSERT(data_type_is_signed(dt));
    return !data_type_is_integral(dt);
}

bool data_type_is_float_fp(DataType dt) {
    KAI_ASSERT(data_type_is_float(dt));
    return !has_q(dt);
}

bool data_type_is_float_bf(DataType dt) {
    KAI_ASSERT(data_type_is_float(dt));
    return has_q(dt);
}

bool data_type_is_signed(DataType dt) {
    if (!has_s(dt)) {
        KAI_ASSERT(data_type_is_integral(dt));
    }

    return has_s(dt);
}

bool data_type_is_quantized(DataType dt) {
    return data_type_is_integral(dt) && has_q(dt);
}

bool data_type_is_quantized_asymm(DataType dt) {
    return data_type_is_quantized(dt) && has_a(dt);
}

}  // namespace kai::test
