//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/memory.hpp"

#include <limits>

#include "test/common/bfloat16.hpp"
#include "test/common/float16.hpp"

namespace kai::test {

double read_array(DataType type, const void* array, size_t index) {
    switch (type) {
        case DataType::FP32:
            return read_array<float>(array, index);
        case DataType::FP16:
            return static_cast<float>(read_array<Float16>(array, index));
        case DataType::BF16:
            return static_cast<float>(read_array<BFloat16<>>(array, index));
        case DataType::I32:
            return read_array<int32_t>(array, index);
        case DataType::QAI8:
        case DataType::QSI8:
            return read_array<int8_t>(array, index);
        case DataType::QSU4:
            return read_array<UInt4>(array, index);
        case DataType::QSI4:
        case DataType::QAI4:
            return read_array<Int4>(array, index);
        case DataType::UNKNOWN:
        default:
            KAI_ERROR("Trying to read unknown data type");
    }
    return std::numeric_limits<double>::signaling_NaN();
}

void write_array(DataType type, void* array, size_t index, double value) {
    switch (type) {
        case DataType::FP32: {
            write_array<float>(array, index, value);
            return;
        }
        case DataType::FP16: {
            write_array<Float16>(array, index, static_cast<Float16>(value));
            return;
        }
        case DataType::BF16: {
            write_array<BFloat16<>>(array, index, static_cast<BFloat16<>>(value));
            return;
        }
        case DataType::I32: {
            write_array<int32_t>(array, index, value);
            return;
        }
        case DataType::QAI8:
        case DataType::QSI8: {
            write_array<int8_t>(array, index, value);
            return;
        }
        case DataType::QSU4: {
            write_array<UInt4>(array, index, static_cast<UInt4>(value));
            return;
        }
        case DataType::QSI4:
        case DataType::QAI4: {
            write_array<Int4>(array, index, static_cast<Int4>(value));
            return;
        }
        case DataType::UNKNOWN:
        default:
            KAI_ERROR("Trying to write unknown data type");
    }
}

}  // namespace kai::test
